from mpi4py import MPI
import numpy as np
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor

from uq_physicell import PhysiCell_Model
from uq_physicell.SA_utils import run_replicate, create_db_structure, insert_metadata, insert_inputs, insert_output, check_existing_sa, summary_function, create_named_function_from_string, load_db_structure

def run_sa_simulations(ini_filePath, strucName, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, samples, qois_dic, db_file, use_mpi=False, use_futures=False, num_workers=1):
    """
    Parameters:
    - ini_filePath: Path to the initialization file.
    - strucName: Structure name.
    - SA_type: Sensitivity analysis type.
    - SA_method: Sensitivity analysis method.
    - SA_sampler: Sampling method.
    - param_names: List of parameter names.
    - ref_values: List of reference values for the parameters.
    - bounds: List of bounds for the parameters.
    - perturbations: List of perturbations for the parameters.
    - samples: Array of sample inputs.
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings) - If empty store all data as a list of mcds.
    - db_file: Path to the database file.
    - use_mpi: Whether to use MPI for parallelism.
    - use_futures: Whether to use concurrent.futures for parallelism.
    """
    if use_mpi:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 1
        rank = 0

    # Initialize the PhysiCell model
    if rank == 0:
        try:
            PhysiCellModel = PhysiCell_Model(ini_filePath, strucName)
        except Exception as e:
            raise ValueError(f"Error initializing PhysiCell model: {e}")
            return
        PhysiCellModel.info()
        # Check if the sensitivity analysis already exists
        try:
            exist_db, All_Parameters, All_Samples, All_Replicates = check_existing_sa(PhysiCellModel, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, samples, qois_dic, db_file)
        except Exception as e: 
            raise ValueError("Sensitivity analysis with the same configuration already exists in the database.")
            return
        # Remove the output folder - to avoid overwriting
        if os.path.exists(PhysiCellModel.output_folder):
            os.system('rm -rf ' + PhysiCellModel.output_folder)
        if not exist_db:
            # Initialize database structure
            print(f"Creating database structure in {db_file}")
            create_db_structure(db_file)
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            # Insert metadata
            print(f"Inserting metadata into the database")
            try:
                insert_metadata(cursor, SA_type, SA_method, SA_sampler, len(samples), param_names, bounds, ref_values, perturbations, qois_dic, ini_filePath, strucName)
            except Exception as e:
                print(f"Error inserting metadata into the database: {e}")
                return
            # Populate Inputs table
            print(f"Inserting input parameters into the database")
            insert_inputs(cursor, samples, PhysiCellModel)
            conn.commit()
    else:
        PhysiCellModel = None
        conn = None
        exist_db = None
        All_Samples = None
        All_Replicates = None
        All_Parameters = None

    if use_mpi:
        PhysiCellModel = comm.bcast(PhysiCellModel, root=0)
        exist_db = comm.bcast(exist_db, root=0)
        All_Samples = comm.bcast(All_Samples, root=0)
        All_Replicates = comm.bcast(All_Replicates, root=0)
        All_Parameters = comm.bcast(All_Parameters, root=0)

    # Number of parameters expected in the XML and rules
    num_params_xml = len(PhysiCellModel.XML_parameters_variable)
    num_params_rules = len(PhysiCellModel.parameters_rules_variable)

    # Generate a three list with size NumSimulations
    if not exist_db:
        if rank == 0: print(f"Generating {samples.shape[0]*PhysiCellModel.numReplicates} simulations")
        for sampleID in range(samples.shape[0]):
            for replicateID in np.arange(PhysiCellModel.numReplicates):
                All_Parameters.append(samples[sampleID])
                All_Samples.append(sampleID)
                All_Replicates.append(replicateID)
    else:
        # Three lists with size NumSimulations from check_existing_sa
        if rank == 0: print(f"Generating {len(All_Samples)} simulations")

    if use_futures and not use_mpi:
        # Use concurrent.futures for parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ind_sim in range(len(All_Samples)):
                ParametersXML = All_Parameters[ind_sim][:num_params_xml] if num_params_xml > 0 else np.array([])
                ParametersRules = All_Parameters[ind_sim][num_params_xml:] if num_params_rules > 0 else np.array([])
                print(f"Submitting simulation: {ind_sim}, Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}, Parameters: {All_Parameters[ind_sim]}")
                # Pass the serialized QoI functions to each process
                futures.append(executor.submit(
                    run_replicate,
                    PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim],
                    ParametersXML, ParametersRules, qois_dic if qois_dic else None
                ))

            for future in futures:
                sample_id, replicate_id, result_data = future.result()
                if rank == 0:
                    insert_output(cursor, sample_id, replicate_id, result_data)
                    conn.commit()
    else:
        # Split simulations into ranks
        SplitIndexes = np.array_split(np.arange(len(All_Samples)), size, axis=0)
        if rank == 0:
            print(f"Total number of simulations: {len(All_Samples)} Simulations per rank: {len(SplitIndexes[0])}")
        # Run simulations (MPI or sequential)
        for ind_sim in SplitIndexes[rank]:
            ParametersXML = All_Parameters[ind_sim][:num_params_xml] if num_params_xml > 0 else np.array([])
            ParametersRules = All_Parameters[ind_sim][num_params_xml:] if num_params_rules > 0 else np.array([])
            print(f'Simulation: {ind_sim}, Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}, '
                  f'Parameters XML: {ParametersXML}, Parameters rules: {ParametersRules}')
            result_data = run_replicate(PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML,
                                                ParametersRules, qois_dic if qois_dic else None)
            if use_mpi:
                if rank == 0:
                    # Receive results from other ranks
                    for _ in range(1, size):
                        received_data = comm.recv(source=MPI.ANY_SOURCE)
                        insert_output(cursor, received_data['SampleID'], received_data['ReplicateID'], received_data['Data'])
                        conn.commit()
                else:
                    # Send results to rank 0
                    comm.send({'SampleID': All_Samples[ind_sim], 'ReplicateID': All_Replicates[ind_sim], 'Data': result_data}, dest=0)
            else:
                # Sequential case: directly insert into the database
                if rank == 0:
                    insert_output(cursor, All_Samples[ind_sim], All_Replicates[ind_sim], result_data)
                    conn.commit()

    if rank == 0:
        conn.close()

    if use_mpi:
        MPI.Finalize()
