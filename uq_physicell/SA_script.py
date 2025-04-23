from mpi4py import MPI
import numpy as np
import os
import sqlite3
import pickle
from concurrent.futures import ProcessPoolExecutor

from uq_physicell import PhysiCell_Model
from uq_physicell.SA_utils import run_replicate, create_db_structure, insert_metadata, insert_inputs, insert_output, check_existing_sa, summary_function, create_named_function_from_string, load_db_structure

def run_replicate_serializable(ini_path, struc_name, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic=None, drop_columns=[], custom_summary_function=None):
    """
    Run a single replicate of the PhysiCell model and return the results.

    Parameters:
    - ini_path: Path to the initialization file.
    - struc_name: Structure name.
    - sampleID: Sample ID.
    - replicateID: Replicate ID.
    - ParametersXML: Dictionary of XML parameters.
    - ParametersRules: Dictionary of rules parameters.
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings).
    - drop_columns: List of columns to drop from the output.
    - custom_summary_function: Custom summary function to use instead of the default generic QoI function.

    Returns:
    - sampleID, replicateID, result_data
    """
    try:
        PhysiCellModel = PhysiCell_Model(ini_path, struc_name)
        if custom_summary_function:
            result_data_nonserialized = PhysiCellModel.runModel(
                sampleID, replicateID, ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=custom_summary_function)
            result_data = pickle.dumps(result_data_nonserialized)
        else:
            _, _, result_data = run_replicate(PhysiCellModel, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic, drop_columns)
        return sampleID, replicateID, result_data
    except Exception as e:
        raise RuntimeError(f"Error running replicate: {e}")


def run_sa_simulations(ini_filePath, strucName, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, dic_samples, qois_dic, db_file, use_mpi=False, use_futures=False, num_workers=1, drop_columns=[], custom_summary_function=None):
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
    - dic_samples: dictionary of the dictionaries of samples
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings) - If empty store all data as a list of mcds.
    - db_file: Path to the database file.
    - use_mpi: Whether to use MPI for parallelism.
    - use_futures: Whether to use concurrent.futures for parallelism.
    - num_workers: Number of workers for concurrent.futures.
    - drop_columns: List of columns to drop from the output (optional, it works when qoi_dic is empty).
    - custom_summary_function: Function to summarize the results (optional, if defined it ignores qoi_dic and drop_columns).
    """
    if use_mpi:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 1
        rank = 0


    # Initialize the PhysiCell model - in all ranks to avoid issues with MPI
    try:
        PhysiCellModel = PhysiCell_Model(ini_filePath, strucName)
    except Exception as e:
        raise ValueError(f"Error initializing PhysiCell model: {e}")

    # Initialize the PhysiCell model
    if rank == 0:
        PhysiCellModel.info()
        # Check if the sensitivity analysis already exists
        try:
            exist_db, All_Parameters, All_Samples, All_Replicates = check_existing_sa(PhysiCellModel, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, dic_samples, qois_dic, db_file)
        except Exception as e: 
            raise ValueError(f"Error checking existing sensitivity analysis: {e}")
        # Remove the output folder - to avoid overwriting
        if os.path.exists(PhysiCellModel.output_folder):
            os.system('rm -rf ' + PhysiCellModel.output_folder)
        if not exist_db:
            # Initialize database structure
            print(f"Creating database structure in {db_file}")
            try: create_db_structure(db_file)
            except Exception as e:
                raise ValueError(f"Error creating database structure: {e}")
            # Load the database structure
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            # Insert metadata
            print(f"Inserting metadata into the database")
            try:
                insert_metadata(cursor, SA_type, SA_method, SA_sampler, len(dic_samples), param_names, bounds, ref_values, perturbations, qois_dic, ini_filePath, strucName)
            except Exception as e:
                raise ValueError(f"Error inserting metadata into the database: {e}")
            # Populate Inputs table
            print(f"Inserting input parameters into the database")
            insert_inputs(cursor, dic_samples)
            conn.commit()
            conn.close()
    else:
        conn = None
        exist_db = None
        All_Samples = None
        All_Replicates = None
        All_Parameters = None

    if use_mpi:
        exist_db = comm.bcast(exist_db, root=0)
        All_Samples = comm.bcast(All_Samples, root=0)
        All_Replicates = comm.bcast(All_Replicates, root=0)
        All_Parameters = comm.bcast(All_Parameters, root=0)

    # Number of parameters expected in the XML and rules
    params_xml = [param_name for param_name in PhysiCellModel.XML_parameters_variable.values()]
    params_rules = [param_name for param_name in PhysiCellModel.parameters_rules_variable.values()]

    # Generate a three list with size NumSimulations
    if not exist_db:
        if rank == 0: print(f"Generating {len(dic_samples)*PhysiCellModel.numReplicates} simulations")
        for sampleID in dic_samples.keys():
            for replicateID in np.arange(PhysiCellModel.numReplicates):
                All_Parameters.append(dic_samples[sampleID])
                All_Samples.append(sampleID)
                All_Replicates.append(replicateID)
    else:
        # Three lists with size NumSimulations from check_existing_sa
        if rank == 0: print(f"Generating {len(All_Samples)} simulations")
    
    ###################################
    # Running using concurrent.futures
    ###################################
    if use_futures:
        # Use concurrent.futures for parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for ind_sim in range(len(All_Samples)):
                ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
                ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
                if summary_function:
                    futures.append(executor.submit(
                        run_replicate_serializable, ini_filePath, strucName,
                        All_Samples[ind_sim], All_Replicates[ind_sim],
                        ParametersXML, ParametersRules, custom_summary_function=custom_summary_function,
                    ))
                else:
                    futures.append(executor.submit(
                        run_replicate_serializable, ini_filePath, strucName,
                        All_Samples[ind_sim], All_Replicates[ind_sim],
                        ParametersXML, ParametersRules, qois_dic if qois_dic else None, drop_columns=drop_columns
                    ))

            # Collect results and write to the database
            for future in futures:
                sample_id, replicate_id, result_data = future.result()
                print(f"Writing to the database for Sample: {sample_id}, Replicate: {replicate_id}")
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    insert_output(cursor, sample_id, replicate_id, result_data)
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"Error writing to the database: {e}")
    
    ###################################
    # Running using MPI
    ###################################
    elif use_mpi:
        # Split simulations into ranks
        SplitIndexes = np.array_split(np.arange(len(All_Samples)), size, axis=0)
        print(f"Rank {rank} assigned {len(SplitIndexes[rank])} simulations.")

        # Run simulations (MPI)
        for round_id, ind_sim in enumerate(SplitIndexes[rank]):
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            if custom_summary_function:
                result_data_nonserialized = PhysiCellModel.runModel(
                    All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=custom_summary_function)
                result_data = pickle.dumps(result_data_nonserialized)
            else:
                _, _, result_data = run_replicate(PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML,
                                            ParametersRules, qois_dic if qois_dic else None, drop_columns=drop_columns)

            # Token-passing mechanism to ensure one rank writes at a time
            if rank > 0:
                # Wait for the token from the previous rank
                comm.recv(source=rank - 1, tag=0)
            print(f"Rank {rank} writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                insert_output(cursor, All_Samples[ind_sim], All_Replicates[ind_sim], result_data)
                conn.commit()
                conn.close()
                print(f"Rank {rank} finished writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            except Exception as e:
                print(f"Rank {rank}: Error writing to the database: {e}")

            # Pass the token to the next rank
            if rank < size - 1:
                comm.send(None, dest=rank + 1, tag=0)
        
        comm.Barrier()
        MPI.Finalize()

    ###################################
    # Running sequentially
    ###################################
    else:
        # Split simulations into rank 0
        SplitIndexes = np.array_split(np.arange(len(All_Samples)), size, axis=0)
        print(f"Rank {rank} assigned {len(SplitIndexes[rank])} simulations.")

        # Run simulations sequentially
        for round_id, ind_sim in enumerate(SplitIndexes[rank]):
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            if custom_summary_function:
                result_data_nonserialized = PhysiCellModel.runModel(
                    All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=custom_summary_function)
                result_data = pickle.dumps(result_data_nonserialized)
            else:
                _, _, result_data = run_replicate(PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML,
                                        ParametersRules, qois_dic if qois_dic else None, drop_columns=drop_columns)

            # Write to the database directly (no locks or MPI synchronization needed)
            print(f"Rank {rank} writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                insert_output(cursor, All_Samples[ind_sim], All_Replicates[ind_sim], result_data)
                conn.commit()
                conn.close()
                print(f"Rank {rank} finished writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            except Exception as e:
                raise ValueError(f"Error inserting output into the database: {e}")