import numpy as np
import os
import pickle

from uq_physicell import PhysiCell_Model
from uq_physicell.model_analysis.samplers import run_local_sampler
from uq_physicell.utils.model_wrapper import run_replicate, run_replicate_serializable
from .database import create_structure, insert_metadata, insert_param_space, insert_qois, insert_samples, insert_output, check_simulations_db

# Check if the required libraries are available - futures and mpi4py
try:
    from concurrent.futures import ProcessPoolExecutor
    futures_available = True
except ImportError:
    futures_available = False
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

class ModelAnalysisContext:
    """Context manager for running PhysiCell model analysis simulations.
    
    This class manages the configuration, database setup, and execution context
    for running sensitivity analysis and uncertainty quantification simulations
    on PhysiCell models.
    
    Args:
        db_path (str): Path to the SQLite database file for storing results.
        model_config (dict): Dictionary containing PhysiCell model configuration.
            Must include 'ini_path' and 'struc_name' keys.
        sampler (str): Name of the sampling method to use (e.g., 'LHS', 'Sobol', 'OAT').
        params_info (dict): Dictionary containing parameter definitions with keys
            for each parameter name and values containing 'ref_value', 'lower_bounds',
            'upper_bounds', and 'perturbation' information.
        qois_info (dict): Dictionary containing Quantities of Interest definitions.
        parallel_method (str, optional): Parallelization method. Options are:
            'inter-process' (single node), 'inter-node' (MPI), or 'serial'.
            Defaults to 'inter-process'.
        num_workers (int, optional): Number of parallel workers for inter-process
            execution. Defaults to 1.
        summary_function (callable, optional): Custom function for summarizing
            simulation output. Defaults to None.
    
    Attributes:
        db_path (str): Database file path.
        params_dict (dict): Parameter configuration dictionary.
        parallel_method (str): Selected parallelization method.
        qois_dict (dict): Quantities of Interest configuration.
        num_workers (int): Number of parallel workers.
        summary_function (callable): Summary function for output processing.
        dic_metadata (dict): Metadata for database storage.
    
    Raises:
        ImportError: If required parallelization libraries are not available.
        ValueError: If invalid parallel_method is specified.
    
    Example:
        >>> model_config = {
        ...     'ini_path': 'model.xml',
        ...     'struc_name': 'tumor_growth'
        ... }
        >>> params = {
        ...     'param1': {'ref_value': 1.0, 'lower_bounds': 0.5, 'upper_bounds': 1.5}
        ... }
        >>> context = ModelAnalysisContext(
        ...     'analysis.db', model_config, 'LHS', params, {}
        ... )
    """
    def __init__(self, db_path:str, model_config:dict, sampler:str, params_info:dict, qois_info:dict, parallel_method:str='inter-process', num_workers:int=1, summary_function=None):
        self.db_path = db_path
        self.params_dict = params_info  # Dictionary with parameter names, ref value, ranges, and perturbations.
        self.parallel_method = parallel_method # inter-process (single node) or inter-node (mpi)
        self.qois_dict = qois_info
        self.num_workers = num_workers
        self.summary_function = summary_function

        # Initialize metadata for database
        self.dic_metadata = {
            'Sampler': sampler,
            'IniFilePath': model_config['ini_path'],
            'StrucName': model_config['struc_name']
        }

        # Validation of the selected parallelization method
        if self.parallel_method == 'inter-node':
            if not mpi_available:
                raise ImportError("mpi4py is not available. Please install mpi4py or set parallel_method='inter-process'.")
        elif self.parallel_method == 'inter-process':
            if not futures_available:
                raise ImportError("concurrent.futures is not available. Please install futures or set parallel_method='inter-node'.")
        elif self.parallel_method == 'serial':
            self.num_workers = 1
        else:
            raise ValueError("Invalid parallel_method. Use 'inter-node' for MPI, 'inter-process' for futures, or 'serial' for single process.") 

def run_simulations(context: ModelAnalysisContext):
    """Run PhysiCell simulations based on the provided analysis context.
    
    This function executes sensitivity analysis simulations using the specified
    parallelization method (serial, inter-process, or MPI). It manages database
    initialization, parameter sampling, simulation execution, and result storage.
    
    Args:
        context (ModelAnalysisContext): The analysis context containing model
            configuration, sampling parameters, parallelization settings, and
            database information.
    
    Raises:
        ValueError: If there are issues with PhysiCell model initialization,
            database operations, or simulation execution.
        ImportError: If required parallelization libraries are missing.
    
    Note:
        This function handles three execution modes:
        - Serial: Single-threaded execution for small analyses
        - Inter-process: Multi-processing on a single node using concurrent.futures
        - Inter-node: Distributed execution across multiple nodes using MPI
    
    Example:
        >>> context = ModelAnalysisContext(db_path, model_config, 'LHS', params, qois)
        >>> context.dic_samples = run_global_sampler(params, 'LHS', N=100)
        >>> run_simulations(context)
    """
    # Initialize the parallelization method
    if context.parallel_method == 'inter-node':
        use_mpi = True
        use_futures = False
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    elif context.parallel_method == 'inter-process':
        use_mpi = False
        use_futures = True
        rank = 0
    else: #context.parallel_method == 'serial':
        use_mpi = False
        use_futures = False
        rank = 0
    
    # Initialize the PhysiCell model - in all ranks to avoid issues with MPI
    try:
        PhysiCellModel = PhysiCell_Model(context.dic_metadata['IniFilePath'], context.dic_metadata['StrucName'])
    except Exception as e:
        raise ValueError(f"Error initializing PhysiCell model: {e}")

    # Initialize or load the database structure
    if rank == 0:
        PhysiCellModel.info()
        # Check if the sensitivity analysis already exists
        try:
            exist_db, All_Parameters, All_Samples, All_Replicates = check_simulations_db(PhysiCellModel, context.dic_metadata['Sampler'], context.params_dict, context.dic_samples, context.qois_dict, context.db_path)
        except Exception as e: 
            raise ValueError(f"Error checking existing database {context.db_path}: {e}")
        # Remove the output folder - to avoid overwriting
        if os.path.exists(PhysiCellModel.output_folder):
            os.system('rm -rf ' + PhysiCellModel.output_folder)
        if not exist_db:
            # Initialize database structure
            print(f"Creating database structure in {context.db_path}")
            try: create_structure(context.db_path)
            except Exception as e:
                raise ValueError(f"Error creating database structure: {e}")
            # Insert metadata
            print(f"Inserting metadata, parameter space, and QoIs into the database")
            try:
                insert_metadata(context.db_path, context.dic_metadata['Sampler'], context.dic_metadata['IniFilePath'], context.dic_metadata['StrucName'])
                insert_param_space(context.db_path, context.params_dict)
                insert_qois(context.db_path, context.qois_dict)
            except Exception as e:
                raise ValueError(f"Error inserting metadata into the database: {e}")
            # Populate Samples table
            print(f"Inserting samples into the database")
            insert_samples(context.db_path, context.dic_samples)
    else:
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
        if rank == 0: print(f"Generating {len(context.dic_samples)*PhysiCellModel.numReplicates} simulations")
        for sampleID in context.dic_samples.keys():
            for replicateID in np.arange(PhysiCellModel.numReplicates):
                All_Parameters.append(context.dic_samples[sampleID])
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
        with ProcessPoolExecutor(max_workers=context.num_workers) as executor:
            futures = []
            for ind_sim in range(len(All_Samples)):
                ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
                ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
                if context.summary_function:
                    futures.append(executor.submit(
                        run_replicate_serializable, context.dic_metadata['IniFilePath'], context.dic_metadata['StrucName'],
                        All_Samples[ind_sim], All_Replicates[ind_sim],
                        ParametersXML, ParametersRules, custom_summary_function=context.summary_function,
                    ))
                else:
                    futures.append(executor.submit(
                        run_replicate_serializable, context.dic_metadata['IniFilePath'], context.dic_metadata['StrucName'],
                        All_Samples[ind_sim], All_Replicates[ind_sim],
                        ParametersXML, ParametersRules, context.qois_dict if context.qois_dict else None
                    ))

            # Collect results and write to the database
            for future in futures:
                sample_id, replicate_id, result_data = future.result()
                print(f"Writing to the database for Sample: {sample_id}, Replicate: {replicate_id}")
                try:
                    insert_output(context.db_path, sample_id, replicate_id, pickle.dumps(result_data))
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
        for ind_sim in SplitIndexes[rank]:
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            if context.summary_function:
                result_data_nonserialized = PhysiCellModel.RunModel(
                    All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=context.summary_function)
                result_data = pickle.dumps(result_data_nonserialized)
            else:
                _, _, result_data = run_replicate(PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML,
                                            ParametersRules, context.qois_dict if context.qois_dict else None)

            # Token-passing mechanism to ensure one rank writes at a time
            if rank > 0:
                # Wait for the token from the previous rank
                comm.recv(source=rank - 1, tag=0)
            print(f"Rank {rank} writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                insert_output(context.db_path, All_Samples[ind_sim], All_Replicates[ind_sim],  pickle.dumps(result_data))
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
        print(f"Rank {rank} assigned {len(All_Samples)} simulations.")
        # Run simulations sequentially
        for ind_sim in range(len(All_Samples)):
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            if context.summary_function:
                result_data = PhysiCellModel.RunModel(
                    All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=context.summary_function)
            else:
                _, _, result_data = run_replicate(PhysiCellModel, All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML,
                                        ParametersRules, context.qois_dict if context.qois_dict else None)

            # Write to the database directly (no locks or MPI synchronization needed)
            print(f"Rank {rank} writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                insert_output(context.db_path, All_Samples[ind_sim], All_Replicates[ind_sim], pickle.dumps(result_data))
                print(f"Rank {rank} finished writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            except Exception as e:
                raise ValueError(f"Error inserting output into the database: {e}")
            
if __name__ == "__main__":
    # Example usage
    db_path = "examples/virus-mac-new/Simulations_LHS.db"  # Path to the database file
    model_config = {"ini_path": "examples/virus-mac-new/uq_pc_struc.ini", "struc_name": "SA_struc", "numReplicates": 2} # Example model configuration
    sampler = 'Latin hypercube sampling (LHS)'  # Example sampler
    params_info = {
        'mac_phag_rate_infected': {'ref_value': 1.0, 'lower_bounds': 0.5, 'upper_bounds': 1.5, 'perturbation': 50.0},
        'mac_motility_bias': {'ref_value': 0.15, 'lower_bounds': 0.075, 'upper_bounds': 0.225, 'perturbation': 50.0},
        'epi2infected_sat': {'ref_value': 0.1, 'lower_bounds': 0.05, 'upper_bounds': 0.15, 'perturbation': 50.0},
        'epi2infected_hfm': {'ref_value': 0.4, 'lower_bounds': 0.2, 'upper_bounds': 0.6, 'perturbation': 50.0}
    }  # Example parameters information
    qois_info = {}  # Example QoIs information (empty for this example)
    # Create the context for model analysis
    context = ModelAnalysisContext(db_path, model_config, sampler, params_info, qois_info, 
                                   parallel_method='inter-process', num_workers=4)
    # Generate samples using the global sampler
    from .samplers import run_global_sampler
    context.dic_samples = run_global_sampler(context.params_dict, sampler, N=50)  # Generate samples using the global sampler
    # Run the simulations
    run_simulations(context)

    # Local parameter sampling example
    db_path = "examples/virus-mac-new/Simulations_OAT.db"  # Path to the database file
    sampler = 'OAT'  # Example sampler for local sampling
    # Add perturbations to the parameters
    for param_name, properties in params_info.items():
        # The lower and upper bounds are set to None for OAT - Analysis is according to the perturbations
        properties['lower_bounds'] = None
        properties['upper_bounds'] = None
        properties['perturbation'] = [1.0, 5.0, 10.0] # Example perturbations
    context = ModelAnalysisContext(db_path, model_config, sampler, params_info, qois_info, 
                                   parallel_method='inter-process', num_workers=4)
    # Generate samples using the local sampler
    from .samplers import run_local_sampler
    context.dic_samples = run_local_sampler(context.params_dict)
    # Run the simulations with the local sampler
    # run_simulations(context)