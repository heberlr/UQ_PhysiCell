import numpy as np
import os, sys, io
import logging
import pickle
import traceback
import signal
import threading
from contextlib import redirect_stdout

# Check if the required libraries are available - futures and mpi4py
try:
    from concurrent.futures import ProcessPoolExecutor
    # Collect results and write to the database
    from concurrent.futures import TimeoutError, as_completed
    futures_available = True
except ImportError:
    futures_available = False
try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False

# My local modules
from uq_physicell import PhysiCell_Model
from .samplers import run_local_sampler, run_global_sampler
from ..utils.model_wrapper import run_replicate, run_replicate_serializable
from ..utils.sumstats import _convert_qoi_function_to_string, recreate_qoi_functions
from ..database.ma_db import create_structure, insert_metadata, insert_param_space, insert_qois, insert_samples, insert_output, check_simulations_db, _disable_wal_mode


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
            for each parameter name and values containing 'ref_value', 'lower_bound',
            'upper_bound', and 'perturbation' information.
        qois_info (dict): Dictionary containing Quantities of Interest definitions.
        qoi_def (dict): first-class object, that can be used in qoi_functions
            lambda string, mapped to their name.
        parallel_method (str, optional): Parallelization method. Options are:
            'inter-process' (single node), 'inter-node' (MPI), or 'serial'.
            Defaults to 'inter-process'.
        num_workers (int, optional): Number of parallel workers for inter-process
            execution. Defaults to 1.
        summary_function (callable, optional): Custom function for summarizing
            simulation output. Defaults to None.
    
    Raises:
        ImportError: If required parallelization libraries are not available.
        ValueError: If invalid parallel_method is specified.
    """
    def __init__(
            self, 
            db_path:str, 
            model_config:dict, 
            sampler:str, 
            params_info:dict, 
            qois_info:dict, 
            qoi_def:dict={},
            parallel_method:str='inter-process', 
            num_workers:int=1, 
            summary_function=None,
            logger: logging.Logger=None):
        self.db_path = db_path
        self.params_dict = params_info

        # Accept model_config as (ini_path, key) tuple or {'ini_path': ..., 'struc_name': ...} dict
        if isinstance(model_config, (tuple, list)):
            model_config = {'ini_path': model_config[0], 'struc_name': model_config[1]}

        # Check free variables before converting to strings — Python's eval() is lazy
        # and only detects missing names at call time, not at lambda-creation time.
        # co_freevars reveals closure references that will fail in worker processes.
        for qoi_name, func in qois_info.items():
            if callable(func) and hasattr(func, '__code__'):
                unresolved = set(func.__code__.co_freevars) - set(qoi_def.keys())
                if unresolved:
                    raise ValueError(
                        f"QoI '{qoi_name}': lambda closes over {sorted(unresolved)} which "
                        f"cannot be serialized for multiprocessing. Pass via "
                        f"qoi_def={{name: object}} in ModelAnalysisContext."
                    )

        # QoI functions are stored as source strings so they can be pickled across processes
        self.qois_dict = {key: _convert_qoi_function_to_string(value, key) if not isinstance(value, (str, type(None))) else value for key, value in qois_info.items()}
        self.qoi_def = qoi_def

        # Secondary check: verify string-form QoIs can be eval'd in the restricted namespace (no None values: particular case with functions defined on python scriipt)
        if self.qois_dict and any(self.qois_dict.values()):
            try:
                recreate_qoi_functions(self.qois_dict, self.qoi_def)
            except Exception as e:
                raise ValueError(
                    f"A QoI function cannot be serialized for multiprocessing. "
                    f"If it closes over an external variable or function, pass it via "
                    f"qoi_def={{name: object}} in ModelAnalysisContext. Details: {e}"
                ) from None

        self.parallel_method = parallel_method
        self.num_workers = num_workers
        self.summary_function = summary_function
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Initialize cancellation flag and process tracking
        self._cancellation_requested = False
        self.futures = []
        self.model = None  # Will be set in run_simulations

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
    
    _GLOBAL_SAMPLERS = frozenset({
        'Sobol', 'Latin hypercube sampling (LHS)', 'Fast',
        'Fractional Factorial', 'Finite Difference',
    })

    def _validate_params_for_sampler(self):
        """Check that params_info contains the fields required by the chosen sampler
        and warn about fields that will be silently ignored."""
        sampler = self.dic_metadata['Sampler']

        if sampler in self._GLOBAL_SAMPLERS:
            for param, props in self.params_dict.items():
                missing = [f for f in ('lower_bound', 'upper_bound')
                           if props.get(f) is None]
                if missing:
                    raise ValueError(
                        f"Parameter '{param}' is missing {missing} required by "
                        f"sampler='{sampler}'. Global samplers use only "
                        f"'lower_bound' and 'upper_bound' for sampling."
                    )
                if props.get('perturbation') is not None:
                    self.logger.warning(
                        f"Parameter '{param}': 'perturbation' is not used by "
                        f"sampler='{sampler}' and will be ignored — only "
                        f"'lower_bound' and 'upper_bound' affect global sampling."
                    )

        elif sampler == 'OAT':
            for param, props in self.params_dict.items():
                if props.get('ref_value') is None:
                    raise ValueError(
                        f"Parameter '{param}' is missing 'ref_value' required by OAT sampling."
                    )
                is_bool = props.get('type') == 'bool'
                if not is_bool and props.get('perturbation') is None:
                    raise ValueError(
                        f"Parameter '{param}' is missing 'perturbation' required by OAT sampling. "
                        f"Provide a list of percentage values, e.g. [1.0, 5.0, 10.0]."
                    )
                for unused in ('lower_bound', 'upper_bound'):
                    if props.get(unused) is not None:
                        self.logger.warning(
                            f"Parameter '{param}': '{unused}' is not used by OAT sampling "
                            f"(only 'ref_value' and 'perturbation' are used) and will be ignored."
                        )

    def generate_samples(self, N: int = None, M: int = 4, seed: int = 42):
        self._validate_params_for_sampler()
        if (self.dic_metadata['Sampler'] == 'OAT'):
            self.dic_samples = run_local_sampler(self.params_dict, self.dic_metadata['Sampler'])
        elif (self.dic_metadata['Sampler'] != 'User-defined'):
            self.dic_samples = run_global_sampler(self.params_dict, self.dic_metadata['Sampler'], N, M, seed)

    def set_samples(self, samples):
        """Set user-defined parameter combinations for the 'User-defined' sampler.

        Args:
            samples: dict mapping integer sample IDs to parameter dicts,
                e.g. {0: {'param1': 1.0, 'param2': 2.0}, 1: {'param1': 1.5, 'param2': 3.0}}
                or a list of parameter dicts (IDs are assigned automatically starting from 0),
                e.g. [{'param1': 1.0}, {'param1': 1.5}]
        """
        if isinstance(samples, list):
            self.dic_samples = {i: s for i, s in enumerate(samples)}
        else:
            self.dic_samples = samples

    def run(self):
        """Run simulations — convenience alias for ``run_simulations(context)``.

        Allows the fluent pattern::

            context.generate_samples(N=8)
            context.run()
        """
        run_simulations(self)

    def cancelled(self):
        """Check if cancellation has been requested.

        Returns:
            bool: True if cancellation was requested, False otherwise
        """
        return self._cancellation_requested
    
    def request_cancellation(self):
        """Request cancellation of all simulations.
        
        This sets the internal cancellation flag to True, which will be checked
        by the simulation process at various points.
        
        Returns:
            bool: Always returns True
        """
        self.logger.info("Cancellation requested")
        self._cancellation_requested = True
        
        # If we have a model instance and it has active processes, terminate them
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'terminate_all_simulations'):
                self.logger.info("Terminating all active simulations...")
                results = self.model.terminate_all_simulations()
                for process_id, return_code in results.items():
                    self.logger.info(f"Process {process_id} terminated with return code {return_code}")
        
        # Cancel futures if they exist
        if self.parallel_method == 'inter-process' and hasattr(self, 'futures'):
            for future in self.futures:
                if not future.done() and not future.cancelled():
                    future.cancel()
                    self.logger.info(f"Cancelled future {future}")
        
        return True

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
    """
    # Only set up signal handlers if we're in the main thread of the main interpreter
    if threading.current_thread() is threading.main_thread():
        def signal_handler(sig, frame):
            context.logger.info(f"Received signal {sig}, initiating graceful shutdown")
            context.request_cancellation()
            # If it's a keyboard interrupt and we're in the main process, exit
            if sig == signal.SIGINT and context.parallel_method != 'inter-process':
                sys.exit(0)
        
        # Register signal handlers
        if context.parallel_method != 'inter-node':  # Don't override MPI's own signal handling
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
    
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
        # Store the model in the context for cancellation support
        context.model = PhysiCellModel
    except Exception as e:
        context.logger.error(f"Error initializing PhysiCell model: {e}")
        raise

    # Initialize or load the database structure
    if rank == 0:
        # Capture PhysiCell model info output and log it
        info_buffer = io.StringIO()
        with redirect_stdout(info_buffer):
            PhysiCellModel.info()
        info_output = info_buffer.getvalue().strip()
        if info_output:
            context.logger.info(f"PhysiCell Model Information:\n{info_output}")
        
        # Check if the db file already exists
        try:
            exist_db, All_Parameters, All_Samples, All_Replicates = check_simulations_db(PhysiCellModel, context.dic_metadata['Sampler'], context.params_dict, context.dic_samples, context.qois_dict, context.db_path)
        except Exception as e: 
            context.logger.error(f"Error checking existing database {context.db_path}: {e}")
            raise
        # Remove the output folder - to avoid overwriting
        if os.path.exists(PhysiCellModel.output_folder):
            os.system('rm -rf ' + PhysiCellModel.output_folder)
        if not exist_db:
            # Initialize database structure
            context.logger.info(f"Creating database structure in {context.db_path}")
            try: create_structure(context.db_path)
            except Exception as e:
                context.logger.error(f"Error creating database structure: {e}")
                raise
            # Insert metadata
            context.logger.info(f"Inserting metadata, parameter space, and QoIs into the database")
            try:
                insert_metadata(context.db_path, context.dic_metadata['Sampler'], context.dic_metadata['IniFilePath'], context.dic_metadata['StrucName'])
                insert_param_space(context.db_path, context.params_dict)
                insert_qois(context.db_path, context.qois_dict)
            except Exception as e:
                # Print traceback for debugging
                traceback.print_exc()
                context.logger.error(f"Error inserting data into the database: {e}")
                raise
            # Populate Samples table
            context.logger.info(f"Inserting samples into the database")
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
        if rank == 0: context.logger.info(f"Generating {len(context.dic_samples)*PhysiCellModel.numReplicates} simulations")
        for sampleID in context.dic_samples.keys():
            for replicateID in np.arange(PhysiCellModel.numReplicates):
                All_Parameters.append(context.dic_samples[sampleID])
                All_Samples.append(sampleID)
                All_Replicates.append(replicateID)
    else:
        # Three lists with size NumSimulations from check_existing_sa
        if rank == 0: context.logger.info(f"Generating {len(All_Samples)} simulations")
    
    ###################################
    # Running using concurrent.futures
    ###################################
    if use_futures:
        # Use concurrent.futures for parallel execution
        with ProcessPoolExecutor(max_workers=context.num_workers) as executor:
            context.futures = []  # Store futures in context for cancellation support
            for ind_sim in range(len(All_Samples)):
                if context.cancelled():
                    context.logger.info("Simulation cancelled before submitting all jobs.")
                    return
                ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
                ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
                model_config = {
                    'ini_path': context.dic_metadata['IniFilePath'],
                    'struc_name': context.dic_metadata['StrucName'],
                }
                # Submit the job to the executor
                context.futures.append(executor.submit(
                    run_replicate_serializable,
                    PhysiCellModel_conf=model_config,
                    sample_id=All_Samples[ind_sim],
                    replicate_id=All_Replicates[ind_sim],
                    ParametersXML=ParametersXML,
                    ParametersRules=ParametersRules,
                    qoi_functions=context.qois_dict,
                    qoi_def=context.qoi_def,
                    return_binary_output=True,
                    #drop_columns,
                    custom_summary_function=context.summary_function,
                ))
             
            # Use as_completed with a short timeout to avoid blocking when cancelled
            remaining_futures = list(context.futures)
            while remaining_futures and not context.cancelled():
                try:
                    # Use a short timeout to check cancellation frequently
                    for future_done in as_completed(remaining_futures, timeout=0.5):
                        remaining_futures.remove(future_done)
                        
                        if context.cancelled():
                            context.logger.info("Simulation cancelled during result collection.")
                            break
                        
                        if future_done.cancelled():
                            context.logger.info("Future was cancelled, skipping result collection.")
                            continue
                        
                        try:
                            sample_id, replicate_id, result_data = future_done.result(timeout=0.5)
                            context.logger.info(f"Writing to the database for Sample: {sample_id}, Replicate: {replicate_id}, Result size: {sys.getsizeof(result_data)/1024:.2f} KB")
                            try:
                                insert_output(context.db_path, sample_id, replicate_id, result_data)
                            except Exception as e:
                                context.logger.error(f"Error writing to the database: {e}")
                        except TimeoutError:
                            # Future is not done yet, will be picked up in the next iteration
                            remaining_futures.append(future_done)
                            context.logger.debug("Future not yet complete, will check again.")
                        except Exception as e:
                            context.logger.error(f"Error retrieving future result: {e}")
                
                except TimeoutError:
                    # No futures completed within timeout, check cancellation and continue
                    if context.cancelled():
                        context.logger.info("Simulation cancelled while waiting for futures to complete.")
                        break
                
                # If cancellation was requested, exit the loop
                if context.cancelled():
                    context.logger.info("Breaking out of future collection loop due to cancellation.")
                    break
            
            # If we cancelled, make sure all futures are cancelled
            if context.cancelled():
                context.logger.info("Cancelling any remaining futures.")
                for future in remaining_futures:
                    if not future.done() and not future.cancelled():
                        future.cancel()
                return
    
    ###################################
    # Running using MPI
    ###################################
    elif use_mpi:
        # Split simulations into ranks
        SplitIndexes = np.array_split(np.arange(len(All_Samples)), size, axis=0)
        context.logger.info(f"Rank {rank} assigned {len(SplitIndexes[rank])} simulations: indices {SplitIndexes[rank].tolist() if len(SplitIndexes[rank]) > 0 else []}")

        # Run simulations (MPI)
        for ind_sim in SplitIndexes[rank]:
            if context.cancelled():
                context.logger.info(f"Rank {rank}: Simulation cancelled.")
                break
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            try:
                if context.summary_function:
                    result_data_nonserialized = PhysiCellModel.RunModel(
                        All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=context.summary_function)
                    result_data = pickle.dumps(result_data_nonserialized)
                else:
                    _, _, result_data = run_replicate(
                        PhysiCellModel=PhysiCellModel,
                        sample_id=All_Samples[ind_sim],
                        replicate_id=All_Replicates[ind_sim],
                        ParametersXML=ParametersXML,
                        ParametersRules=ParametersRules,
                        qoi_functions=context.qois_dict,
                        qoi_def=context.qoi_def,
                        #return_binary_output,
                        #drop_columns,
                        #custom_summary_function,
                    )
 
            except Exception as e:
                context.logger.error(
                    f"Rank {rank}: Error running simulation for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}: {e}"
                )
                continue

            # Write to database with retry logic
            context.logger.info(f"Rank {rank} writing to database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                insert_output(context.db_path, All_Samples[ind_sim], All_Replicates[ind_sim], result_data)
                context.logger.info(f"Rank {rank} successfully wrote results for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            except Exception as e:
                context.logger.error(f"Rank {rank} failed to write results for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}: {e}")
                raise

        comm.Barrier()
        MPI.Finalize()

    ###################################
    # Running sequentially
    ###################################
    else:
        context.logger.info(f"Rank {rank} assigned {len(All_Samples)} simulations.")
        # Run simulations sequentially
        for ind_sim in range(len(All_Samples)):
            if context.cancelled():
                context.logger.info("Sequential simulation cancelled.")
                break
            ParametersXML = {key: All_Parameters[ind_sim][key] for key in params_xml} if params_xml else np.array([])
            ParametersRules = {key: All_Parameters[ind_sim][key] for key in params_rules} if params_rules else np.array([])
            if context.summary_function:
                result_data_nonserialized = PhysiCellModel.RunModel(
                    All_Samples[ind_sim], All_Replicates[ind_sim], ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=context.summary_function)
                result_data = pickle.dumps(result_data_nonserialized)
            else:
                _, _, result_data = run_replicate(
                    PhysiCell_Model=PhysiCellModel,
                    sample_id=All_Samples[ind_sim],
                    replicate_id=All_Replicates[ind_sim],
                    ParametersXML=ParametersXML,
                    ParametersRules=ParametersRules,
                    qoi_functions=context.qois_dict,
                    qoi_def=context.qoi_def,
                    #return_binary_output,
                    #drop_columns,
                    #custom_summary_function,
                )

            # Write to the database directly (no locks or MPI synchronization needed)
            context.logger.info(f"Rank {rank} writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            try:
                insert_output(context.db_path, All_Samples[ind_sim], All_Replicates[ind_sim], pickle.dumps(result_data))
                context.logger.info(f"Rank {rank} finished writing to the database for Sample: {All_Samples[ind_sim]}, Replicate: {All_Replicates[ind_sim]}")
            except Exception as e:
                context.logger.error(f"Error inserting output into the database: {e}")
                raise

    if rank == 0:
        print(f"Simulations completed and results stored in the database: {context.db_path}.")
        # Disable WAL mode of the database to allow reading results without locks
        _disable_wal_mode(context.db_path)
            
if __name__ == "__main__":
    pass
