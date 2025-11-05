import os
import sys
import logging
import concurrent.futures
from typing import Union, Optional, Dict, List, Callable, Any
import gc

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# pyABC imports
from pyabc import ABCSMC, sampler, LocalTransition, AdaptiveAggregatedDistance, AggregatedDistance, QuantileEpsilon, History, RV, Distribution
from pyabc.populationstrategy import AdaptivePopulationSize
from pyabc.storage import load_dict_from_json
from dask.distributed import Client, get_worker

# UQ PhysiCell imports
from uq_physicell import PhysiCell_Model
from uq_physicell.utils import run_replicate_serializable


class CalibrationContext:
    """
    Context for Approximate Bayesian Computation (ABC) calibration using pyABC.

    This class encapsulates all necessary parameters and configurations for model calibration
    using ABC-SMC with sophisticated handling of multiple models, parallel computation,
    and adaptive strategies.

    Attributes:
        db_path (str): Path to the database file for storing and retrieving calibration results.
        obsData (str or dict): Path to observed data CSV file or dictionary containing observed data.
        obsData_columns (dict): Dictionary mapping QoI names to their corresponding columns in the observed data.
        model_config (dict): Configuration dictionary for the PhysiCell model, including paths and structure names.
        qoi_functions (dict): Dictionary of functions to compute quantities of interest (QoIs) from model outputs.
        distance_functions (dict): Dictionary of distance functions with their weights for comparing model outputs to observed data.
        prior (Distribution): Distribution defining the prior distributions for parameters
        abc_options (dict): Options for ABC-SMC including population parameters, sampling strategies, and convergence criteria.
        logger (logging.Logger): Logger instance for logging messages during the calibration process.
    """
    
    def __init__(
        self,
        db_path: str, 
        obsData: Union[str, dict], 
        obsData_columns: dict, 
        model_config: dict, 
        qoi_functions: dict, 
        distance_functions: dict, 
        prior: Distribution, 
        abc_options: dict, 
        logger: Optional[logging.Logger] = None
    ):
        """Initialize CalibrationContext with comprehensive validation and setup."""
        # Core configuration
        self.db_path = db_path
        self.model_config = model_config
        self.qoi_functions = qoi_functions
        self.distance_functions = distance_functions
        self.prior = prior
        self.abc_options = abc_options
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
        else:
            self.logger = logger

        # Load and validate observed data
        if isinstance(obsData, dict):
            self.dic_obsData = obsData
            self.obsData_path = None
        else:  # obsData is a path
            try:
                self.obsData_path = obsData
                self.dic_obsData = pd.read_csv(obsData).to_dict('list')
                # Replace column names according to obsData_columns mapping
                for qoi, column_name in obsData_columns.items():
                    if column_name in self.dic_obsData:
                        self.dic_obsData[qoi] = np.array(self.dic_obsData.pop(column_name), dtype=np.float64)
                    else:
                        raise ValueError(f"Column '{column_name}' not found in observed data.")
                self.logger.debug(f"Successfully loaded observed data from {obsData}")
            except Exception as e:
                self.logger.error(f"Error reading observed data from {obsData}: {e}")
                raise

        # ABC-SMC configuration
        self.max_populations = abc_options.get("max_populations", 20)
        self.max_simulations = abc_options.get("max_simulations", 1000)
        self.population_strategy = abc_options.get("population_strategy", "adaptive")
        self.min_population_size = abc_options.get("min_population_size", 100)
        self.max_population_size = abc_options.get("max_population_size", 500)
        self.epsilon_strategy = abc_options.get("epsilon_strategy", "quantile")
        self.epsilon_alpha = abc_options.get("epsilon_alpha", 0.4)
        
        # Transition and distance configuration
        self.transition_strategy = abc_options.get("transition_strategy", "multivariate")  # "local" or "multivariate"
        self.adaptive_distance = abc_options.get("adaptive_distance", False)
        self.adaptive_distance_file = abc_options.get("adaptive_distance_file", None)
        self.convergence_check_func = abc_options.get("convergence_check_func", None)
        
        # Sampling configuration
        self.sampler_type = abc_options.get("sampler", "multicore")  # "dask" or "multicore"
        self.cluster_setup_func = abc_options.get("cluster_setup_func", None) # Function to setup Dask cluster
        self.num_workers = abc_options.get("num_workers", os.cpu_count())
        self.mode = abc_options.get("mode", "cluster")  # "cluster" or "local"
        
        # Model configuration
        self.num_replicates = self.model_config.get('numReplicates', 10)
        self.fixed_params = abc_options.get('fixed_params', {})
        self.summary_function = abc_options.get("summary_function", None)
        self.aggregation_func = abc_options.get("custom_aggregation_func", self._default_aggregation_func)
        self.custom_run_single_replicate_func = abc_options.get("custom_run_single_replicate_func", None)
        
        
        # Parameter scaling
        self.log_scale = abc_options.get("log_scale", False)
        
        # Multiple models support
        self.num_models = abc_options.get("num_models", 1)
        self.model_selection = abc_options.get("model_selection", False)
        
        # Parallelization setup
        self._setup_parallelization()
        
        # Validate configuration
        self._validate_configuration()
        
        self.logger.info(f"🔧 CalibrationContext initialized for ABC-SMC calibration")
        self.logger.info(f"📊 Database: {self.db_path}")
        self.logger.info(f"🎯 QoIs: {list(self.qoi_functions.keys())}")
        self.logger.info(f"🔍 Parameters: {self.prior.get_parameter_names()}")
        self.logger.info(f"⚙️ Sampler: {self.sampler_type} with {self.num_workers} workers")

    def _setup_parallelization(self):
        """Setup parallelization strategy based on configuration."""
        if self.mode == "cluster":
            if self.sampler_type == "dask":
                self.workers_inner = None  # Dask handles its own parallelization
                self.workers_outer = self.num_workers
            elif self.sampler_type == "multicore":
                # Calculate nested parallelization for multicore
                self.workers_inner = min(self.num_workers, self.num_replicates)  # workers for replicates
                self.workers_outer = max(1, self.num_workers // self.workers_inner)  # workers for parameter sets
            else:
                raise ValueError(f"Sampler {self.sampler_type} is not supported. Use 'dask' or 'multicore'.")
        else:  # local mode
            self.num_replicates = min(self.num_replicates, 2)  # Reduce replicates for local testing
            if self.sampler_type == "dask":
                self.workers_inner = None
                self.workers_outer = min(self.num_workers, 4)  # Limit for local testing
            elif self.sampler_type == "multicore":
                self.workers_inner = min(self.num_workers, self.num_replicates)
                self.workers_outer = max(1, self.num_workers // self.workers_inner)
            else:
                raise ValueError(f"Sampler {self.sampler_type} is not supported. Use 'dask' or 'multicore'.")

    def _validate_configuration(self):
        """Validate the configuration parameters."""
        required_model_keys = ['ini_path', 'struc_name']
        for key in required_model_keys:
            if key not in self.model_config:
                raise ValueError(f"Missing required model_config key: {key}")
        
        if not self.qoi_functions:
            raise ValueError("qoi_functions cannot be empty")
        
        if not self.distance_functions:
            raise ValueError("distance_functions cannot be empty")
        
        if not self.prior:
            raise ValueError("prior cannot be empty")

        # Validate QoI consistency
        for qoi in self.qoi_functions.keys():
            if qoi not in self.distance_functions:
                raise ValueError(f"Distance function not defined for QoI: {qoi}")

    def setup_sampler(self, cluster_setup_func=None):
        """Setup the pyABC sampler based on configuration."""
        if self.sampler_type == "dask":
            if self.mode == "cluster":
                if cluster_setup_func is None:
                    raise ValueError("cluster_setup_func must be provided for cluster mode with Dask")
                cluster = cluster_setup_func()
                my_sampler = sampler.DaskDistributedSampler(Client(cluster))
                self.logger.info(f"Using Dask sampler with {self.num_workers} workers")
        elif self.sampler_type == "multicore":
            my_sampler = sampler.MulticoreParticleParallelSampler(n_procs=self.workers_outer)
            if self.mode == "cluster":
                self.logger.info(f"Using nested multicore parallelization: {self.workers_outer} outer processes × {self.workers_inner} inner threads = {self.workers_outer * self.workers_inner} total")
            else:
                self.logger.info(f"Using local nested multicore parallelization: {self.workers_outer} outer processes × {self.workers_inner} inner threads = {self.workers_outer * self.workers_inner} total")
        else:
            raise ValueError(f"Sampler {self.sampler_type} is not supported.")
        
        return my_sampler

    def setup_population_strategy(self):
        """Setup population size strategy."""
        if self.population_strategy == "adaptive":
            return AdaptivePopulationSize(
                start_nr_particles=self.max_population_size,
                mean_cv=0.2,
                min_population_size=self.min_population_size,
                max_population_size=self.max_population_size
            )
        else:
            # Fixed population size (useful for local testing)
            return self.min_population_size if self.mode == "local" else self.max_population_size

    def setup_distance_function(self, distances_dict):
        qois = list(self.qoi_functions.keys())
        distance_funcs = [distance_func["function"] for distance_func in distances_dict.values()]
        distance_weights = [distance_func.get("weight", 1.0) for distance_func in distances_dict.values()]
        """Setup distance function with optional adaptive weighting."""
        if len(qois) > 1:
            if self.adaptive_distance:
                # If adaptive distance, try to load previous weights if database exists
                if os.path.exists(self.db_path) and hasattr(self, '_load_adaptive_weights'):
                    try:
                        distance_weights = self._load_adaptive_weights(self.db_path)
                    except Exception as e:
                        raise ValueError(f"Could not load adaptive weights: {e}")
                    # Starting with loaded adaptive weights.
                    distance_func = AdaptiveAggregatedDistance(distance_funcs, initial_weights=distance_weights, log_file=self.adaptive_distance_file)
                else:
                    # Start with adaptive weighting
                    distance_func = AdaptiveAggregatedDistance(distance_funcs, log_file=self.adaptive_distance_file)
                    self.logger.info("Starting with adaptive distance weighting")
            else:
                # Use provided weights or equal weights
                distance_func = AggregatedDistance(distance_funcs, weights=distance_weights)
                self.logger.info(f"Using fixed distance weights: {distance_weights}")
        else:
            distance_func = distance_funcs[0]
            self.logger.info(f"Using single distance function: {distances_dict[qois[0]]['function']}")

        return distance_func

    def setup_transition_function(self):
        """Setup transition function for ABC-SMC."""
        if self.transition_strategy == "local":
            return LocalTransition()
        else:
            return None  # Default - multivariate normal transitions

    def setup_epsilon_function(self):
        """Setup epsilon (tolerance) function for ABC-SMC."""
        if self.epsilon_strategy == "quantile":
            return QuantileEpsilon(alpha=self.epsilon_alpha)
        else:
            raise ValueError(f"Epsilon strategy '{self.epsilon_strategy}' not supported")

    def create_model_wrapper(self, fixed_params_dict, workers_inner=None):
        """Create wrapper function for PhysiCell model evaluation."""
        def model_wrapper(pars):
            return self._run_physicell_model(pars, fixed_params_dict, workers_inner)
        return model_wrapper

    def _run_physicell_model(self, pars, fixed_params, workers_inner=None):
        """Run PhysiCell model with given parameters."""
        try:
            # Convert parameters from log scale if needed
            if self.log_scale and hasattr(self, '_convert_params_to_linear_scale'):
                pars = self._convert_params_to_linear_scale(pars)
            
            # Choose parallelization strategy based on sampler
            if self.sampler_type == 'multicore' and workers_inner is not None:
                return self._run_replicates_parallel(workers_inner, pars, fixed_params)
            else:
                return self._run_physicell_model_sequential(pars, fixed_params)
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {e}")
            raise ValueError(f"Error in model evaluation: {e}")
        
    def _default_aggregation_func(self, replicate_results):
        """Define function to aggregate the replicates"""
        try:
            results_df = pd.concat(list(replicate_results.values()), ignore_index=True)
            # Take the mean of all columns in the same sampleID and time
            return results_df.pivot_table(index=['sampleID','time'])
        except Exception as e:
            raise ValueError(f"Error in _default_aggregation_func for sampleID: {replicate_results.values()[0]['sampleID'].unique()}")

    def _run_physicell_model_sequential(self, pars, fixed_params, sample_id=None, replicate_id=None):
        """Run PhysiCell model sequentially."""
        # Create the PhysiCell model instance for each worker
        physicell_model = PhysiCell_Model(self.model_config['ini_path'], self.model_config['struc_name'])
        physicell_model.numReplicates = self.num_replicates
        
        # Configure input/output folders
        if "input_folder" in self.model_config:
            physicell_model.input_folder += self.model_config['input_folder']
        if "output_folder" in self.model_config:
            physicell_model.output_folder += self.model_config['output_folder']
        
        physicell_model.timeout = 600
        physicell_model.output_summary_Path = None

        # Get parameter names
        params_xml = [param_name for param_name in physicell_model.XML_parameters_variable.values()]
        params_rules = [param_name for param_name in physicell_model.parameters_rules_variable.values()]

        # Get worker ID
        if sample_id is None:
            sample_id = self._get_worker_id()

        # Prepare parameters
        dic_pars_xml = {par: (pars[par] if par in pars.keys() else None) for par in params_xml}
        dic_pars_rules = {par: (pars[par] if par in pars.keys() else None) for par in params_rules}

        # Include fixed parameters
        for par in fixed_params.keys():
            if par in dic_pars_xml.keys():
                dic_pars_xml[par] = fixed_params[par]
            if par in dic_pars_rules.keys():
                dic_pars_rules[par] = fixed_params[par]

        # Validation
        if None in dic_pars_xml.values() or None in dic_pars_rules.values():
            raise ValueError(f"Some parameters are None: {dic_pars_xml}, {dic_pars_rules}")

        # Run replicates
        replicates = range(self.num_replicates) if replicate_id is None else [replicate_id]
        
        dic_all_replicates = {}
        for replicate_id in replicates:
            try:
                _, _, result_data = run_replicate_serializable(self.model_config, sample_id, replicate_id, dic_pars_xml, dic_pars_rules, qoi_functions=self.qoi_functions, return_binary_output=False, custom_summary_function=self.summary_function)
                dic_all_replicates[replicate_id] = result_data
            except Exception as e:
                raise RuntimeError(f"Error in RunModel (SampleID: {sample_id}): {e}")
            
            # Check if RunModel returned valid data
            if not hasattr(result_data, 'columns') or len(result_data) == 0:
                raise RuntimeError(f"RunModel returned empty or invalid DataFrame for SampleID: {sample_id}, ReplicateID: {replicate_id}")
        # All replicates done, run aggregation function
        if replicate_id is None:
            return self.aggregation_func(dic_all_replicates)
        else:
            return dic_all_replicates

    def _run_replicates_parallel(self, workers_inner, params, fixed_params):
        """Run replicates in parallel using ThreadPoolExecutor."""
        
        with ThreadPoolExecutor(max_workers=workers_inner) as executor:
            futures = []
            for replicate_id in range(self.num_replicates):
                future = executor.submit(
                    self._run_physicell_model_sequential,
                    params, fixed_params, self._get_worker_id(), replicate_id
                )
                futures.append(future)
            
            dic_all_replicates = {}
            for future_done in as_completed(futures):
                dict_result = future_done.result(timeout=0.5)
                for key, value in dict_result.items():
                    dic_all_replicates[key] = value

        return self.aggregation_func(dic_all_replicates)

    def _get_worker_id(self):
        """Get worker ID for distributed computing."""
        try:
            worker_id = int(get_worker().name)
        except:
            try:
                worker_id = int(get_worker().name.split("-")[-1])
            except:
                worker_id = os.getpid()
        return worker_id

    def setup_abc_smc(self, models_list, priors_list, distance_function, population_size, transitions_func, my_sampler, eps_function):
        """Setup the ABC-SMC object."""
        return ABCSMC(
            models=models_list,
            parameter_priors=priors_list,
            distance_function=distance_function,
            population_size=population_size,
            transitions=transitions_func,
            sampler=my_sampler,
            eps=eps_function
        )

    def load_or_create_database(self, abc_smc, abc_id=1):
        """Load existing database or create new one."""
        db_file = "sqlite:///" + os.path.join(self.db_path)
        
        if os.path.exists(self.db_path):
            try:
                abc_smc.load(db_file, abc_id=abc_id)
                self.logger.info(f"Loaded existing database: {self.db_path}")
                return True, abc_smc.history.n_populations, abc_smc.history.total_nr_simulations
            except ValueError as e:
                self.logger.error(f"Error loading database {db_file}: {e}")
                raise
        else:
            abc_smc.new(db_file, observed_sum_stat=self.dic_obsData)
            self.logger.info(f"Created new database: {self.db_path}")
            return False, 0, 0

    def run_calibration(self, abc_smc, resume_db=False, current_populations=0, current_simulations=0):
        """Run the ABC-SMC calibration."""
        if resume_db:
            extra_populations = max(0, self.max_populations - current_populations)
            extra_simulations = max(0, self.max_simulations - current_simulations)
            self.logger.info(f"Resuming: extra populations: {extra_populations}, extra simulations: {extra_simulations}")
            
            if extra_populations > 0 and extra_simulations > 0:
                abc_smc.run(max_nr_populations=self.max_populations, max_total_nr_simulations=self.max_simulations)
            else:
                self.logger.info("No additional calibration needed")
        else:
            self.logger.info(f"Starting calibration: max populations: {self.max_populations}, max simulations: {self.max_simulations}")
            abc_smc.run(max_nr_populations=self.max_populations, max_total_nr_simulations=self.max_simulations)

    def check_convergence(self, abc_smc):
        """Check convergence criteria."""
        # This would need the check_convergence_generic function
        # For now, return False to continue until max iterations
        return False

    def include_additional_metadata(self, **metadata):
        """Include additional metadata in the database."""
        # This would need the include_additional_data_in_db function
        # For now, just log the metadata
        self.logger.info(f"Additional metadata: {metadata}")


def run_abc_calibration( calib_context: CalibrationContext) -> History:
    """
    Execute the complete ABC-SMC calibration process.
    
    This function orchestrates the entire ABC-SMC workflow using the CalibrationContext,
    including sampler setup, distance function configuration, model wrapper creation,
    and calibration execution with convergence checking.
    
    Args:
        calib_context (CalibrationContext): The calibration context containing all configuration
    
    Returns:
        History: The pyABC History object containing calibration results
    """
    logger = calib_context.logger
    
    try:
        logger.info("🚀 Starting ABC-SMC calibration process")
        logger.info(f"📊 Database: {calib_context.db_path}")
        logger.info(f"🎯 Max populations: {calib_context.max_populations}")
        logger.info(f"🔬 Max simulations: {calib_context.max_simulations}")

        
        # Setup sampler
        logger.info("⚙️ Setting up sampler...")
        my_sampler = calib_context.setup_sampler(calib_context.cluster_setup_func)
        
        # Setup population strategy
        logger.info("👥 Setting up population strategy...")
        population_size = calib_context.setup_population_strategy()
        
        # Setup distance function
        logger.info("📏 Setting up distance function...")
        distance_function = calib_context.setup_distance_function(calib_context.distance_functions)
        
        # Setup transition function
        logger.info("🔄 Setting up transition function...")
        transitions_func = calib_context.setup_transition_function()
        
        # Setup epsilon function
        logger.info("🎯 Setting up epsilon function...")
        eps_function = calib_context.setup_epsilon_function()
        
        # Setup model wrappers
        logger.info("🧬 Setting up model wrappers...")
        
        # Handle multiple models if specified
        if calib_context.model_selection and calib_context.num_models > 1:
            # This would require additional logic for multiple models
            # For now, use single model approach
            logger.info("⚠️ Multiple model selection not fully implemented, using single model")
        
        # Create model wrapper
        model_wrapper = calib_context.create_model_wrapper(
            calib_context.fixed_params, 
            calib_context.workers_inner
        )
        
        models_list = [model_wrapper]
        priors_list = [calib_context.prior]
        
        # Setup ABC-SMC object
        logger.info("🔧 Setting up ABC-SMC object...")
        abc_smc = calib_context.setup_abc_smc(
            models_list=models_list,
            priors_list=priors_list,
            distance_function=distance_function,
            population_size=population_size,
            transitions_func=transitions_func,
            my_sampler=my_sampler,
            eps_function=eps_function
        )
        
        # Load or create database
        logger.info("💾 Managing database...")
        resume_db, current_populations, current_simulations = calib_context.load_or_create_database(abc_smc)
        
        # Run calibration
        logger.info("🎲 Starting calibration run...")
        calib_context.run_calibration(abc_smc, resume_db, current_populations, current_simulations)
        
        # Check convergence and run additional populations if needed
        if calib_context.convergence_check_func is not None and calib_context.mode == 'cluster':
            logger.info("🔍 Checking convergence...")
            while True:
                if calib_context.convergence_check_func(abc_smc.history):
                    logger.info("✅ Convergence achieved!")
                    break
                
                # If max limits reached, extend by one to run one more population
                if abc_smc.history.total_nr_simulations >= calib_context.max_simulations:
                    calib_context.max_simulations = abc_smc.history.total_nr_simulations + 1
                
                if abc_smc.history.n_populations >= calib_context.max_populations:
                    calib_context.max_populations = abc_smc.history.n_populations + 1
                
                logger.info(f"🔄 Continuing calibration: {calib_context.max_simulations} simulations, {calib_context.max_populations} populations")
                
                # Run one more iteration
                abc_smc.run(
                    max_nr_populations=calib_context.max_populations,
                    max_total_nr_simulations=calib_context.max_simulations
                )
        # Remove temporary file and folders
        physicell_model = PhysiCell_Model(calib_context.model_config['ini_path'], calib_context.model_config['struc_name'])
        physicell_model.remove_io_folders()
        
        # Final results
        final_history = abc_smc.history
        logger.info("✅ ABC-SMC calibration completed successfully!")
        logger.info(f"📈 Final statistics:")
        logger.info(f"   - Populations: {final_history.n_populations}")
        logger.info(f"   - Total simulations: {final_history.total_nr_simulations}")
        logger.info(f"   - Database: {calib_context.db_path}")
        
        return final_history
        
    except Exception as e:
        logger.error(f"❌ Error in ABC-SMC calibration: {e}")
        raise
