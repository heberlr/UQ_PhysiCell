import os, sys
import logging
import concurrent.futures
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import torch
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch import fit_gpytorch_mll
from botorch.utils.multi_objective.pareto import is_non_dominated

# All the specific classes we need
from typing import Union, Optional
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.logei import qLogExpectedImprovement

# My local modules
from uq_physicell import PhysiCell_Model
from ..utils.model_wrapper import run_replicate_serializable
from ..database.bo_db import (
    create_structure, insert_metadata, insert_param_space, insert_qois, 
    insert_gp_models, insert_samples, insert_output, load_structure
)
from ..utils.distances import SumSquaredDifferences, Manhattan, Chebyshev
from .utils import unnormalize_params, tensor_to_param_dict, normalize_params, param_dict_to_tensor

class CalibrationContext:
    """
    Context for Bayesian Optimization calibration with enhanced acquisition strategies.
    
    This class encapsulates all necessary parameters and configurations for model calibration
    using multi-objective Bayesian optimization with sophisticated handling of parameter
    non-identifiability issues through acquisition function enhancement strategies.
    
    Attributes:
        db_path (str): Path to the database file for storing and retrieving samples.
        obsData (str or dict): Path or dict containing the observed data.
        obsData_columns (dict): Dictionary mapping QoI names to their corresponding columns in the observed data.
        model_config (dict): Configuration dictionary for the PhysiCell model, including paths and structure names.
        qoi_functions (dict): Dictionary of functions to compute quantities of interest (QoIs) from model outputs.
        distance_functions (dict): Dictionary of functions to compute distances between model outputs and observed data.
        search_space (dict): Dictionary defining the search space for parameters, including bounds and types.
        bo_options (dict): Options for Bayesian Optimization including sampling parameters and acquisition strategy.
                          Use 'acq_func_strategy' key for strategy: 'diversity_bonus', 'uncertainty_weighting', 
                          'soft_constraints', 'adaptive_scaling', 'combined', or 'none' (default: 'none').
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
        search_space: dict, 
        bo_options: dict, 
        logger: logging.Logger=None
    ):
        """Initialize CalibrationContext with comprehensive validation and setup."""
        # Core configuration
        self.db_path = db_path
        self.model_config = model_config
        self.qoi_functions = qoi_functions
        self.distance_functions = distance_functions
        self.search_space = search_space
        self.bo_options = bo_options
        if logger is None:
            # Create a logger with proper configuration
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Only add handler if none exist to avoid duplicates
            if not self.logger.handlers:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
            
            # Always prevent propagation to root logger to avoid duplicate messages
            self.logger.propagate = False
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
        
        # Validate acquisition strategy early to catch configuration issues
        self._validate_acquisition_strategy()

        # Optional model customizations
        self.fixed_params = bo_options.get("fixed_params", {})
        self.summary_function = bo_options.get("summary_function", None)
        self.custom_run_single_replicate_func = bo_options.get("custom_run_single_replicate_func", None)
        self.custom_aggregation_func = bo_options.get("custom_aggregation_func", None)
        
        # For multi-objective optimization the reference point is 0
        # Fitness values are in range (0, 1] -> fitness 1 is the best (distance=0), approaches 0 for very large distances
        self.ref_point = torch.tensor([0.0] * len(qoi_functions), dtype=torch.float64)  # Standard ref point for inverse fitness
        self.max_workers = bo_options.get("max_workers", os.cpu_count())  # Number of parallel workers, can be adjusted based on system capabilities
        # Initialize random samples for the search space
        # Split the initial_args into chunks for parallel processing
        # We set workers for the outer pool to (max_workers // num_replicates) so that the total number of processes
        # (outer pool workers * num_replicates per sample) does not exceed max_workers. This prevents CPU oversubscription.
        # For example, if max_workers=8 and num_replicates=2, then workers_out=4, so at most 4 samples run in parallel,
        # each with 2 replicates, for a total of 8 processes.
        self.num_replicates = self.model_config['numReplicates']
        self.workers_inner = min(self.max_workers, self.num_replicates)
        self.workers_out = max(1, self.max_workers // self.workers_inner)
        
        # BO-specific parameters (Model evaluation = [num_initial_samples + batch_size_bo * batch_size_per_iteration]* num_replicates)
        self.num_initial_samples = bo_options.get("num_initial_samples", 2 * (len(search_space) + 1))  # Initial samples based on the number of parameters 2*(num_params + 1)  # Number of initial samples for Bayesian optimization
        self.batch_size_bo =  bo_options.get("num_iterations", 10)  # Number of BO iterations
        self.batch_size_per_iteration = bo_options.get("batch_size_per_iteration", 1)  # Batch size for each BO iteration
        self.samples_per_batch_act_func = bo_options.get("samples_per_batch_act_func", 128)  # Number of samples per batch for the acquisition function
        self.num_restarts_act_func = bo_options.get("num_restarts_act_func", 20)  # Number of restarts for acquisition function optimization
        self.raw_samples_act_func = bo_options.get("raw_samples_act_func", 512)  # Number of raw samples for acquisition function
        self.use_exponential_fitness = bo_options.get("use_exponential_fitness", True)
        
        # Initialize metadata for database
        self.dic_metadata = {
            "BO_Method": "Multi-objective optimization with qNEHVI" if len(self.qoi_functions.keys()) > 1 else "Single-objective optimization",
            "ObsData_Path": self.obsData_path,
            "Ini_File_Path": self.model_config["ini_path"],
            "StructureName": self.model_config["struc_name"],
        }
        
        # Store enhanced strategy metadata
        acq_func_strategy = bo_options.get("acq_func_strategy", "none")
        if acq_func_strategy != "none":
            self.dic_metadata["Enhancement_Strategy"] = acq_func_strategy
            # Store strategy-specific weights if provided
            if "diversity_weight" in bo_options:
                self.dic_metadata["Diversity_Weight"] = bo_options["diversity_weight"]
            if "uncertainty_weight" in bo_options:
                self.dic_metadata["Uncertainty_Weight"] = bo_options["uncertainty_weight"]
            if "constraint_strength" in bo_options:
                self.dic_metadata["Constraint_Strength"] = bo_options["constraint_strength"]

        # Initialize the qoi_details
        self.qoi_details = {
            "QOI_Name": list(self.qoi_functions.keys()),
            "QOI_Function": [self.qoi_functions[key] for key in self.qoi_functions.keys()],
            "ObsData_Column": [obsData_columns[key] for key in self.qoi_functions.keys()],
            "QoI_distanceFunction": [
                (func.__name__ if callable(func) else eval(func).__name__)
                for func in [self.distance_functions[key]['function'] for key in self.distance_functions.keys()]
            ],
            "QoI_distanceWeight": [self.distance_functions[key]['weight'] for key in self.distance_functions.keys()],
        }

        self.logger.info(f"🔧 CalibrationContext initialized with {self.max_workers} max workers, {self.workers_inner} inner workers, and {self.workers_out} outer workers.")
        # Cancellation flag for cooperative cancellation support
        self.cancel_requested = False

    def _validate_acquisition_strategy(self) -> None:
        """
        Validate acquisition strategy configuration and warn about invalid values.
        Raises:
            ValueError: If acq_func_strategy contains invalid values.
        """
        # Validate strategy if specified
        acq_func_strategy = self.bo_options.get("acq_func_strategy", "none")
        valid_strategies = ["diversity_bonus", "uncertainty_weighting", "soft_constraints", 
                          "adaptive_scaling", "combined", "none"]
        
        if acq_func_strategy not in valid_strategies:
            self.logger.error(f"❌ Invalid acquisition strategy '{acq_func_strategy}'. Valid strategies: {valid_strategies}")
            raise
        
        # Log the strategy being used
        if acq_func_strategy != "none":
            self.logger.debug(f"✅ Using enhanced identification strategy: {acq_func_strategy}")
            if acq_func_strategy == "combined":
                self.logger.debug("   - Combining diversity bonus + uncertainty weighting for balanced exploration")
        else:
            self.logger.debug("ℹ️  Using pure BoTorch acquisition (no enhancement strategy)")

    def default_run_single_replicate(self, sample_id: int, replicate_id: int, params: dict) -> dict:
        """
        Run a single replicate of the PhysiCell model.
        This function is responsible for executing the model with the given parameters and returning the results.
        Args:
            sample_id (int): Unique identifier for the sample being processed.
            replicate_id (int): Unique identifier for the replicate being processed.
            params (dict): Dictionary of parameters to be used in the model run.
        Returns:
            dict: dictionary of model outputs.
        """
        PC_model = PhysiCell_Model(self.model_config["ini_path"], self.model_config["struc_name"])
        dic_params_xml = {par_name: par_value for par_name, par_value in params.items() if par_name in PC_model.XML_parameters_variable.values()}
        dic_params_rules = {par_name: par_value for par_name, par_value in params.items() if par_name in PC_model.parameters_rules_variable.values()}
        _, _, result_data = run_replicate_serializable(
            self.model_config, 
            sample_id, replicate_id,
            dic_params_xml, dic_params_rules,
            qoi_functions=self.qoi_functions, return_binary_output=False,
            custom_summary_function=self.summary_function
        )
        dic_result_data = result_data.to_dict(orient='list')
        dic_result_data_np = {key: np.array(list_values) for key, list_values in dic_result_data.items()}
        return dic_result_data_np

    def default_aggregation_func(self, replicate_results:list, sample_id:int) -> tuple:
        """
        Aggregate results from multiple replicates. This function computes the mean and standard deviation
        for each key in the results.
        Args:
            replicate_results (list): List of dictionaries containing the results from each replicate.
        Returns:
            tuple: A tuple containing the aggregated results, noise estimates, and a dictionary of all results.
        """
        agg_results = {}
        agg_noise = {}
        for key in replicate_results[0].keys():
            agg_results[key] = np.mean([r[key] for r in replicate_results], axis=0)
            agg_noise[key] = np.std([r[key] for r in replicate_results], axis=0)

        # Convert the results in a dataframe with all replicates
        dic_results = {}
        for replicate_id, result in enumerate(replicate_results):
            dic_results[replicate_id] = result

        # Compute each objective metric (QoI fitness) for each replicate first
        # Convert distance (lower is better) to fitness (higher is better) using inverse transformation
        objectives_per_replicate = []
        for replicate_id, replicate_result in enumerate(replicate_results):
            replicate_objectives = {}
            for qoi, dist_info in self.distance_functions.items():
                dicObsData = {'time': self.dic_obsData['time'], 'value': self.dic_obsData[qoi]}
                dicModel = {'time': replicate_result['time'], 'value': replicate_result[qoi]}
                if callable(dist_info["function"]):
                    distance = dist_info["weight"] * dist_info["function"](dicObsData, dicModel)
                else:
                    distance = dist_info["weight"] * eval(dist_info["function"]).__call__(dicObsData, dicModel)
                # Convert distance to fitness
                if self.use_exponential_fitness: # Use exponential fitness transformation
                    fitness = np.exp(-distance)
                    # IMPROVED: Ensure exponential fitness stays in reasonable range
                    min_fitness = 1e-3  # Higher minimum for numerical stability
                    max_fitness = 1.0   # Cap maximum to prevent extreme values
                    fitness = np.clip(fitness, min_fitness, max_fitness)
                else: # Use inverse distance transformation
                    fitness = 1.0 / (1.0 + distance)
                    # Standard clamping for inverse transformation
                    min_fitness = 1e-3
                    max_fitness = 1.0   # Cap maximum to prevent extreme values
                    fitness = np.clip(fitness, min_fitness, max_fitness)
                
                # Check for problematic values
                if fitness < min_fitness or np.isnan(fitness):
                    fitness = min_fitness
                    self.logger.warning(f"Fitness value clamped for QoI '{qoi}' - setting to minimum {min_fitness}")
                
                replicate_objectives[qoi] = fitness
                # Debug: log distance and fitness values
                if sample_id < 3:  # Only for first few samples to avoid log spam
                    self.logger.debug(f"\t sampleID: {sample_id} replicateID: {replicate_id} - QoI '{qoi}': distance={distance:.3f}, fitness={fitness:.6f}")
            objectives_per_replicate.append(replicate_objectives)

        # Compute mean and std of objectives across replicates
        objectives = {}
        obj_noise = {}
        for qoi in self.distance_functions.keys():
            qoi_values = [rep_obj[qoi] for rep_obj in objectives_per_replicate]
            objectives[qoi] = np.mean(qoi_values)
            obj_noise[qoi] = np.std(qoi_values)

        return objectives, obj_noise, dic_results

    def evaluate_params(self, params, sample_index):
        """Evaluate a single parameter set by running replicates in parallel, aggregate outputs,
           compute multi-objective metrics and return a dict.
           Args:
            params (dict): Dictionary of parameters to be evaluated.
            sample_index (int): Index of the sample being evaluated.
           Returns:
            tuple: objectives (dict) containing the computed objectives for the given parameters,
                   obj_noise (dict) containing the standard deviation of objectives across replicates,
                   dic_results (dict) containing the results from all replicates.
        """
        self.logger.debug(f"Sample {sample_index} with params: {params}")
        # Check if using default run_single_replicate function
        if not self.custom_run_single_replicate_func:
            # Include fixed_params into params
            params.update(self.fixed_params)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers_inner) as executor:
                replicate_results = list(executor.map(
                    self.default_run_single_replicate,
                    [sample_index] * self.num_replicates,
                    range(self.num_replicates),
                    [params] * self.num_replicates
                ))
        else: # Use the custom run_single_replicate function provided in bo_options
            # Note that this function signature MUST be:  
            # sample_index:int, replicate_id:int, params:dict, fixed_params:dict, model_config:dict
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers_inner) as executor:
                replicate_results = list(executor.map(
                    self.custom_run_single_replicate_func,
                    [sample_index] * self.num_replicates,
                    range(self.num_replicates),
                    [params] * self.num_replicates,
                    [self.fixed_params] * self.num_replicates,
                    [self.model_config] * self.num_replicates
                ))
        
        if not self.custom_aggregation_func: # Default: Aggregate replicate results (mean and standard deviation)
            objectives, obj_noise, dic_results = self.default_aggregation_func(replicate_results, sample_index)
        else: # Custom aggregation function (without scale factor - weights handle scaling)
            objectives, obj_noise, dic_results = self.custom_aggregation_func(replicate_results, sample_index, self.distance_functions, self.dic_obsData)

        # NOTE: Enhanced strategies are now handled in the acquisition function, not as objectives
        # This avoids unnecessary GP modeling of deterministic enhancement terms

        return objectives, obj_noise, dic_results

    def save_results_to_db(self, sample_index:int, objectives:dict, noise_std:dict, dic_results:dict):
        """
        Save results to database.
        Args:
            sample_index (int): Index of the sample being saved.
            objectives (dict): Dictionary containing the objective values.
            noise_std (dict): Dictionary containing the noise standard deviations.
            dic_results (dict): Dictionary containing the results to save.
        """
        try:
            binary_objectives = pickle.dumps(objectives)  # Convert dict to binary format
            binary_noise_std = pickle.dumps(noise_std)  # Convert dict to binary format
            binary_df = pickle.dumps(dic_results)  # Convert dict to binary format
            insert_output(self.db_path, sample_index, binary_objectives, binary_noise_std, binary_df)
            self.logger.debug(f"Successfully saved results for sample {sample_index} to database.")
        except Exception as e:
            self.logger.error(f"Error saving results for sample {sample_index} to database: {e}")
            raise

    def generate_and_evaluate_samples(self, num_samples:int, start_sample_id:int = 0, iteration_id:int = 0) -> tuple:
        """
        Generate and evaluate samples for Bayesian optimization using Sobol sequences.
        
        This function generates samples in the search space using Sobol sequences, evaluates them
        with the model, and saves results to the database. Used for both initial sampling and 
        restart functionality.
        
        Args:
            num_samples (int): Number of samples to generate and evaluate.
            start_sample_id (int, optional): Starting sample ID. default will use 0 for new databases
                                           or caller should provide the appropriate starting ID.
        Returns:
            tuple: (train_x, train_obj, train_obj_true, train_obj_std) - Tensors ready for BO pipeline.
        """
        num_params = len(self.search_space)
        bounds = torch.stack([torch.zeros(num_params), torch.ones(num_params)]) # 0 to 1 bounds for each parameter
        
        sample_ids = np.arange(start_sample_id, start_sample_id + num_samples, dtype=int)
        self.logger.debug(f"Generating {num_samples} samples with IDs {start_sample_id} to {start_sample_id + num_samples - 1}")
        train_x = draw_sobol_samples(bounds, n=num_samples, q=1).squeeze(1).to(torch.float64)  # Convert to float64 and squeeze q dimension
        train_x_dic_params = []
        for i in range(num_samples):
            train_x_unnorm_i = unnormalize_params(train_x[i], self.search_space)
            dic_params_i = tensor_to_param_dict(train_x_unnorm_i, self.search_space)
            # save parameters in the database
            insert_samples(self.db_path, iteration_id, {sample_ids[i]: dic_params_i})
            train_x_dic_params.append(dic_params_i)

        self.logger.debug(f"Evaluating {num_samples} samples with model...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers_out) as outer_executor:
            list_output_tuples = list(outer_executor.map(self.evaluate_params, train_x_dic_params, sample_ids))

        # Save results to database sequentially to avoid concurrency issues
        for i, (objectives, obj_noise, dic_results) in enumerate(list_output_tuples):
            self.save_results_to_db(sample_ids[i], objectives, obj_noise, dic_results)

        # TENSOR CREATION EXPLANATION:
        # list_output_tuples contains tuples like (objectives_dict, noise_dict, dic_results) for each sample
        # objectives_dict is like {"epi_": 1.23, "epi_infected": 4.56} (scalar values per objective)
        # list(output[0].values()) extracts values as [1.23, 4.56] for one sample
        # The list comprehension creates [[1.23, 4.56], [2.34, 5.67], ...] for all samples
        # torch.tensor() converts this to a 2D tensor with shape (n_samples, n_objectives)
        train_obj_true = torch.tensor([list(output[0].values()) for output in list_output_tuples], dtype=torch.float64)
        train_obj_std = torch.tensor([list(output[1].values()) for output in list_output_tuples], dtype=torch.float64)
        # Add noise to the average of mean based on the standard deviation of the outputs (normal distributed noise)
        train_obj = train_obj_true + torch.randn_like(train_obj_true) * train_obj_std
        # Ensure fitness values remain positive (fitness should not be negative)
        min_fitness = 1e-3
        train_obj = torch.clamp(train_obj, min=min_fitness)

        return train_x, train_obj, train_obj_true, train_obj_std

    def load_existing_data(self) -> tuple:
        """
        Load existing data from the database for resume functionality.
        Returns:
            tuple: A tuple containing training data tensors, latest iteration, and hypervolume.
        """
        self.logger.debug("Loading existing data from database...")
        
        # Load all data from the database
        df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(self.db_path)
        
        # Validate that the loaded data is compatible with current configuration
        self._validate_loaded_data(df_metadata, df_param_space, df_qois)
        
        # Reconstruct training tensors from loaded data
        train_x, train_obj, train_obj_true, train_obj_std = self._reconstruct_training_data(df_samples, df_output)
        
        # Get the latest iteration and hypervolume (don't load the model - we'll recreate it)
        if not df_gp_models.empty:
            latest_iteration = df_gp_models['IterationID'].max()
            latest_hypervolume = df_gp_models[df_gp_models['IterationID'] == latest_iteration]['Hypervolume'].iloc[0]
        else:
            latest_iteration = -1
            latest_hypervolume = 0.0
        self.logger.debug(f"Loaded {len(train_x)} samples from {latest_iteration + 1} iterations")
        self.logger.debug(f"Latest hypervolume: {latest_hypervolume}")

        # Return without the model - we'll recreate it from the training data
        return train_x, train_obj, train_obj_true, train_obj_std, latest_iteration, latest_hypervolume

    def _validate_loaded_data(self, df_metadata, df_param_space, df_qois):
        """
        Validate that loaded data is compatible with current configuration.
        """
        # Check if parameter space matches
        loaded_params = set(df_param_space['ParamName'].tolist())
        current_params = set(self.search_space.keys())
        if loaded_params != current_params:
            raise ValueError(f"Parameter space mismatch. Loaded: {loaded_params}, Current: {current_params}")
            
        # Check if QoIs match
        loaded_qois = set(df_qois['QoI_Name'].tolist())
        current_qois = set(self.qoi_functions.keys())
        if loaded_qois != current_qois:
            raise ValueError(f"QoI mismatch. Loaded: {loaded_qois}, Current: {current_qois}")
            
        self.logger.debug("Loaded data validation passed")

    def _reconstruct_training_data(self, df_samples, df_output) -> tuple:
        """
        Reconstruct training tensors from loaded database data.
        """
        # Get unique sample IDs and sort them
        sample_ids = sorted(df_output['SampleID'].unique())
        
        # Reconstruct parameters tensor (train_x)
        train_x_list = []
        for sample_id in sample_ids:
            sample_params = df_samples[df_samples['SampleID'] == sample_id]
            param_dict = {}
            for _, row in sample_params.iterrows():
                param_dict[row['ParamName']] = row['ParamValue']
            
            # Normalize parameters to [0,1] range
            param_tensor = param_dict_to_tensor(param_dict, self.search_space)
            normalized_tensor = normalize_params(param_tensor, self.search_space)
            train_x_list.append(normalized_tensor)
        
        train_x = torch.stack(train_x_list).to(torch.float64)
        
        # Reconstruct objectives tensors
        train_obj_true_list = []
        train_obj_std_list = []
        train_obj_list = []
        
        for sample_id in sample_ids:
            output_row = df_output[df_output['SampleID'] == sample_id].iloc[0]
            objectives = output_row['ObjFunc']
            noise_std = output_row['Noise_Std']
            
            # Convert objectives dict to tensor (maintain order)
            obj_values = []
            std_values = []
            for qoi in self.qoi_functions.keys():
                obj_values.append(objectives[qoi])
                std_values.append(noise_std[qoi])
            
            obj_tensor = torch.tensor(obj_values, dtype=torch.float64)
            std_tensor = torch.tensor(std_values, dtype=torch.float64)
            
            train_obj_true_list.append(obj_tensor)
            train_obj_std_list.append(std_tensor)
            
            # Add noise to create train_obj (same as in original code)
            noisy_obj = obj_tensor + torch.randn_like(obj_tensor) * std_tensor
            # Ensure fitness values remain positive (fitness should not be negative)
            min_fitness = 1e-3
            noisy_obj = torch.clamp(noisy_obj, min=min_fitness)
            train_obj_list.append(noisy_obj)
        
        train_obj_true = torch.stack(train_obj_true_list)
        train_obj_std = torch.stack(train_obj_std_list)
        train_obj = torch.stack(train_obj_list)
        
        return train_x, train_obj, train_obj_true, train_obj_std

    def update_bo_iterations(self, additional_iterations: int):
        """
        Update the number of BO iterations to include additional iterations for resume.
        """
        original_iterations = self.batch_size_bo
        self.batch_size_bo += additional_iterations
        self.logger.debug(f"Updated max iterations from {original_iterations} to {self.batch_size_bo} (added {additional_iterations} iterations)")

    def load_existing_data(self) -> tuple:
        """
        Load existing data from the database for resume functionality.
        Returns:
            tuple: A tuple containing training data tensors, latest iteration, and hypervolume.
        """
        self.logger.debug("Loading existing data from database...")
        
        # Load all data from the database
        df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(self.db_path)
        
        # Validate that the loaded data is compatible with current configuration
        self._validate_loaded_data(df_metadata, df_param_space, df_qois)
        
        # Reconstruct training tensors from loaded data
        train_x, train_obj, train_obj_true, train_obj_std = self._reconstruct_training_data(df_samples, df_output)
        
        # Get the latest iteration and hypervolume (don't load the model - we'll recreate it)
        if not df_gp_models.empty:
            latest_iteration = df_gp_models['IterationID'].max()
            latest_hypervolume = df_gp_models[df_gp_models['IterationID'] == latest_iteration]['Hypervolume'].iloc[0]
        else:
            latest_iteration = -1
            latest_hypervolume = 0.0
            
        self.logger.debug(f"Loaded {len(train_x)} samples from {latest_iteration + 1} iterations")
        self.logger.debug(f"Latest hypervolume: {latest_hypervolume}")
        
        # Return without the model - we'll recreate it from the training data
        return train_x, train_obj, train_obj_true, train_obj_std, latest_iteration, latest_hypervolume

    def _validate_loaded_data(self, df_metadata, df_param_space, df_qois):
        """
        Validate that loaded data is compatible with current configuration.
        """
        # Check if parameter space matches
        loaded_params = set(df_param_space['ParamName'].tolist())
        current_params = set(self.search_space.keys())
        if loaded_params != current_params:
            raise ValueError(f"Parameter space mismatch. Loaded: {loaded_params}, Current: {current_params}")
            
        # Check if QoIs match
        loaded_qois = set(df_qois['QoI_Name'].tolist())
        current_qois = set(self.qoi_functions.keys())
        if loaded_qois != current_qois:
            raise ValueError(f"QoI mismatch. Loaded: {loaded_qois}, Current: {current_qois}")
            
        self.logger.debug("Loaded data validation passed")

    def analyze_convergence(self, hvs_list: list, train_obj_true: torch.Tensor, train_x: torch.Tensor, iteration: int) -> dict:
        """
        Sophisticated convergence analysis that properly distinguishes between:
        1. True convergence (optimization found optimal solutions)
        2. Stagnation with good coverage (likely converged to optimal region)
        3. Stagnation with poor coverage (stuck in suboptimal region)
        4. Still in progress
        
        Args:
            hvs_list (list): History of hypervolume values
            train_obj_true (torch.Tensor): True objective values
            train_x (torch.Tensor): Parameter values (normalized)
            iteration (int): Current iteration
            
        Returns:
            dict: Convergence analysis results with recommendations
        """
        result = {
            "converged": False,
            "stagnant": False,
            "needs_restart": False,
            "status": "in_progress",
            "reason": "",
            "convergence_confidence": 0.0,
            "suggestion": ""
        }
        
        if len(hvs_list) < 10:
            result["reason"] = "Insufficient data for convergence analysis"
            result["status"] = "insufficient_data"
            return result
        
        # Strategy-aware analysis logging
        acq_func_strategy = self.bo_options.get("acq_func_strategy", "none")
        self.logger.debug(f"🔍 Convergence analysis with strategy: {acq_func_strategy}")
        
        # 1. Hypervolume trend analysis
        hv_improvements = [hvs_list[i] - hvs_list[i-5] for i in range(5, len(hvs_list))]
        recent_improvements = hv_improvements[-5:] if len(hv_improvements) >= 5 else hv_improvements
        
        # 2. Calculate hypervolume stability (coefficient of variation)
        if len(hvs_list) >= 15:
            recent_hvs = hvs_list[-15:]
            hv_mean = np.mean(recent_hvs)
            hv_std = np.std(recent_hvs)
            hv_stability = hv_std / hv_mean if hv_mean > 0 else float('inf')
            result["hv_stability"] = hv_stability
        else:
            hv_stability = float('inf')
            result["hv_stability"] = hv_stability
            
        # 3. Pareto front quality analysis
        fitness_values = train_obj_true.detach().numpy()
        pareto_analysis = self._analyze_pareto_front(fitness_values)
        result.update(pareto_analysis)
        
        # 4. Parameter space coverage analysis
        coverage_analysis = self._analyze_parameter_coverage(train_x)
        result.update(coverage_analysis)
        
        # 5. Acquisition function diversity (if we have recent acquisition values)
        acq_diversity = self._estimate_acquisition_diversity(train_x)
        result["acquisition_diversity"] = acq_diversity
        
        # 6. IMPROVED CONVERGENCE DECISION LOGIC
        # Strategy-specific threshold adjustments
        if acq_func_strategy in ["combined", "diversity_bonus"]:
            hv_stability_threshold = 0.01
            coverage_threshold_good = 0.7
            coverage_threshold_poor = 0.3
            acq_diversity_threshold = 0.05
            pareto_quality_threshold = 0.6
        elif acq_func_strategy == "uncertainty_weighting":
            hv_stability_threshold = 0.008
            coverage_threshold_good = 0.65
            coverage_threshold_poor = 0.25
            acq_diversity_threshold = 0.08
            pareto_quality_threshold = 0.7
        else:
            # Pure BoTorch - use original strict thresholds
            hv_stability_threshold = 0.005
            coverage_threshold_good = 0.6
            coverage_threshold_poor = 0.2
            acq_diversity_threshold = 0.1
            pareto_quality_threshold = 0.7
        
        # Check if hypervolume has stabilized (minimal recent improvements)
        is_hv_stable = (hv_stability < hv_stability_threshold and
                        len(recent_improvements) >= 3 and
                        all(imp >= -1e-10 for imp in recent_improvements) and
                        max(recent_improvements) < 1e-6)
        
        # Check for long-term stagnation
        is_stagnant = (len(hvs_list) >= 10 and
                       abs(hvs_list[-1] - hvs_list[-10]) < 1e-10)
        
        # DECISION TREE:
        
        # CASE 1: TRUE CONVERGENCE - Stable HV + Good quality + Good coverage
        if (is_hv_stable and
            pareto_analysis["pareto_quality"] >= pareto_quality_threshold and
            coverage_analysis["coverage"] >= coverage_threshold_good):
            
            result["converged"] = True
            result["status"] = "converged"
            result["reason"] = f"Optimal solutions found: stable hypervolume with excellent Pareto quality and parameter coverage (strategy: {acq_func_strategy})"
            result["convergence_confidence"] = min(0.95, 
                0.3 + 0.3 * pareto_analysis["pareto_quality"] + 
                0.2 * coverage_analysis["coverage"] + 
                0.2 * (1 - min(hv_stability / hv_stability_threshold, 1.0)))
        
        # CASE 2: LIKELY CONVERGED - Stagnant but good coverage and quality
        elif (is_stagnant and
              coverage_analysis["coverage"] >= coverage_threshold_good and
              pareto_analysis["pareto_quality"] >= pareto_quality_threshold * 0.8):  # Slightly relaxed quality
            
            result["converged"] = True
            result["stagnant"] = True
            result["status"] = "converged_stagnant"
            result["reason"] = f"Likely converged: good parameter space exploration with acceptable Pareto quality despite stagnation"
            result["convergence_confidence"] = min(0.85,
                0.4 * pareto_analysis["pareto_quality"] + 
                0.4 * coverage_analysis["coverage"] + 
                0.2 * (1 - min(hv_stability / hv_stability_threshold, 1.0)))
            result["suggestion"] = "Consider stopping optimization - likely found optimal region"
        
        # CASE 3: STUCK IN SUBOPTIMAL REGION - Stagnant with poor coverage or quality
        elif (is_stagnant and
              (coverage_analysis["coverage"] < coverage_threshold_poor or
               pareto_analysis["pareto_quality"] < pareto_quality_threshold * 0.5 or
               acq_diversity < acq_diversity_threshold)):
            
            result["stagnant"] = True
            result["needs_restart"] = True
            result["status"] = "stuck_suboptimal"
            result["reason"] = f"Stuck in suboptimal region: poor exploration or quality despite stagnation"
            result["exploration_quality"] = f"Coverage: {coverage_analysis['coverage']:.1%}, Pareto: {pareto_analysis['pareto_quality']:.2f}, AcqDiv: {acq_diversity:.3f}"
            
            # Provide specific restart suggestions
            if coverage_analysis["coverage"] < coverage_threshold_poor:
                if acq_func_strategy in ["combined", "diversity_bonus"]:
                    result["suggestion"] = "Despite diversity enhancement, coverage is low - try increasing diversity_weight or restart with more initial samples"
                else:
                    result["suggestion"] = "Poor parameter space coverage - enable diversity_bonus strategy or restart with more initial samples"
            elif pareto_analysis["pareto_quality"] < pareto_quality_threshold * 0.5:
                result["suggestion"] = "Poor fitness values: check distance function weights or try different acquisition strategy"
            else:
                result["suggestion"] = "Low acquisition diversity: restart with enhanced exploration settings"
        
        # CASE 4: EARLY STAGNATION WARNING - Some stagnation but not severe
        elif (is_stagnant and iteration >= 15):  # Only warn after sufficient iterations
            result["stagnant"] = True
            result["status"] = "stagnant_warning"
            result["reason"] = f"Stagnation detected but exploration quality unclear - monitor closely"
            result["suggestion"] = "Monitor next few iterations; consider restart if no improvement"
    
        # CASE 5: NORMAL PROGRESS
        else:
            result["status"] = "in_progress"
            if len(hvs_list) >= 3:
                recent_improvement = hvs_list[-1] - hvs_list[-3]
                if recent_improvement < 1e-8:
                    result["reason"] = "Slow but steady progress"
                else:
                    result["reason"] = "Normal optimization progress"
            else:
                result["reason"] = "Early stage optimization"
    
        # Calculate overall convergence confidence for non-converged cases
        if not result["converged"]:
            stability_score = max(0, 1 - hv_stability / 0.01) if hv_stability != float('inf') else 0
            quality_score = pareto_analysis["pareto_quality"]
            coverage_score = coverage_analysis["coverage"]
            
            result["convergence_confidence"] = (stability_score + quality_score + coverage_score) / 3.0
            
        return result
    
    def _analyze_pareto_front(self, fitness_values):
        """
        Analyze the quality of the Pareto front using BoTorch's optimized implementations.
        
        OPTIMIZATION: Use cached data from acquisition function when available to avoid recomputation.
        
        Args:
            fitness_values (np.ndarray): Fitness values with shape (n_samples, n_objectives)
            
        Returns:
            dict: Dictionary containing Pareto front analysis
        """
        
        # Check if we have cached Pareto data from the acquisition function
        if hasattr(self, '_cached_pareto_data') and self._cached_pareto_data is not None:
            cached_data = self._cached_pareto_data
            self.logger.debug("🚀 Using cached Pareto analysis from acquisition function")
            
            # Clear the cache after use and return cached result
            self._cached_pareto_data = None
            return {
                "pareto_ratio": cached_data["pareto_ratio"],
                "pareto_quality": cached_data["pareto_quality"], 
                "pareto_spread": cached_data["pareto_spread"],
                "n_pareto_points": cached_data["n_pareto_points"]
            }
        
        # Fallback: Compute from fitness values
        self.logger.debug("🔄 Computing Pareto analysis from fitness values")
        
        n_samples, n_objectives = fitness_values.shape
        fitness_tensor = torch.tensor(fitness_values, dtype=torch.float64)
        
        # Find Pareto-optimal points
        pareto_mask = is_non_dominated(fitness_tensor, maximize=True, deduplicate=True)
        pareto_points = fitness_tensor[pareto_mask].numpy()
        n_pareto = len(pareto_points)
        
        # Calculate metrics
        pareto_ratio = n_pareto / n_samples
        
        if n_pareto > 1:
            # Quality: average fitness + distance to ideal point
            pareto_avg_fitness = np.mean(pareto_points)
            ideal_point = np.ones(n_objectives)
            distances_to_ideal = np.sqrt(np.sum((pareto_points - ideal_point)**2, axis=1))
            avg_distance_to_ideal = np.mean(distances_to_ideal)
            max_possible_distance = np.sqrt(n_objectives)
            
            pareto_quality = 0.7 * pareto_avg_fitness + 0.3 * (1.0 - avg_distance_to_ideal / max_possible_distance)
            pareto_quality = np.clip(pareto_quality, 0.0, 1.0)
            pareto_spread = np.std(pareto_points, axis=0).mean()
        else:
            pareto_quality = np.mean(fitness_values)
            pareto_spread = 0.0
            if n_pareto <= 1:
                self.logger.warning("Only one Pareto point found - may indicate poor exploration")

        return {
            "pareto_ratio": pareto_ratio,
            "pareto_quality": pareto_quality,
            "pareto_spread": pareto_spread,
            "n_pareto_points": n_pareto
        }
    
    def _analyze_parameter_coverage(self, train_x):
        """
        Analyze parameter space coverage.
        
        Args:
            train_x (torch.Tensor): Normalized parameter values with shape (n_samples, n_params)
            
        Returns:
            dict: Dictionary containing coverage analysis
        """
        train_x_np = train_x.detach().numpy()
        n_samples, n_params = train_x_np.shape
        
        # Calculate coverage in each dimension
        coverage_per_dim = []
        for dim in range(n_params):
            param_values = train_x_np[:, dim]
            # Coverage is the range covered divided by total range [0, 1]
            coverage = (np.max(param_values) - np.min(param_values))
            coverage_per_dim.append(coverage)
        
        # Overall coverage (geometric mean to penalize poor coverage in any dimension)
        overall_coverage = np.exp(np.mean(np.log(np.maximum(coverage_per_dim, 1e-6))))
        
        # Uniformity measure (lower is more uniform)
        # Calculate pairwise distances and check for clustering
        if n_samples > 1:
            distances = pdist(train_x_np)
            uniformity = 1.0 / (1.0 + np.std(distances))  # Higher is more uniform
        else:
            uniformity = 0.0
        
        return {
            "coverage": overall_coverage,
            "coverage_per_dim": coverage_per_dim,
            "uniformity": uniformity
        }
    
    def _estimate_acquisition_diversity(self, train_x):
        """
        Estimate diversity of acquisition function sampling.
        
        Args:
            train_x (torch.Tensor): Normalized parameter values with shape (n_samples, n_params)
            
        Returns:
            float: Diversity metric (higher is more diverse)
        """
        train_x_np = train_x.detach().numpy()
        n_samples, n_params = train_x_np.shape
        
        if n_samples < 10:
            return 1.0  # Not enough data to estimate diversity
        
        # Look at recent samples (last 10 or 20% of samples, whichever is larger)
        recent_count = max(10, int(0.2 * n_samples))
        recent_samples = train_x_np[-recent_count:]
        
        # Calculate average distance between recent samples
        if len(recent_samples) > 1:
            recent_distances = pdist(recent_samples)
            avg_distance = np.mean(recent_distances)
            
            # Normalize by expected distance in unit hypercube
            # Expected distance between random points in [0,1]^d is approximately sqrt(d/6)
            expected_distance = np.sqrt(n_params / 6.0)
            diversity = avg_distance / expected_distance
        else:
            diversity = 0.0
        
        return min(diversity, 1.0)  # Cap at 1.0


def single_objective_bayesian_optimization(calib_context, train_x, train_obj, train_obj_true, train_obj_std, start_iteration):
    """Single-objective Bayesian optimization loop."""
    logger = calib_context.logger
    batch_size_bo = calib_context.batch_size_bo
    batch_size_per_iteration = calib_context.batch_size_per_iteration
    num_restarts = calib_context.num_restarts_act_func
    raw_samples = calib_context.raw_samples_act_func
    samples_per_batch = calib_context.samples_per_batch_act_func
    search_space = calib_context.search_space
    db_path = calib_context.db_path
    qoi_name = calib_context.qoi_details['QOI_Name'][0]
    hvs_list = []
    best_fitness_list = [torch.max(train_obj_true).item()]
    for iteration in range(start_iteration, batch_size_bo + 1):
        logger.info(f"{'='*60}")
        logger.info(f"🔄 Single-Objective BO Iteration {iteration}/{batch_size_bo}")
        logger.info(f"{'='*60}")
        # Cooperative cancellation check
        if getattr(calib_context, 'cancel_requested', False):
            logger.info("🛑 Cancellation requested — stopping single-objective optimization loop.")
            break
        # Fit GP model
        logger.info("🔧 Fitting Gaussian Process model...")
        train_y = train_obj[:, 0:1]
        train_yvar = train_obj_std[:, 0:1] ** 2
        train_yvar = torch.clamp(train_yvar, min=1e-6)
        model = SingleTaskGP(train_x, train_y, train_yvar)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        # Acquisition function: qLogExpectedImprovement
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([samples_per_batch]))
        acq_func = qLogExpectedImprovement(model=model, best_f=train_yvar.max().item(), sampler=sampler)
        num_params = len(search_space)
        bounds = torch.stack([torch.zeros(num_params), torch.ones(num_params)]).to(torch.float64)
        candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size_per_iteration,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        logger.info(f"🎯 Optimized acquisition function, best value = {acq_values.max():.6f}")
        next_sample_ids = [len(train_x) + i for i in range(len(candidates))]
        next_params_list = []
        for i, x in enumerate(candidates):
            x_unnorm = unnormalize_params(x, search_space)
            params_dict = tensor_to_param_dict(x_unnorm, search_space)
            next_params_list.append(params_dict)
            insert_samples(db_path, iteration, {next_sample_ids[i]: params_dict})
        with concurrent.futures.ProcessPoolExecutor(max_workers=calib_context.workers_out) as executor:
            new_results = list(executor.map(calib_context.evaluate_params, next_params_list, next_sample_ids))
        for i, (objectives, obj_noise, dic_results) in enumerate(new_results):
            calib_context.save_results_to_db(next_sample_ids[i], objectives, obj_noise, dic_results)
        new_obj_true = torch.tensor([list(result[0].values()) for result in new_results], dtype=torch.float64)
        new_obj_std = torch.tensor([list(result[1].values()) for result in new_results], dtype=torch.float64)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_obj_std
        new_obj = torch.clamp(new_obj, min=1e-3)
        train_x = torch.cat([train_x, candidates])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_true = torch.cat([train_obj_true, new_obj_true])
        train_obj_std = torch.cat([train_obj_std, new_obj_std])
        best_fitness = torch.max(train_obj_true).item()
        best_fitness_list.append(best_fitness)
        logger.info(f"✅ Completed iteration {iteration}/{batch_size_bo} - Total samples: {len(train_x)}")
        logger.info(f"🎯 Best fitness value for {qoi_name}: {best_fitness:.6f}")
    logger.info("✅ Single-objective Bayesian optimization completed successfully!")

def multi_objective_bayesian_optimization(calib_context, train_x, train_obj, train_obj_true, train_obj_std, start_iteration, latest_hypervolume, resume_from_db):
    """Multi-objective Bayesian optimization loop."""
    logger = calib_context.logger
    batch_size_bo = calib_context.batch_size_bo
    batch_size_per_iteration = calib_context.batch_size_per_iteration
    num_restarts = calib_context.num_restarts_act_func
    raw_samples = calib_context.raw_samples_act_func
    samples_per_batch = calib_context.samples_per_batch_act_func
    search_space = calib_context.search_space
    db_path = calib_context.db_path
    qoi_names = calib_context.qoi_details['QOI_Name']
    hvs_list = [latest_hypervolume] if resume_from_db else []
    for iteration in range(start_iteration, batch_size_bo + 1):
        logger.info(f"{'='*60}")
        logger.info(f"🔄 Multi-Objective BO Iteration {iteration}/{batch_size_bo}")
        logger.info(f"{'='*60}")
        # Cooperative cancellation check
        if getattr(calib_context, 'cancel_requested', False):
            logger.info("🛑 Cancellation requested — stopping multi-objective optimization loop.")
            break
        logger.info("🔧 Fitting Gaussian Process models...")
        model = _fit_gp_models(train_x, train_obj, train_obj_std, calib_context)
        logger.info("🎯 Optimizing acquisition function...")
        next_x = _optimize_acquisition_function(model, train_x, train_obj_true, calib_context)
        logger.info(f"📊 Evaluating {len(next_x)} new candidate(s)...")
        next_sample_ids = [len(train_x) + i for i in range(len(next_x))]
        next_params_list = []
        for i, x in enumerate(next_x):
            x_unnorm = unnormalize_params(x, search_space)
            params_dict = tensor_to_param_dict(x_unnorm, search_space)
            next_params_list.append(params_dict)
            insert_samples(db_path, iteration, {next_sample_ids[i]: params_dict})
        with concurrent.futures.ProcessPoolExecutor(max_workers=calib_context.workers_out) as executor:
            new_results = list(executor.map(calib_context.evaluate_params, next_params_list, next_sample_ids))
        for i, (objectives, obj_noise, dic_results) in enumerate(new_results):
            calib_context.save_results_to_db(next_sample_ids[i], objectives, obj_noise, dic_results)
        new_obj_true = torch.tensor([list(result[0].values()) for result in new_results], dtype=torch.float64)
        new_obj_std = torch.tensor([list(result[1].values()) for result in new_results], dtype=torch.float64)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_obj_std
        new_obj = torch.clamp(new_obj, min=1e-3)
        train_x = torch.cat([train_x, next_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_true = torch.cat([train_obj_true, new_obj_true])
        train_obj_std = torch.cat([train_obj_std, new_obj_std])
        try:
            cached_hv = calib_context._cached_pareto_data.get('hypervolume', None)
            current_hv = cached_hv
            logger.debug("🚀 Using cached hypervolume from acquisition function")
        except Exception:
            current_hv = None
            logger.warning("Could not retrieve hypervolume from acquisition function")
        hvs_list.append(current_hv)
        insert_gp_models(db_path, iteration, model, current_hv)
        logger.info(f"📊 Iteration {iteration} Sample(s) {next_sample_ids} : Hypervolume = {current_hv}")
        if len(hvs_list) >= 10:
            logger.info("🔍 Analyzing convergence...")
            convergence_result = calib_context.analyze_convergence(hvs_list, train_obj_true, train_x, iteration)
            logger.info(f"\t📋 Convergence Status: {convergence_result['status']}")
            logger.info(f"\t💡 Reason: {convergence_result['reason']}")
            logger.info(f"\t🎯 Confidence: {convergence_result['convergence_confidence']:.2%}")
            if convergence_result["suggestion"]:
                logger.info(f"\t💡 Suggestion: {convergence_result['suggestion']}")
            if convergence_result["converged"] and convergence_result["convergence_confidence"] > 0.8:
                logger.info("\t🎉 Convergence detected with high confidence - stopping optimization")
                break
            elif convergence_result["needs_restart"]:
                logger.warning("\t⚠️  Suboptimal stagnation detected - consider restarting with different settings")
        logger.info(f"✅ Completed iteration {iteration}/{batch_size_bo} - Total samples: {len(train_x)}")
        logger.info(f"🎯 Best fitness values: {[f'{qoi}: {fitness:.6f}' for qoi, fitness in zip(qoi_names, torch.max(train_obj_true, dim=0)[0].tolist())]}")
    logger.info("✅ Multi-objective Bayesian optimization completed successfully!")


def run_bayesian_optimization(calib_context: CalibrationContext, additional_iterations: Optional[int] = None):
    """
    Execute the complete Bayesian optimization process.
    
    Args:
        calib_context (CalibrationContext): The calibration context containing all configuration
        additional_iterations (Optional[int]): Additional iterations for resume functionality
    """
    logger = calib_context.logger
    try:
        resume_from_db = os.path.exists(calib_context.db_path)
        single_qoi = len(calib_context.qoi_details['QOI_Name']) == 1
        if resume_from_db:
            logger.info(f"🔄 Resuming optimization from existing database: {calib_context.db_path}")
            if additional_iterations is not None:
                calib_context.update_bo_iterations(additional_iterations)
            train_x, train_obj, train_obj_true, train_obj_std, latest_iteration, latest_hypervolume = calib_context.load_existing_data()
            start_iteration = latest_iteration + 1
        else:
            logger.info(f"🆕 Starting fresh optimization with database: {calib_context.db_path}")
            create_structure(calib_context.db_path)
            insert_metadata(calib_context.db_path, calib_context.dic_metadata)
            insert_param_space(calib_context.db_path, calib_context.search_space)
            insert_qois(calib_context.db_path, calib_context.qoi_details)
            logger.info(f"🎲 Generating {calib_context.num_initial_samples} initial samples...")
            train_x, train_obj, train_obj_true, train_obj_std = calib_context.generate_and_evaluate_samples(
                calib_context.num_initial_samples, start_sample_id=0, iteration_id=0
            )
            start_iteration = 1
            latest_hypervolume = 0.0
        if single_qoi:
            logger.info("🔬 Detected single QoI - using single-objective Bayesian optimization loop.")
            single_objective_bayesian_optimization(
                calib_context, train_x, train_obj, train_obj_true, train_obj_std, start_iteration
            )
        else:
            logger.info("🔬 Detected multiple QoIs - using multi-objective Bayesian optimization loop.")
            multi_objective_bayesian_optimization(
                calib_context, train_x, train_obj, train_obj_true, train_obj_std, start_iteration, latest_hypervolume, resume_from_db
            )
        
        # Remove temporary file and folders
        physicell_model = PhysiCell_Model(calib_context.model_config['ini_path'], calib_context.model_config['struc_name'])
        physicell_model.remove_io_folders()
        
    except Exception as e:
        logger.error(f"❌ Error during Bayesian optimization: {e}")
        raise


def _fit_gp_models(train_x: torch.Tensor, train_obj: torch.Tensor, train_obj_std: torch.Tensor, calib_context: CalibrationContext) -> ModelListGP:
    """
    Fit Gaussian Process models for multi-objective optimization.
    
    Args:
        train_x (torch.Tensor): Training inputs (normalized parameters)
        train_obj (torch.Tensor): Training objectives (fitness values)
        train_obj_std (torch.Tensor): Noise standard deviations
        calib_context (CalibrationContext): Calibration context
        
    Returns:
        ModelListGP: Fitted GP model list
    """
    models = []
    
    # Fit a separate GP for each objective
    for i in range(train_obj.shape[1]):
        # Extract single objective data
        train_y = train_obj[:, i:i+1]  # Keep 2D shape: (n_samples, 1)
        train_yvar = train_obj_std[:, i:i+1] ** 2  # Convert std to variance: (n_samples, 1)
        
        # Ensure minimum noise for numerical stability
        train_yvar = torch.clamp(train_yvar, min=1e-6)
        
        # Create GP model (following the old code pattern)
        model = SingleTaskGP(
            train_x,     # 2D: (n_samples, n_features)
            train_y,     # 2D: (n_samples, 1)
            train_yvar   # 2D: (n_samples, 1)
        )
        
        models.append(model)
        calib_context.logger.debug(f"Created GP for objective {i}")
    
    # Combine into ModelListGP
    model_list = ModelListGP(*models)
    
    # Create MLL object and fit it (following the old code pattern)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    fit_gpytorch_mll(mll)
    
    calib_context.logger.debug(f"Fitted {len(models)} GP models successfully")
    
    return model_list


def _optimize_acquisition_function(model: ModelListGP, train_x: torch.Tensor, train_obj_true: torch.Tensor, calib_context: CalibrationContext) -> torch.Tensor:
    """
    Optimize the acquisition function to find next candidate points.
    
    Args:
        model (ModelListGP): Fitted GP models
        train_x (torch.Tensor): Current training inputs
        train_obj_true (torch.Tensor): Current training objectives (true values)
        calib_context (CalibrationContext): Calibration context
        
    Returns:
        torch.Tensor: Next candidate points to evaluate
    """
    # Set up bounds for optimization (normalized space [0, 1]^d)
    num_params = len(calib_context.search_space)
    bounds = torch.stack([torch.zeros(num_params), torch.ones(num_params)]).to(torch.float64)
    
    # Create acquisition function (qNEHVI)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([calib_context.samples_per_batch_act_func]))
    
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=calib_context.ref_point,
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,  # Remove dominated points
        alpha=0.0,  # No risk aversion
    )
    
    # Apply acquisition function enhancement strategies
    enhanced_acq_func = _enhance_acquisition_function(acq_func, train_x, calib_context)
    
    # Extract Pareto data from acquisition function to avoid recomputation
    extracted_data = _extract_pareto_and_hypervolume_from_acqf(acq_func, calib_context)
    
    # Optimize acquisition function
    candidates, acq_values = optimize_acqf(
        acq_function=enhanced_acq_func,
        bounds=bounds,
        q=calib_context.batch_size_per_iteration,
        num_restarts=calib_context.num_restarts_act_func,
        raw_samples=calib_context.raw_samples_act_func,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,  # Use sequential optimization for batch
    )

    calib_context.logger.debug(f"Acquisition optimization with {extracted_data['n_pareto_points']} Pareto points: best value = {acq_values.max():.6f}")

    # Store extracted data for use in convergence analysis
    calib_context._cached_pareto_data = extracted_data
    
    return candidates


def _enhance_acquisition_function(base_acq_func, train_x: torch.Tensor, calib_context: CalibrationContext):
    """
    Enhance the acquisition function with identification strategies.
    
    Args:
        base_acq_func: Base acquisition function
        train_x (torch.Tensor): Current training inputs
        calib_context (CalibrationContext): Calibration context
        
    Returns:
        Enhanced acquisition function
    """
    strategy = calib_context.bo_options.get("acq_func_strategy", "none")
    
    if strategy == "none":
        return base_acq_func
    
    class EnhancedAcquisition:
        def __init__(self, base_func, strategy, train_x, options):
            self.base_func = base_func
            self.strategy = strategy
            self.train_x = train_x
            self.options = options
            
        def __call__(self, X):
            # Get base acquisition value
            base_value = self.base_func(X)
            
            # Apply enhancements based on strategy
            if self.strategy == "diversity_bonus":
                diversity_weight = self.options.get("diversity_weight", 0.1)
                diversity_bonus = self._compute_diversity_bonus(X)
                return base_value + diversity_weight * diversity_bonus
                
            elif self.strategy == "uncertainty_weighting":
                uncertainty_weight = self.options.get("uncertainty_weight", 0.2)
                uncertainty_bonus = self._compute_uncertainty_bonus(X)
                return base_value * (1.0 + uncertainty_weight * uncertainty_bonus)
                
            elif self.strategy == "combined":
                # Use both diversity and uncertainty
                diversity_weight = self.options.get("diversity_weight", 0.05)
                uncertainty_weight = self.options.get("uncertainty_weight", 0.15)
                
                diversity_bonus = self._compute_diversity_bonus(X)
                uncertainty_bonus = self._compute_uncertainty_bonus(X)
                
                enhancement = (diversity_weight * diversity_bonus + 
                             uncertainty_weight * uncertainty_bonus)
                
                return base_value + enhancement
                
            else:
                # For other strategies (soft_constraints, adaptive_scaling), just return base
                # These would require more complex implementations
                return base_value
        
        def _compute_diversity_bonus(self, X):
            """Compute diversity bonus to encourage exploration of new regions."""
            if len(self.train_x) == 0:
                return torch.ones(X.shape[0])
            
            # Compute minimum distance to existing points
            distances = torch.cdist(X, self.train_x, p=2)
            min_distances = distances.min(dim=1)[0]
            
            # Normalize and convert to bonus (higher distance = higher bonus)
            max_distance = np.sqrt(len(self.train_x))  # Approximate max distance in unit hypercube
            normalized_distances = min_distances / max_distance
            
            return normalized_distances
        
        def _compute_uncertainty_bonus(self, X):
            """Compute uncertainty bonus to encourage exploration of uncertain regions."""
            # This is a simplified implementation
            # In practice, you'd use the GP posterior variance
            if len(self.train_x) == 0:
                return torch.ones(X.shape[0])
            
            # Use distance as a proxy for uncertainty (farther = more uncertain)
            distances = torch.cdist(X, self.train_x, p=2)
            avg_distances = distances.mean(dim=1)
            
            # Normalize
            max_distance = np.sqrt(len(self.train_x))
            normalized_uncertainty = avg_distances / max_distance
            
            return normalized_uncertainty
    
    return EnhancedAcquisition(base_acq_func, strategy, train_x, calib_context.bo_options)


def _extract_pareto_and_hypervolume_from_acqf(acq_func, calib_context: CalibrationContext) -> dict:
    """
    Extract Pareto front points and hypervolume from BoTorch acquisition function.
    
    This avoids recomputing the same values that BoTorch already calculated internally.
    
    Args:
        acq_func: BoTorch acquisition function (qLogNoisyExpectedHypervolumeImprovement)
        calib_context: Calibration context for logging
        
    Returns:
        dict: Contains pareto_points, hypervolume, and related metrics, or None if extraction fails
    """
    try:
        # Extract Pareto points from the acquisition function's partitioning
        if not (hasattr(acq_func, 'partitioning') and acq_func.partitioning is not None and 
                hasattr(acq_func.partitioning, 'pareto_Y')):
            return None
            
        # Get Pareto points tensor
        pareto_y = acq_func.partitioning.pareto_Y
        if isinstance(pareto_y, list):
            pareto_points_tensor = pareto_y[0]  # Take first sample for analysis
        else:
            pareto_points_tensor = pareto_y
            
        # Handle extra dimensions
        if len(pareto_points_tensor.shape) > 2:
            pareto_points_tensor = pareto_points_tensor[0]
            
        pareto_points = pareto_points_tensor.detach().cpu().numpy()
        n_pareto = len(pareto_points)
        
        # Validate shape
        if len(pareto_points.shape) != 2:
            return None
            
        # Extract hypervolume
        mean_hypervolume = None
        try:
            if hasattr(acq_func, '_hypervolumes'):
                hypervolumes_tensor = acq_func._hypervolumes
                if hypervolumes_tensor.numel() > 1:
                    mean_hypervolume = float(hypervolumes_tensor.mean().detach().cpu())
                else:
                    mean_hypervolume = float(hypervolumes_tensor.detach().cpu())
        except Exception:
            raise ValueError("Failed to extract hypervolume from acquisition function.")
    
            
        # Calculate Pareto quality metrics
        n_objectives = pareto_points.shape[1]
        
        if n_pareto > 1:
            # Average fitness and distance to ideal point
            pareto_avg_fitness = np.mean(pareto_points)
            ideal_point = np.ones(n_objectives)
            distances_to_ideal = np.sqrt(np.sum((pareto_points - ideal_point)**2, axis=1))
            avg_distance_to_ideal = np.mean(distances_to_ideal)
            max_possible_distance = np.sqrt(n_objectives)
            
            # Combined quality measure (higher is better)
            pareto_quality = 0.7 * pareto_avg_fitness + 0.3 * (1.0 - avg_distance_to_ideal / max_possible_distance)
            pareto_quality = np.clip(pareto_quality, 0.0, 1.0)
            pareto_spread = np.std(pareto_points, axis=0).mean()
        else:
            pareto_quality = np.mean(pareto_points) if n_pareto == 1 else 0.0
            pareto_spread = 0.0
        
        return {
            "n_pareto_points": n_pareto,
            "pareto_ratio": n_pareto / max(n_pareto, 1),  # Safe division
            "pareto_quality": pareto_quality,
            "pareto_spread": pareto_spread,
            "hypervolume": mean_hypervolume,
        }
        
    except Exception as e:
        calib_context.logger.debug(f"⚠️ Failed to extract from acquisition function: {e}")
        return None