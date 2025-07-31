import sys, os
import logging
import concurrent.futures
import torch
import pandas as pd
import numpy as np
import time
import pickle
from typing import Union

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP # Each objective is a separate GP model
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.optim.optimize import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler

from uq_physicell import PhysiCell_Model
from .distances import Euclidean, Manhattan, Chebyshev, test_volume
from .utils import unnormalize_params, tensor_to_param_dict, normalize_params, param_dict_to_tensor
from .database import create_structure, insert_metadata, insert_param_space, insert_qois, insert_gp_models, insert_samples, insert_output, load_structure
from uq_physicell.utils.model_wrapper import run_replicate_serializable

class CalibrationContext:
    """
    Context for Bayesian Optimization calibration, encapsulating all necessary parameters and configurations.
    This class is designed to be passed around to functions that require access to the calibration parameters,
    model configurations, and other relevant data.
    Attributes:
        db_path (str): Path to the database file for storing and retrieving samples.
        obsData (str or dict): Path or dict containing the observed data.
        obsData_columns (dict): Dictionary mapping QoI names to their corresponding columns in the observed data.
        model_config (dict): Configuration dictionary for the PhysiCell model, including paths and structure names.
        qoi_functions (dict): Dictionary of functions to compute quantities of interest (QoIs) from model outputs.
        distance_functions (dict): Dictionary of functions to compute distances between model outputs and observed data.
        search_space (dict): Dictionary defining the search space for parameters, including bounds and types.
        hyperparams (dict): Hyperparameters for the optimization process, such as regularization terms.
        bo_options (dict): Options for Bayesian Optimization, including scale factors and sampling parameters.
        logger (logging.Logger): Logger instance for logging messages during the calibration process.
    """
    def __init__(self, db_path:str, obsData:Union[str,dict], obsData_columns:dict, model_config:dict, qoi_functions:dict, distance_functions:dict, search_space:dict, hyperparams:dict, bo_options:dict, logger:logging.Logger):
        self.db_path = db_path
        if isinstance(obsData, dict):
            self.dic_obsData = obsData
            self.obsData_path = None
        else: # obsData is a path
            try:
                self.obsData_path = obsData
                self.dic_obsData = pd.read_csv(obsData).to_dict('list')
                # replace columns names equivalent to obsData_columns
                for qoi, column_name in obsData_columns.items():
                    if column_name in self.dic_obsData:
                        self.dic_obsData[qoi] = np.array(self.dic_obsData.pop(column_name), dtype=np.float64)
                    else:
                        raise ValueError(f"Column {column_name} not found in observed data.")
            except Exception as e:
                logger.error(f"Error reading observed data from {self.obsData_path}: {e}")
                sys.exit(1)
        self.model_config = model_config
        self.qoi_functions = qoi_functions
        self.distance_functions = distance_functions
        self.search_space = search_space
        self.hyperparams = hyperparams

        # Additional fixed parameters that .ini file expects
        self.fixed_params = bo_options.get("fixed_params", {})
        # Custom summary function for the PhysiCell model
        self.summary_function = bo_options.get("summary_function", None)
        # Custom run_single_replicate function for the PhysiCell model
        self.custom_run_single_replicate_func = bo_options.get("custom_run_single_replicate_func", None)
        # Custom aggregation function for the results
        self.custom_aggregation_func = bo_options.get("custom_aggregation_func", None)
        #  A tensor with m elements representing the reference point (in the outcome space) w.r.t. to which compute the hypervolume. This is a reference point for the outcome
        # Using fitness values (1/(1+distance)), ref_point should be below the worst expected fitness
        # Fitness values are in range (0, 1] -> fitness 1 is the best (distance=0), approaches 0 for very large distances
        # Setting ref_point to 0 ensures all fitness values contribute to hypervolume
        self.ref_point = torch.tensor([0.0] * (len(qoi_functions) + len(hyperparams)), dtype=torch.float64)  # Reference point for fitness values
        self.logger = logger
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

        # Initialize metadata for database
        self.dic_metadata = {
            "BO_Method": "Multi-objective optimization with qNEHVI",
            "ObsData_Path": self.obsData_path,
            "Ini_File_Path": self.model_config["ini_path"],
            "StructureName": self.model_config["struc_name"],
            "Scale_Factor": bo_options.get("scale_factor", 1.0),  # Fixed scale factor
        }
        if "l1" in self.hyperparams.keys():
            self.dic_metadata["HyperParam_l1"] = self.hyperparams["l1"]
        if "l2" in self.hyperparams.keys():
            self.dic_metadata["HyperParam_l2"] = self.hyperparams["l2"]

        # Initialize the qoi_details
        self.qoi_details = {
            "QOI_Name": list(self.qoi_functions.keys()),
            "QOI_Function": [self.qoi_functions[key] for key in self.qoi_functions.keys()],
            "ObsData_Column": [obsData_columns[key] for key in self.qoi_functions.keys()],
            "QoI_distanceFunction": [self.distance_functions[key]['function'].__name__ for key in self.distance_functions.keys()],
            "QoI_distanceWeight": [self.distance_functions[key]['weight'] for key in self.distance_functions.keys()],
        }

        self.logger.info(f"CalibrationContext initialized with {self.max_workers} max workers, {self.workers_inner} inner workers, and {self.workers_out} outer workers.")

    def default_run_single_replicate(self, sample_id:int, replicate_id:int, params:dict) -> dict:
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
            self.model_config["ini_path"], 
            self.model_config["struc_name"], 
            sample_id, replicate_id, 
            dic_params_xml, dic_params_rules, 
            qois_dic=self.qoi_functions, return_binary_output=False,
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
                distance = dist_info["weight"] * dist_info["function"](dicObsData, dicModel)
                # Convert distance to fitness using a more gradual transformation
                # Option 1: 1/(1+distance/scale) - current approach
                # Option 2: exp(-distance/scale) - with scaling factor
                # self.dic_metadata['Scale_Factor'] adjust this based on typical distance ranges
                scaled_distance = distance / self.dic_metadata['Scale_Factor']
                fitness = 1.0 / (1.0 + scaled_distance)
                replicate_objectives[qoi] = fitness
                # Debug: log distance and fitness values
                if sample_id < 3:  # Only for first few samples to avoid log spam
                    self.logger.info(f"\t sampleID: {sample_id} replicateID: {replicate_id} - QoI '{qoi}': distance={distance:.3f}, scaled_dist={scaled_distance:.3f}, fitness={fitness:.6f}")
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
        self.logger.info(f"Evaluating sample {sample_index} with params: {params}")
        # Check if using default run_single_replicate function
        if not self.custom_run_single_replicate_func:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers_inner) as executor:
                replicate_results = list(executor.map(
                    self.default_run_single_replicate,
                    [sample_index] * self.num_replicates,
                    range(self.num_replicates),
                    [params] * self.num_replicates
                ))
        else: # Use the custom run_single_replicate function provided in bo_options
            # Note that this function signature MUST be:  
            # sample_index:int, replicate_id:int,params:dict, fixed_params:dict, model_config:dict
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
        else: # Custom aggregation function
            objectives, obj_noise, dic_results = self.custom_aggregation_func(replicate_results, sample_index, self.distance_functions, self.dic_obsData, self.dic_metadata['Scale_Factor'])

        # Add l1 regularization to the objectives if specified
        if "l1" in self.hyperparams.keys():
            dic_params_diff = {key: (1.0/(self.search_space[key]["upper_bound"]-self.search_space[key]["lower_bound"]))*(params[key] - self.search_space[key]["default_value"]) for key in self.search_space.keys()}
            objectives["l1_reg"] = self.hyperparams["l1"] * sum(abs(dic_params_diff[key]) for key in self.search_space.keys())
            obj_noise["l1_reg"] = 0.0  # l1_reg no noise in parameters values

        # Add l2 regularization to the objectives if specified
        if "l2" in self.hyperparams.keys():
            dic_params_diff = {key: (1.0/(self.search_space[key]["upper_bound"]-self.search_space[key]["lower_bound"]))*(params[key] - self.search_space[key]["default_value"]) for key in self.search_space.keys()}
            objectives["l2_reg"] = self.hyperparams["l2"] * sum(dic_params_diff[key]**2 for key in self.search_space.keys())
            obj_noise["l2_reg"] = 0.0

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
            self.logger.info(f"Successfully saved results for sample {sample_index} to database.")
        except Exception as e:
            self.logger.error(f"Error saving results for sample {sample_index} to database: {e}")
            raise

    def generate_initial_samples(self, num_samples:int) -> tuple:
        """
        Generate initial samples for Bayesian optimization using Sobol sequences.
        This function generates a set of initial samples in the search space using Sobol sequences.
        Args:
            num_samples (int): Number of initial samples to generate.
        Returns:
            tuple: A tuple containing the generated samples and their corresponding sample IDs.
        """
        num_params = len(self.search_space)
        bounds = torch.stack([torch.zeros(num_params), torch.ones(num_params)]) # 0 to 1 bounds for each parameter
        sample_ids = np.arange(num_samples, dtype=int)
        train_x = draw_sobol_samples(bounds, n=num_samples, q=1).squeeze(1).to(torch.float64)  # Convert to float64 and squeeze q dimension
        train_x_dic_params = []
        for i in range(num_samples):
            train_x_unnorm_i = unnormalize_params(train_x[i], self.search_space)
            dic_params_i = tensor_to_param_dict(train_x_unnorm_i, self.search_space)
            # save parameters in the database
            insert_samples(self.db_path, 0, {sample_ids[i]: dic_params_i})
            train_x_dic_params.append(dic_params_i)

        self.logger.info(f"Running {num_samples} initial samples...")
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

        return train_x, train_obj, train_obj_true, train_obj_std

    def load_existing_data(self) -> tuple:
        """
        Load existing data from the database for resume functionality.
        Returns:
            tuple: A tuple containing training data tensors, latest iteration, and hypervolume.
        """
        self.logger.info("Loading existing data from database...")
        
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
            
        self.logger.info(f"Loaded {len(train_x)} samples from {latest_iteration + 1} iterations")
        self.logger.info(f"Latest hypervolume: {latest_hypervolume}")
        
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
            
        self.logger.info("Loaded data validation passed")

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
            
            # Add hyperparameter objectives if present
            for hyperparam in self.hyperparams.keys():
                if hyperparam in objectives:
                    obj_values.append(objectives[hyperparam])
                    std_values.append(noise_std[hyperparam])
            
            obj_tensor = torch.tensor(obj_values, dtype=torch.float64)
            std_tensor = torch.tensor(std_values, dtype=torch.float64)
            
            train_obj_true_list.append(obj_tensor)
            train_obj_std_list.append(std_tensor)
            
            # Add noise to create train_obj (same as in original code)
            noisy_obj = obj_tensor + torch.randn_like(obj_tensor) * std_tensor
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
        self.logger.info(f"Updated max iterations from {original_iterations} to {self.batch_size_bo} (added {additional_iterations} iterations)")


    def initialize_modelgp(self,train_x:torch.Tensor, train_obj:torch.Tensor, train_obj_std:torch.Tensor, iteration_id:int, hypervolume:float, insertOnDataBase:bool=True) -> tuple:
        """ Initialize a Gaussian Process model for Bayesian optimization.
        Args:
            train_x (torch.Tensor): Input features for training.
            train_obj (torch.Tensor): Target values for training.
            train_obj_std (torch.Tensor): Standard deviation of the target values for training.
            iteration_id (int): Current iteration of the optimization process.
            hypervolume (float): The hypervolume value for this iteration.
        Returns:
            tuple: A tuple containing the GP model and the scaler.
        """
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i : i + 1]  # Shape: (n_samples, 1)
            train_yvar = train_obj_std[..., i : i + 1] ** 2  # Shape: (n_samples, 1)
            
            models.append(
                SingleTaskGP(
                    train_x,     # 2D: (n_samples, n_features)
                    train_y,     # 2D: (n_samples, 1)
                    train_yvar   # 2D: (n_samples, 1)
                )
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # save the GP model to the database with hypervolume
        if insertOnDataBase:
            insert_gp_models(self.db_path, iteration_id, model, hypervolume)
        return mll, model

    def optimize_qnehvi_and_get_observation(self, model:ModelListGP, train_x: torch.Tensor, train_obj: torch.Tensor, sampler:SobolQMCNormalSampler, sample_id: int, iteration_id: int):
        """Optimizes the qEHVI acquisition function, and returns a new candidate and observation.
        Args:
            model (ModelListGP): The GP model to use for optimization.
            train_x (torch.Tensor): Training input features.
            train_obj (torch.Tensor): Training objective values.
            sampler (SobolQMCNormalSampler): Sampler for generating samples.
            sample_id (int): Unique identifier for the sample being processed.
            iteration_id (int): Current iteration of the optimization process.
        Returns:
            tuple: A tuple containing the new candidate parameters, observed objectives, true objectives, and standard deviations.
        """ 
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,  # use known reference point
            X_baseline=train_x,  # Use train_x directly (2D format for acquisition)
            prune_baseline=False,  # Keep disabled for stability
            sampler=sampler,
        )
        # optimize
        num_params = len(self.search_space)
        standard_bounds = torch.stack([torch.zeros(num_params), torch.ones(num_params)]) # 0 to 1 bounds for each parameter
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.batch_size_per_iteration,  # batch size for parallel processing
            num_restarts=self.num_restarts_act_func,
            raw_samples=self.raw_samples_act_func,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        # observe new values
        new_x = candidates.detach().to(torch.float64)  # Ensure float64 consistency
        
        # SIMPLIFIED TENSOR HANDLING: With 2D model consistency, we just need to ensure new_x is 2D
        if new_x.dim() == 3 and new_x.shape[1] == 1:
            new_x = new_x.squeeze(1)  # Common case: (batch_size, 1, n_features) -> (batch_size, n_features)
        elif new_x.dim() == 1:
            new_x = new_x.unsqueeze(0)  # Single candidate: (n_features,) -> (1, n_features)
        elif new_x.dim() != 2:
            raise ValueError(f"Unexpected candidates shape: {new_x.shape}")
            
        # new_x should now be 2D: (batch_size, n_features)
        new_x_unnorm = unnormalize_params(new_x, self.search_space)
        new_x_dic_params = tensor_to_param_dict(new_x_unnorm, self.search_space)
        # save the new parameters to the database
        insert_samples(self.db_path, iteration_id, {sample_id: new_x_dic_params})
        # run the model with the new parameters and get the observed objectives
        dic_objectives, dic_noise_std, dic_results = self.evaluate_params(new_x_dic_params, sample_id)
        # save results to database
        self.save_results_to_db(sample_id, dic_objectives, dic_noise_std, dic_results)

        new_obj_true = torch.tensor([list(dic_objectives.values())], dtype=torch.float64)
        new_obj_std = torch.tensor([list(dic_noise_std.values())], dtype=torch.float64)
        new_obj = new_obj_true + torch.randn_like(new_obj_true) * new_obj_std
        return new_x, new_obj, new_obj_true, new_obj_std

# --- Main BO calibration routine ---
def run_bayesian_optimization(calib_context:CalibrationContext, additional_iterations: int = 0):
    """ Run Bayesian Optimization for model calibration.
    This function initializes the Bayesian optimization process, loads existing samples if available,
    and performs the optimization iterations to find the best parameters that minimize the loss function.
    Args:
        calib_context (CalibrationContext): Context object containing all necessary parameters and configurations for the calibration.
        additional_iterations (int): Number of additional iterations to run if resuming from existing database.
    """
    hvs_list = []  # List to store hypervolume values for each iteration
    start_iteration = 1  # Default start iteration for new runs
    
    if not os.path.exists(calib_context.db_path):
        # Create a new database file
        calib_context.logger.info(f"Creating a new database file for Bayesian optimization: {calib_context.db_path}.")
        create_structure(calib_context.db_path)
        
        # Insert the metadata and qois into the database
        insert_metadata(calib_context.db_path, calib_context.dic_metadata)
        insert_param_space(calib_context.db_path, calib_context.search_space)
        insert_qois(calib_context.db_path, calib_context.qoi_details)

        # generate initial training data and initialize model
        train_x, train_obj, train_obj_true, train_obj_std  = calib_context.generate_initial_samples(num_samples=calib_context.num_initial_samples)
        
        # compute hypervolume using test_volume function
        volume = test_volume(calib_context.ref_point, train_obj_true, calib_context.dic_metadata["Scale_Factor"], iteration=0, logger=calib_context.logger)
        hvs_list.append(volume)
        
        mll, modelgp = calib_context.initialize_modelgp(train_x, train_obj, train_obj_std, iteration_id=0, hypervolume=volume)
        
        calib_context.logger.info(f"Initial fitness values - Min: {train_obj_true.min(dim=0)[0]}, Max: {train_obj_true.max(dim=0)[0]}, Mean: {train_obj_true.mean(dim=0)}")
        calib_context.logger.info(f"Reference point: {calib_context.ref_point}")
        calib_context.logger.info(f"Initial hypervolume: {volume}")

    else:
        # Load existing database and resume optimization
        calib_context.logger.info(f"Database file {calib_context.db_path} already exists. Loading existing samples and GP model.")
        
        if additional_iterations > 0:
            calib_context.update_bo_iterations(additional_iterations)
        
        try:
            train_x, train_obj, train_obj_true, train_obj_std, latest_iteration, latest_hypervolume = calib_context.load_existing_data()
            
            # Set starting iteration for resume
            start_iteration = latest_iteration + 1
            
            # Hypervolume list (simplified - just use latest value)
            hvs_list = [latest_hypervolume]
            
            # Always recreate the GP model from the loaded training data
            # This is more robust than trying to deserialize the model from the database
            calib_context.logger.info("Recreating GP model from loaded training data...")
            mll, modelgp = calib_context.initialize_modelgp(train_x, train_obj, train_obj_std, iteration_id=latest_iteration, hypervolume=latest_hypervolume, insertOnDataBase=False)
            
            calib_context.logger.info(f"Resuming from iteration {start_iteration} with {len(train_x)} existing samples")
            calib_context.logger.info(f"Current fitness values - Min: {train_obj_true.min(dim=0)[0]}, Max: {train_obj_true.max(dim=0)[0]}, Mean: {train_obj_true.mean(dim=0)}")
            calib_context.logger.info(f"Latest hypervolume: {latest_hypervolume}")
                
        except Exception as e:
            calib_context.logger.error(f"Error loading existing data: {e}")
            raise ValueError(f"Failed to load existing data. Given database: {calib_context.db_path} may be corrupted or incompatible.")

    # run batch_size_bo rounds of BayesOpt after the initial random batch or resume point
    for iteration in range(start_iteration, calib_context.batch_size_bo + 1):
        calib_context.logger.info(f"Starting Bayesian optimization iteration {iteration}.")
        # Track the time taken for each iteration
        t0 = time.monotonic()

        # fit the models
        fit_gpytorch_mll(mll)

        # define the qNEI acquisition modules using a QMC sampler
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([calib_context.samples_per_batch_act_func]))

        # optimize acquisition functions and get new observations
        sample_id = len(train_x)  # Use current number of samples as the new sample ID
        (
            new_x,
            new_obj,
            new_obj_true,
            new_obj_std,
        ) = calib_context.optimize_qnehvi_and_get_observation(
            modelgp, train_x, train_obj, sampler, sample_id, iteration
        )
        
        # update training points
        # CORRECTED: All tensors are consistently 2D throughout the BoTorch pipeline
        # This matches the successful "Testing CORRECT 2D Approach" from test_botorch_dims.py
        
        # Validate tensor dimensions before concatenation
        if train_x.dim() != 2:
            raise ValueError(f"train_x should be 2D, got shape: {train_x.shape}")
        if new_x.dim() != 2:
            raise ValueError(f"new_x should be 2D, got shape: {new_x.shape}")
            
        train_x = torch.cat([train_x, new_x], dim=0)  # Concatenate along batch dimension
        train_obj = torch.cat([train_obj, new_obj], dim=0)
        train_obj_true = torch.cat([train_obj_true, new_obj_true], dim=0)
        train_obj_std = torch.cat([train_obj_std, new_obj_std], dim=0)

        # compute hypervolume using test_volume function
        volume = test_volume(calib_context.ref_point, train_obj_true, calib_context.dic_metadata["Scale_Factor"], iteration=iteration, logger=calib_context.logger)
        hvs_list.append(volume)
        
        calib_context.logger.info(f"Iteration {iteration} - Current fitness values - Min: {train_obj_true.min(dim=0)[0]}, Max: {train_obj_true.max(dim=0)[0]}, Mean: {train_obj_true.mean(dim=0)}")

        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration - (https://github.com/pytorch/botorch/blob/main/tutorials/multi_objective_bo/multi_objective_bo.ipynb)
        mll, modelgp = calib_context.initialize_modelgp(train_x, train_obj, train_obj_std, iteration_id=iteration, hypervolume=volume)

        # Finish the iteration and log the results
        t1 = time.monotonic()

        calib_context.logger.info(
            f"\nBatch {iteration:>2}: Hypervolume (qNEHVI) = {hvs_list[-1]:>2.4e}), "
            f"time = {t1-t0:>4.2f}."
        )

if __name__ == "__main__":
    # Example usage of the Bayesian Optimization calibration routine
    db_path = "examples/virus-mac-new/BO_calibration.db"  # Path to the database file
    obs_data_path = "examples/virus-mac-new/cell_counts.csv"  # Path to the observed data file
    model_config = {"ini_path": "examples/virus-mac-new/uq_pc_struc.ini", "struc_name": "SA_struc", "numReplicates": 2}
    qoi_functions = {"epi_": "lambda df: len(df[df['cell_type'] == 'epithelial'])", 
                     "epi_infected": "lambda df: len(df[df['cell_type'] == 'epithelial_infected'])"}  # Example QoI functions
    obs_data_columns = {'time': "Time", "epi_": "Healthy Epithelial Cells", "epi_infected": "Infected Epithelial Cells"}
    distance_functions = {"epi_": {"function": Euclidean, "weight": 1.0},
                "epi_infected": {"function": Euclidean, "weight": 1.0}}
    # If using regularization, the hyperparameters and default parameters should be defined, 
    # defaults_parameters should be in the range of each parameter and defined in the search_space.
    search_space = {"mac_phag_rate_infected": {"type": "real", "lower_bound": 0.8, "upper_bound": 1.2},
                    "mac_motility_bias": {"type": "real", "lower_bound": 0.12, "upper_bound": 0.18},
                    "epi2infected_sat": {"type": "real", "lower_bound": 0.08, "upper_bound": 0.12},
                    "epi2infected_hfm": {"type": "real", "lower_bound": 0.32, "upper_bound": 0.48},
                    }  # Example search space
    hyperparams = {}#{"l1": 0.01, "l2": 0.01}  # If not using regularization should be empty
    bo_options = {
        "num_initial_samples": 10,  # Number of initial samples for Bayesian optimization
        "num_iterations": 5,  # Number of iterations for Bayesian optimization
        "max_workers": 8,  # Number of parallel workers
        "scale_factor": 100.0,  # Scale factor for distance functions
    }
    # logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger()

    # Create the calibration context
    calib_context = CalibrationContext(
        db_path=db_path,
        obsData_path=obs_data_path,
        obsData_columns=obs_data_columns,
        model_config=model_config,
        qoi_functions=qoi_functions,
        distance_functions=distance_functions,
        search_space=search_space,
        hyperparams=hyperparams,
        bo_options=bo_options,
        logger=logger
    )

    # Run the Bayesian Optimization calibration s
    # Using the enhanced main function for resume
        
    # Option 1: Run with current settings (default behavior)
    # This will either start fresh or resume with original iteration count
    # run_bayesian_optimization(calib_context)

    # Option 2: Resume with additional iterations (if database exists)
    # This will add 10 more iterations to your existing optimization
    run_bayesian_optimization(calib_context, additional_iterations=50)

    # Option 3: Example of checking database existence before deciding
    # if os.path.exists(db_path):
    #     logger.info("Database exists, resuming with 5 additional iterations")
    #     run_bayesian_optimization(calib_context, additional_iterations=5)
    # else:
    #     logger.info("No database found, starting fresh optimization")
    #     run_bayesian_optimization(calib_context)
