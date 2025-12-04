import pandas as pd
from typing import Union
import torch
from ..database.bo_db import load_structure

def normalize_params_df(df_params, df_search_space) -> pd.DataFrame:
    """
    Normalize the parameters values based on the search space.
    Parameters:
    - df_params: DataFrame containing the parameters to normalize.
    - df_search_space: DataFrame defining the search space for each parameter.
    Returns:
    - DataFrame with normalized parameters in the range [0, 1].
    """
    # Merge parameter values with their corresponding bounds
    df_merged = pd.merge(df_params, df_search_space, on='ParamName', how='left')
    # Normalize each parameter value
    df_merged['ParamValue'] = (df_merged['ParamValue'] - df_merged['lower_bound']) / (df_merged['upper_bound'] - df_merged['lower_bound'])
    # Return only the relevant columns
    df_norm = df_merged[['SampleID', 'ParamName', 'ParamValue']]
    return df_norm

def extract_best_parameters(df_gp_models: pd.DataFrame, df_samples: pd.DataFrame) -> tuple:
    """
    Extract the best parameters from the database file based on the maximum hypervolume.
    Parameters:
    - df_gp_models: DataFrame containing the Gaussian Process models.
    - df_samples: DataFrame containing the samples.
    Returns:
    - Dictionary with parameter names as keys and their best values as values.
    - The sample ID corresponding to the best parameters.
    """
    # Find the maximum hypervolume and its corresponding iteration ID (if multiple, take the first)
    max_hypervolume = df_gp_models['Hypervolume'].max()
    best_iteration = df_gp_models[df_gp_models['Hypervolume'] == max_hypervolume]['IterationID'].min()
    best_sample_id = df_samples[df_samples['IterationID'] == best_iteration]['SampleID'].values[0] # Assuming one sample per iteration

    # Get the parameters for the best iteration
    best_params = df_samples[df_samples['IterationID'] == best_iteration].set_index('ParamName')['ParamValue'].to_dict()

    return best_params, best_sample_id

def extract_best_parameters_db(db_file:str) -> tuple:
    """
    Extract the best parameters from the database file based on the maximum hypervolume.
    Parameters:
    - db_file: Path to the database file.
    Returns:
    - Dictionary with parameter names as keys and their best values as values.
    - The sample ID corresponding to the best parameters.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)
    
    return extract_best_parameters(df_gp_models, df_samples)

def extract_all_pareto_points(df_qois: pd.DataFrame, df_samples: pd.DataFrame, df_output: pd.DataFrame) -> dict:
    """
    Extract all Pareto points from the optimization database.
    
    Args:
        db_path (str): Path to the database file
        
    Returns:
        dict: Contains all Pareto points and their corresponding sample IDs and parameters
    """
    import torch
    import numpy as np
    from botorch.utils.multi_objective.pareto import is_non_dominated
    
    # Extract all fitness values (objectives) to compute Pareto points
    all_fitness_values = []
    sample_ids = []
    parameter_values = []
    
    if not df_output.empty:
        for _, row in df_output.iterrows():
            sample_id = row['SampleID']
            objectives = row['ObjFunc']  # This is a pickled dictionary
            
            # Convert objectives dict to list (maintain order)
            qoi_names = df_qois['QoI_Name'].tolist()
            fitness_vals = [objectives[qoi] for qoi in qoi_names]
            
            all_fitness_values.append(fitness_vals)
            sample_ids.append(sample_id)
            
            # Get corresponding parameters
            sample_params = df_samples[df_samples['SampleID'] == sample_id]
            param_dict = {}
            for _, param_row in sample_params.iterrows():
                param_dict[param_row['ParamName']] = param_row['ParamValue']
            parameter_values.append(param_dict)
    
    # Convert to numpy array for Pareto analysis
    if all_fitness_values:
        fitness_array = np.array(all_fitness_values)
        
        # Use BoTorch to find Pareto points
        fitness_tensor = torch.tensor(fitness_array, dtype=torch.float64)
        pareto_mask = is_non_dominated(fitness_tensor, maximize=True, deduplicate=True)
        
        # Extract Pareto optimal data
        pareto_indices = torch.where(pareto_mask)[0].numpy()
        pareto_fitness_values = fitness_array[pareto_indices]
        pareto_sample_ids = [sample_ids[i] for i in pareto_indices]
        pareto_parameters = [parameter_values[i] for i in pareto_indices]
        
    else:
        pareto_fitness_values = np.array([])
        pareto_sample_ids = []
        pareto_parameters = []
    
    return {
        "pareto_front": {
            "fitness_values": pareto_fitness_values,
            "sample_ids": pareto_sample_ids,
            "parameters": pareto_parameters,
            "n_points": len(pareto_sample_ids)}
        }

def analyze_pareto_results(df_qois, df_samples, df_output):
    """
    Comprehensive analysis of Pareto results from the optimization.
    """
    print("=" * 60)
    print("🔍 COMPREHENSIVE PARETO ANALYSIS")
    print("=" * 60)
    
    # Extract all Pareto points
    print("\n📊 Extracting all Pareto points ...")
    all_data = extract_all_pareto_points(df_qois, df_samples, df_output)
    
    print(f"🎯 Current Pareto front size: {all_data['pareto_front']['n_points']}")
    
    # Display current Pareto front
    print("\n🎯 CURRENT PARETO FRONT:")
    if all_data['pareto_front']['n_points'] > 0:
        print(f"   Number of Pareto optimal points: {all_data['pareto_front']['n_points']}")
        print("   Sample IDs:", all_data['pareto_front']['sample_ids'])
        
        print("\n   📋 Pareto Front Details:")
        for i, (sample_id, fitness, params) in enumerate(zip(
            all_data['pareto_front']['sample_ids'],
            all_data['pareto_front']['fitness_values'],
            all_data['pareto_front']['parameters']
        )):
            print(f"   Point {i+1} (Sample {sample_id}):")
            print(f"      Fitness: {dict(zip(df_qois['QoI_Name'], fitness))}")
            print(f"      Parameters: {params}")
    else:
        print("   ⚠️ No Pareto points found!")
    
    return all_data

def get_observed_qoi(obsDataFile:str, df_qois:pd.DataFrame) -> pd.DataFrame:
    """
    Load the observed QoI values from a CSV file and rename the columns to match the QoI names.
    Parameters:
    - obsDataFile: Path to the observed QoI data file.
    - df_qois: DataFrame containing the QoI definitions.
    Returns:
    - DataFrame with SampleID as index and observed QoI values as columns.
    """
    # Load the Observed QoI values
    df_obs_qoi = pd.read_csv(obsDataFile)
    # Rename the columns to match the QoI names
    dic_columns = df_qois.set_index('ObsData_Column')['QoI_Name'].to_dict()
    if "Time" in df_obs_qoi.columns:
        dic_columns["Time"] = "time"
    df_obs_qoi = df_obs_qoi.rename(columns=dic_columns)
    return df_obs_qoi

def normalize_params(params: torch.Tensor, search_space: dict) -> torch.Tensor:
    """
    Normalize parameters to the range [0, 1] based on the search space.
    Args:
        params (torch.Tensor): Tensor containing parameter values.
        search_space (dict): Dictionary defining the search space.
    Returns:
        torch.Tensor: Normalized tensor of parameters in the range [0, 1].
    """
    normalized_params = []
    # Handle both 1D and 2D tensors by flattening if necessary
    if params.dim() > 1:
        params = params.squeeze()
    params_list = params.tolist()
    
    for i, param in enumerate(search_space.keys()):
        if search_space[param]["type"] == "integer":
            normalized_value = (params_list[i] - search_space[param]["lower_bound"]) / (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])
        elif search_space[param]["type"] == "real":
            normalized_value = (params_list[i] - search_space[param]["lower_bound"]) / (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])
        else:
            raise ValueError(f"Unknown parameter type: {search_space[param]['type']}")
        normalized_params.append(normalized_value)
    return torch.tensor(normalized_params, dtype=torch.double)

def unnormalize_params(normalized_params: torch.Tensor, search_space: dict) -> torch.Tensor:
    """
    Unnormalize parameters from the range [0, 1] back to their original scale.
    Supports both single parameter vectors and batches of parameter vectors.
    
    Args:
        normalized_params (torch.Tensor): Tensor containing normalized parameter values.
                                        Shape: (n_params,) for single vector or (batch_size, n_params) for batch
        search_space (dict): Dictionary defining the search space.
    Returns:
        torch.Tensor: Unnormalized tensor of parameters with same shape as input.
    """
    if normalized_params.dim() == 1:
        # Single parameter vector case
        unnormalized_params = []
        normalized_params_list = normalized_params.tolist()
        
        for i, param in enumerate(search_space.keys()):
            if search_space[param]["type"] == "integer":
                unnormalized_value = int(round(search_space[param]["lower_bound"] + normalized_params_list[i] * (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])))
            elif search_space[param]["type"] == "real":
                unnormalized_value = search_space[param]["lower_bound"] + normalized_params_list[i] * (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])
            else:
                raise ValueError(f"Unknown parameter type: {search_space[param]['type']}")
            unnormalized_params.append(unnormalized_value)
        return torch.tensor(unnormalized_params, dtype=torch.double)
    
    elif normalized_params.dim() == 2:
        # Batch of parameter vectors case
        batch_size, n_params = normalized_params.shape
        unnormalized_batch = []
        
        for batch_idx in range(batch_size):
            single_params = normalized_params[batch_idx]
            unnormalized_single = []
            single_params_list = single_params.tolist()
            
            for i, param in enumerate(search_space.keys()):
                if search_space[param]["type"] == "integer":
                    unnormalized_value = int(round(search_space[param]["lower_bound"] + single_params_list[i] * (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])))
                elif search_space[param]["type"] == "real":
                    unnormalized_value = search_space[param]["lower_bound"] + single_params_list[i] * (search_space[param]["upper_bound"] - search_space[param]["lower_bound"])
                else:
                    raise ValueError(f"Unknown parameter type: {search_space[param]['type']}")
                unnormalized_single.append(unnormalized_value)
            unnormalized_batch.append(unnormalized_single)
        
        return torch.tensor(unnormalized_batch, dtype=torch.double)
    
    else:
        raise ValueError(f"unnormalize_params expects 1D or 2D tensors, got {normalized_params.dim()}D tensor with shape {normalized_params.shape}")

def tensor_to_param_dict(params_tensor: torch.Tensor, search_space: dict):
    """
    Convert a tensor of parameters to parameter dictionary(ies) based on the search space.
    Supports both single parameter vectors and batches of parameter vectors.
    
    Args:
        params_tensor (torch.Tensor): Tensor containing parameter values.
                                     Shape: (n_params,) for single vector or (batch_size, n_params) for batch
        search_space (dict): Dictionary defining the search space.
    Returns:
        dict or list: Dictionary for single vector, or list of dictionaries for batch.
    """
    if params_tensor.dim() == 1:
        # Single parameter vector case
        params_list = params_tensor.tolist()
        params_dict = {}
        for i, param in enumerate(search_space.keys()):
            params_dict[param] = params_list[i]
        return params_dict
    
    elif params_tensor.dim() == 2:
        # Batch of parameter vectors case
        batch_size, n_params = params_tensor.shape
        params_dict_list = []
        
        for batch_idx in range(batch_size):
            single_params = params_tensor[batch_idx]
            params_list = single_params.tolist()
            params_dict = {}
            for i, param in enumerate(search_space.keys()):
                params_dict[param] = params_list[i]
            params_dict_list.append(params_dict)
        
        return params_dict_list
    
    else:
        raise ValueError(f"tensor_to_param_dict expects 1D or 2D tensors, got {params_tensor.dim()}D tensor with shape {params_tensor.shape}")

def param_dict_to_tensor(params_dict:dict, search_space:dict) -> torch.Tensor:
    """
    Convert a dictionary of parameters to a tensor in the range [0, 1] based on the search space.
    Args:
        params_dict (dict): Dictionary containing parameter names and their corresponding values.
        search_space (dict): Dictionaries defining the search space.
    Returns:
        torch.Tensor: Tensor containing parameter values in the range [0, 1].
    """
    params_list = []
    for param in search_space.keys():
        params_list.append( params_dict[param] )
    return torch.tensor(params_list, dtype=torch.double)