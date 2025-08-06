from typing import Union
import torch

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