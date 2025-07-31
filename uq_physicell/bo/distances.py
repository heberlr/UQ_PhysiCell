import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

def test_volume(ref_point: torch.Tensor, train_obj_true: torch.Tensor, scale_factor: float, iteration: int, logger=None) -> float:
    """
    Test hypervolume and provide error checking with helpful messages.
    
    Args:
        ref_point (torch.Tensor): Reference point for hypervolume calculation
        train_obj_true (torch.Tensor): True objective values tensor
        scale_factor (float): Current scale factor used in distance-to-fitness transformation
        iteration (int, optional): Current iteration number (0 for initial samples)
        logger (optional): Logger instance for error messages
    Returns:
        float: Hypervolume value
        
    Raises:
        ValueError: If hypervolume is zero or negative, with detailed diagnostic information
    """
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_true)
    volume = bd.compute_hypervolume().item()
    
    # Check for zero hypervolume and provide helpful error message
    if volume <= 0.0:
        iteration_msg = f"at iteration {iteration}"
        
        if logger:
            logger.error(f"ZERO HYPERVOLUME DETECTED {iteration_msg}!")
            logger.error(f"Current scale factor: {scale_factor}")
            logger.error(f"Current fitness range: {train_obj_true.min(dim=0)[0]} to {train_obj_true.max(dim=0)[0]}")
        
        raise ValueError(
            f"Hypervolume is zero ({volume:.6f}) {iteration_msg}, indicating all fitness values are at or below the reference point.\n"
            f"This suggests the scale factor ({scale_factor}) is too small, causing fitness values to be too close to 0.\n"
            f"Solutions:\n"
            f"  1. Increase the scale_factor in bo_options (currently {scale_factor})\n"
            f"  2. Check if distance values are unusually large\n"
            f"  3. Consider using a different distance function or QoI weights\n"
            f"  4. Verify that your observed data and model outputs are in compatible units"
        )
    
    return volume

def Euclidean(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the Euclidean distance (L2 norm) between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The Euclidean distance between the model data and observational data.
    """
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.sqrt(np.sum(diff ** 2))

def Manhattan(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the Manhattan distance (L1 norm) between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The Manhattan distance between the model data and observational data.
    """
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.sum(np.abs(diff))

def Chebyshev(dic_model_data:dict, dic_obs_data:dict)-> float:
    """
    Compute the Chebyshev distance (L∞ norm) between simulation outputs and observational data.
    Args:
        dic_model_data (dict): Dictionary containing model data with keys "time" and "value".
        dic_obs_data (dict): Dictionary containing observational data with keys "time" and "value".
    Returns:
        float: The Chebyshev distance between the model data and observational data.
    """
    indices_model = np.where(np.isin(dic_model_data["time"], dic_obs_data["time"]))[0]
    indices_obsData = np.where(np.isin(dic_obs_data["time"], dic_model_data["time"]))[0]
    if len(indices_model) == 0 or len(indices_obsData) == 0:
        raise ValueError("No matching time points found between model data and observational data.")
    diff = dic_model_data["value"][indices_model] - dic_obs_data["value"][indices_obsData]
    return np.max(np.abs(diff))