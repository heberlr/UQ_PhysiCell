import pandas as pd
import numpy as np
import pickle
from typing import Union

from uq_physicell import PhysiCell_Model
from .sumstats import generic_QoI


# Helper function to create named functions from strings
def create_named_function_from_string(func_str:str, qoi_name:str) -> callable:
    """
    Dynamically creates a named function from a string and assigns it to the module's global scope.
    Parameters:
    - func_str: The string representation of the function.
    - qoi_name: The name of the function to be created.
    Return:
    - The created function.
    """
    # Check if the function already exists in the global scope
    func_name = f"named_{qoi_name}"
    if func_name not in globals():
        exec(
            f"def {func_name}(*args, **kwargs):\n"
            f"    return eval({repr(func_str)}, {{'len': len, 'pd': pd, 'np': np}})(*args, **kwargs)",
            globals()
        )
    return globals()[func_name]

def summary_function(outputPath:str, summaryFile:str, dic_params:dict, SampleID:int, ReplicateID:int, qoi_functions:dict, drop_columns:Union[list, None]=None):
    """
    A standalone function to encapsulate the summary function logic.
    Parameters:
    - outputPath: Path to the output folder
    - summaryFile: Path to the summary file
    - dic_params: Dictionary of parameters
    - SampleID
    - ReplicateID
    - qoi_functions: Dictionary of QoI functions (keys as names, values as lambda functions or strings)
    - drop_columns: List of columns to drop from the DataFrame (optional)
    Return:
    - The result of the generic_QoI function.
    """
    try:
        return generic_QoI(
            outputPath=outputPath,
            summaryFile=summaryFile,
            dic_params=dic_params,
            SampleID=SampleID,
            ReplicateID=ReplicateID,
            qoi_funcs=qoi_functions,
            mode='time_series',
            RemoveFolder=True,
            drop_columns=drop_columns
        )
    except Exception as e:
        raise RuntimeError(f"Error in summary function: {e}")

def run_replicate(PhysiCellModel:PhysiCell_Model, sample_id:int, replicate_id:int, ParametersXML:dict, ParametersRules:dict, qoi_functions:dict, return_binary_output:bool=True, drop_columns:Union[list, None]=None) -> tuple:
    """
    Run a single replicate of the simulation and return the results.
    Parameters:
    - PhysiCellModel: The PhysiCell model instance.
    - sample_id: The sample ID.
    - replicate_id: The replicate ID.
    - ParametersXML: The parameters for the XML.
    - ParametersRules: The parameters for the rules.
    - qoi_functions: The QoI functions (strings) or None.
    - results_binary: If True, return results as binary (default is True).
    - drop_columns: List of columns to drop from the DataFrame (optional).
    Return:
    - sampleID, replicateID, result_data
        - if qoi_functions: result_data = QoIs of the simulation.
        - if qoi_functions==None: result_data = list of mcds from pcdl.
    """
    # Check if qoi_functions is not None
    if qoi_functions:
        # Recreate QoI functions from their string representations
        recreated_qoi_funcs = {
            qoi_name: create_named_function_from_string(qoi_value, qoi_name)
            for qoi_name, qoi_value in qoi_functions.items()
        }
    else: 
        recreated_qoi_funcs = None

    # Pass the updated qoi_functions to the PhysiCellModel
    result_data_nonserialized = PhysiCellModel.RunModel(
        sample_id, replicate_id, ParametersXML,
        ParametersRules=ParametersRules,
        SummaryFunction=lambda *args: summary_function(*args, qoi_functions=recreated_qoi_funcs, drop_columns=drop_columns),
    )
    # print(f"Simulation completed for SampleID: {sample_id}, ReplicateID: {replicate_id}\n Result.head(): {result_data.head()}")
    
    # Serialize the DataFrame using pickle
    if return_binary_output: result_data = pickle.dumps(result_data_nonserialized)
    else: result_data = result_data_nonserialized
        
    return sample_id, replicate_id, result_data

def run_replicate_serializable(ini_path:str, struc_name:str, sampleID:int, replicateID:int, ParametersXML:dict, ParametersRules:dict, return_binary_output:bool=True, qois_dic:Union[dict, None]=None, drop_columns:list=[], custom_summary_function:Union[callable, None]=None) -> tuple:
    """
    Run a single replicate of the PhysiCell model and return the results.
    Parameters:
    - ini_path: Path to the initialization file.
    - struc_name: Structure name.
    - sampleID: Sample ID.
    - replicateID: Replicate ID.
    - ParametersXML: Dictionary of XML parameters.
    - ParametersRules: Dictionary of rules parameters.
    - qois_dic: Dictionary of QoIs (keys as names, values as strings).
    - drop_columns: List of columns to drop from the output.
    - custom_summary_function: Custom summary function to use instead of the default generic QoI function. `qois_dic` and `drop_columns` are not used if `custom_summary_function` is provided.
    Returns:
    - sampleID, replicateID, result_data
    """
    try:
        PhysiCellModel = PhysiCell_Model(ini_path, struc_name)
        if custom_summary_function:
            result_data_nonserialized = PhysiCellModel.RunModel(
                sampleID, replicateID, ParametersXML, ParametersRules, RemoveConfigFile=True, SummaryFunction=custom_summary_function)
            if return_binary_output: result_data = pickle.dumps(result_data_nonserialized)
            else: result_data = result_data_nonserialized
        else:
            _, _, result_data = run_replicate(PhysiCellModel, sampleID, replicateID, ParametersXML, ParametersRules, qois_dic, return_binary_output, drop_columns)
        return sampleID, replicateID, result_data
    except Exception as e:
        raise RuntimeError(f"Error running replicate: {e}")