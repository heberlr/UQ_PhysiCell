import pandas as pd
import numpy as np
import pickle
import re
from typing import Union

from uq_physicell import PhysiCell_Model
from .sumstats import generic_QoI

def compute_persistent_homology(df:pd.DataFrame, Plot=False) -> pd.Series:
    """
    Compute persistent homology vectorization using muspan.
    (source: https://docs.muspan.co.uk/latest/_collections/topology/Topology%203%20-%20persistence%20vectorisation.html)
    
    Parameters:
    - df_cells: DataFrame -> DataFrame containing cell data with 'position_x', 'position_y', and 'cell_type' columns.
    
    Returns:
    - pd.Series -> Vectorized persistent homology features.
    """
    try:
        import muspan
    except ImportError:
        raise ImportError("muspan library is required for computing persistent homology. Please install it via 'pip install muspan'.")

    # Extract cell positions
    points = df[['position_x', 'position_y']].to_numpy()
    labels = df['cell_type'].to_numpy()
    
    # Create a muspan domain and add points
    domain = muspan.domain('Position Data')
    domain.add_points(points, 'Cell Positions')
    domain.add_labels('Celltype', labels)

    # Query to select cells of types 'A', 'B', ... in each domain
    q_cell_types = muspan.query.query(domain, ('label', 'Celltype'), 'in', df['cell_type'].unique().tolist())

    # Plot domain with cell types (optional)
    if Plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        muspan.visualise.visualise(domain, 'Celltype', ax=ax, add_cbar=False, marker_size=2.5, objects_to_plot=q_cell_types)

    # Compute Vietoris-Rips filtrations
    feature_persistence = muspan.topology.vietoris_rips_filtration(domain, population=q_cell_types, max_dimension=1)

    # Plot persistence diagram (optional)
    if Plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        muspan.visualise.persistence_diagram(feature_persistence, ax=ax)
        plt.show()

    # Vectorise the persistence homology diagram for the domain using statistical method
    vectorised_ph,name_of_features = muspan.topology.vectorise_persistence(feature_persistence, method='statistics')

    return pd.Series(vectorised_ph, index=name_of_features)

# Helper function to create named functions from strings
def create_named_function_from_string(func_str: str, qoi_name: str) -> callable:
    """
    Dynamically creates a named function from a string.
    Parameters:
    - func_str: The string representation of the function.
    - qoi_name: The name of the function to be created.
    Return:
    - The created function with preserved parameter inspection capability.
    """
    # Extract parameter name from lambda (e.g., "lambda df_subs:" -> "df_subs")
    param_match = re.search(r'lambda\s+(\w+):', func_str)
    if param_match:
        param_name = param_match.group(1)
    else:
        # Default parameter name if parsing fails
        param_name = 'df'
    
    # Create a namespace with necessary imports
    namespace = {'pd': pd, 'np': np, 'len': len}
    
    # Create a wrapper function that accepts any arguments
    def wrapper(*args, **kwargs):
        # Evaluate the lambda function string
        func = eval(func_str, namespace)
        # Call it with the provided arguments
        return func(*args, **kwargs)
    
    # Store the original parameter name as an attribute for inspection
    wrapper.__param_name__ = param_name
    wrapper.__name__ = f"named_{qoi_name}"
    
    return wrapper

def summary_function(outputPath:str, summaryFile:str, dic_params:dict, SampleID:int, ReplicateID:int, qoi_functions:dict, drop_columns:Union[list, None]=None, mode:str='time_series'):
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
            mode=mode,
            RemoveFolder=True,
            drop_columns=drop_columns
        )
    except Exception as e:
        raise RuntimeError(f"Error in summary function: {e}")

def run_replicate(PhysiCellModel: PhysiCell_Model, sample_id: int, replicate_id: int, 
                 ParametersXML: dict, ParametersRules: dict, qoi_functions:Union[dict, None]=None, 
                 return_binary_output: bool = True, drop_columns: Union[list, None] = None, custom_summary_function:Union[callable, None]=None) -> tuple:
    """Run a single replicate of the PhysiCell simulation.
    
    This function executes one simulation replicate with specified parameters and
    returns either processed QoI results or raw simulation data.
    
    Args:
        PhysiCellModel (PhysiCell_Model): The PhysiCell model instance to run.
        sample_id (int): Unique identifier for the parameter sample.
        replicate_id (int): Identifier for the simulation replicate.
        ParametersXML (dict): Parameters to modify in the XML configuration.
        ParametersRules (dict): Parameters for custom rules modifications.
        qoi_functions (dict): Dictionary of QoI functions (keys as names, values as strings).
                             If None, returns raw simulation data.
        return_binary_output (bool, optional): Whether to return results as binary data. 
                                             Defaults to True.
        drop_columns (Union[list, None], optional): List of columns to drop from DataFrame. 
                                                   Defaults to None.
        custom_summary_function (callable, optional): Custom summary function to use 
                                                   instead of the default generic QoI function.
                                                   Defaults to None.
    
    Returns:
        tuple: A 3-tuple containing (sample_id, replicate_id, result_data) where:
            - If qoi_functions provided: result_data contains calculated QoI values
            - If qoi_functions is None: result_data contains list of MCDS objects
    """
    if custom_summary_function:
        # Use the enhanced RunModel method that tracks processes
        result_data_nonserialized = PhysiCellModel.RunModel(
            sample_id, replicate_id, ParametersXML, ParametersRules,
            RemoveConfigFile=True, SummaryFunction=custom_summary_function)
    else:
        if qoi_functions:
            # Recreate QoI functions from their string representations
            recreated_qoi_funcs = {
                qoi_name: create_named_function_from_string(qoi_value, qoi_name)
                for qoi_name, qoi_value in qoi_functions.items()
            }

        # According qoi_functions run PhysiCellModel, if qoi_functions=NONE will return a list of MCDS objects
        result_data = PhysiCellModel.RunModel(
            sample_id, replicate_id, ParametersXML,
            ParametersRules=ParametersRules,
            SummaryFunction=lambda *args: summary_function(*args, qoi_functions=recreated_qoi_funcs, drop_columns=drop_columns),
        )
    
    # Convert DataFrame into binary using pickle
    if return_binary_output:
        return sample_id, replicate_id, pickle.dumps(result_data)
    else:
        return sample_id, replicate_id, result_data

def run_replicate_serializable(PhysiCellModel_conf:dict, sampleID:int, replicateID:int, 
                               ParametersXML:dict, ParametersRules:dict, qoi_functions:Union[dict, None]=None, 
                               return_binary_output:bool=True, drop_columns: Union[list, None] = None, custom_summary_function:Union[callable, None]=None) -> tuple:
    """
    Run a single replicate of the PhysiCell model and return the results.
    
    Parameters:
        PhysiCellModel_conf (dict): Configuration dictionary for the PhysiCell model with ini_path and struc_name.
        sampleID (int): Sample ID.
        replicateID (int): Replicate ID.
        ParametersXML (dict): Dictionary of XML parameters.
        ParametersRules (dict): Dictionary of rules parameters.
        return_binary_output (bool, optional): Whether to return results as binary data. 
                                             Defaults to True.
        qoi_functions (dict, optional): Dictionary of QoIs (keys as names, values as strings).
                                 Defaults to None.
        drop_columns (list, optional): List of columns to drop from the output.
                                     Defaults to [].
        custom_summary_function (callable, optional): Custom summary function to use 
                                                   instead of the default generic QoI function.
                                                   Defaults to None.
    
    Returns:
        tuple: (sampleID, replicateID, result_data)
    
    Note:
        If custom_summary_function is provided, qois_dic and drop_columns are not used.
    """
    try:
        # Initialize PhysiCell model with process tracking capabilities
        PhysiCellModel = PhysiCell_Model(PhysiCellModel_conf['ini_path'], PhysiCellModel_conf['struc_name'])
    except Exception as e:
        raise ValueError(f"Error initializing PhysiCell model: {e}")
            
    return run_replicate(PhysiCellModel, sampleID, replicateID, 
                         ParametersXML, ParametersRules, 
                         qoi_functions, return_binary_output, 
                         drop_columns, custom_summary_function)