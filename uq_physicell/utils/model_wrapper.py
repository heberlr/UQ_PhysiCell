import pickle
from typing import Union

from uq_physicell import PhysiCell_Model
from .sumstats import recreate_qoi_functions, summary_function



def run_replicate(PhysiCellModel: PhysiCell_Model, sample_id: int, replicate_id: int,
                 ParametersXML: dict, ParametersRules: dict, qoi_functions:Union[dict, None]=None, qoi_def:dict={},
                 return_binary_output: bool = True, drop_columns: Union[list, None] = None, custom_summary_function:Union[callable, None]=None,
                 return_seed: bool = False, random_seed: Union[int, None] = None) -> tuple:
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
        qoi_def (dict): first-class object, that can be used in qoi_functions
            lambda string, mapped to their name.
            Defaults to {}.
        return_binary_output (bool, optional): Whether to return results as binary data. 
            Defaults to True.
        drop_columns (Union[list, None], optional): List of columns to drop from DataFrame. 
            Defaults to None.
        custom_summary_function (callable, optional): Custom summary function to use
            instead of the default generic QoI function.
            Defaults to None.
        return_seed (bool, optional): If True, append the random seed PhysiCell used
            for this replicate (read from random_seed.txt, or None if unavailable) as
            a 4th element of the returned tuple. Defaults to False.
        random_seed (int, optional): If given, forces this exact seed for the replicate
            (written to the XML). Defaults to None (reference-XML behaviour).

    Returns:
        tuple: A 3-tuple (sample_id, replicate_id, result_data), or a 4-tuple
            (sample_id, replicate_id, result_data, seed) when return_seed=True, where:
            - If qoi_functions provided: result_data contains calculated QoI values
            - If qoi_functions is None: result_data contains list of MCDS objects
    Note:
        If custom_summary_function is provided, qois_dic and drop_columns are not used.
    """
    if custom_summary_function:
        # Use the enhanced RunModel method that tracks processes
        result_data_nonserialized = PhysiCellModel.RunModel(
            sample_id, replicate_id, ParametersXML, ParametersRules,
            RemoveConfigFile=True, SummaryFunction=custom_summary_function, random_seed=random_seed)
    else:
        if qoi_functions:
            # Recreate QoI functions from their string representations
            recreated_qoi_funcs = recreate_qoi_functions(qoi_functions=qoi_functions, qoi_def=qoi_def)
        else:
            recreated_qoi_funcs = None

        # According qoi_functions run PhysiCellModel, if qoi_functions=NONE will return a list of MCDS objects
        result_data = PhysiCellModel.RunModel(
            sample_id, replicate_id, ParametersXML,
            ParametersRules=ParametersRules,
            SummaryFunction=lambda *args: summary_function(*args, qoi_functions=recreated_qoi_funcs, drop_columns=drop_columns, RemoveFolder=True),
            random_seed=random_seed,
        )
    
    seed = getattr(PhysiCellModel, "_last_random_seed", None)

    # Convert DataFrame into binary using pickle
    if return_binary_output:
        result = (sample_id, replicate_id, pickle.dumps(result_data))
    else:
        result = (sample_id, replicate_id, result_data)
    return result + (seed,) if return_seed else result

def run_replicate_serializable(PhysiCellModel_conf:dict, sample_id:int, replicate_id:int,
                               ParametersXML:dict, ParametersRules:dict, qoi_functions:Union[dict, None]=None, qoi_def:dict={},
                               return_binary_output:bool=True, drop_columns: Union[list, None] = None, custom_summary_function:Union[callable, None]=None,
                               return_seed: bool = False, random_seed: Union[int, None] = None) -> tuple:
    """
    Run a single replicate of the PhysiCell model and return the results. This wrapper function initializes the PhysiCell model and then calls the run_replicate function to execute the simulation. It is designed to be serializable for use in parallel processing contexts.
    
    Args:
        PhysiCellModel_conf (dict): Configuration dictionary for the PhysiCell model with ini_path and struc_name.
        sample_id (int): Sample ID.
        replicate_id (int): Replicate ID.
        ParametersXML (dict): Dictionary of XML parameters.
        ParametersRules (dict): Dictionary of rules parameters.
        return_binary_output (bool, optional): Whether to return results as binary data. 
            Defaults to True.
        qoi_functions (dict, optional): Dictionary of QoIs (keys as names, values as strings).
            Defaults to None.
        qoi_def (dict): first-class object, that can be used in qoi_functions
            lambda string, mapped to their name.
            Defaults to {}.
        drop_columns (list, optional): List of columns to drop from the output.
            Defaults to None.
        custom_summary_function (callable, optional): Custom summary function to use 
            instead of the default generic QoI function.
            Defaults to None.
    
        return_seed (bool, optional): If True, append the random seed PhysiCell used
            (or None if unavailable) as a 4th element of the returned tuple.
            Defaults to False.
        random_seed (int, optional): If given, forces this exact seed for the replicate.
            Defaults to None (reference-XML behaviour).

    Returns:
        tuple: A 3-tuple (sample_id, replicate_id, result_data), or a 4-tuple
            (sample_id, replicate_id, result_data, seed) when return_seed=True, where:
            - If qoi_functions provided: result_data contains calculated QoI values
            - If qoi_functions is None: result_data contains list of MCDS objects

    Note:
        If custom_summary_function is provided, qois_dic and drop_columns are not used.
    """
    try:
        # Initialize PhysiCell model with process tracking capabilities
        PhysiCellModel = PhysiCell_Model(PhysiCellModel_conf['ini_path'], PhysiCellModel_conf['struc_name'])
    except Exception as e:
        raise ValueError(f"Error initializing PhysiCell model: {e}")

    # Optional per-config IO subfolders (appended, matching the ABC/BO convention).
    # Used e.g. to isolate concurrent candidate models during ABC-SMC model selection.
    if PhysiCellModel_conf.get('input_folder'):
        PhysiCellModel.input_folder += PhysiCellModel_conf['input_folder']
    if PhysiCellModel_conf.get('output_folder'):
        PhysiCellModel.output_folder += PhysiCellModel_conf['output_folder']

    return run_replicate(
        PhysiCellModel=PhysiCellModel,
        sample_id=sample_id,
        replicate_id=replicate_id,
        ParametersXML=ParametersXML,
        ParametersRules=ParametersRules,
        qoi_functions=qoi_functions,
        qoi_def=qoi_def,
        return_binary_output=return_binary_output,
        drop_columns=drop_columns,
        custom_summary_function=custom_summary_function,
        return_seed=return_seed,
        random_seed=random_seed,
    )
