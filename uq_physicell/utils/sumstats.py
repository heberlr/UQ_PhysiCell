import pcdl
import pandas as pd
from shutil import rmtree
from typing import Union
import inspect

def check_functions_need_microenv(qoi_funcs):
    """
    Check if any of the QoI functions require microenvironment data (df_subs).
    
    Parameters:
    - qoi_funcs: dict -> Dictionary of QoI functions
    
    Returns:
    - bool: True if any function needs microenvironment data
    """
    # If no functions provided, return True (default to loading microenvironment)
    if not qoi_funcs:
        return True
    
    # If any function needs microenvironment data, return True
    for func in qoi_funcs.values():
        # Check if any parameter name indicates substrate/concentration data
        param_lower = func.__param_name__.lower()
        if 'subs' in param_lower or 'conc' in param_lower or 'micro' in param_lower:
                return True
    # If none need it,
    return False

def safe_call_qoi_function(func, df_cell, df_subs=None):
    """
    Safely call a QoI function with the appropriate dataframe based on parameter inspection.
    
    Parameters:
    - func: The QoI function to call
    - df_cell: Cell dataframe
    - df_subs: Substrate dataframe (optional)
    
    Returns:
    - Result of the QoI function
    """
    
    # Check if function has our custom parameter name attribute (from string creation)
    param_name = func.__param_name__
    needs_subs = any(keyword in param_name.lower() for keyword in ['subs', 'conc', 'micro'])
    
    # Call function with appropriate dataframe
    if needs_subs:
        if df_subs is None:
            raise ValueError(f"Function {func.__name__} needs substrate data but df_subs is None")
        return func(df_subs)
    else:
        return func(df_cell)

def summ_func_FinalPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Final population of live and dead cells
    
    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    # read the last file
    mcds = pcdl.TimeStep('final.xml',outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    # dataframe of cells
    df_cell = mcds.get_cell_df() 
    # population stats live and dead cells
    live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
    dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
    # dataframe structure
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

# Population over time of live and dead cells
def summ_func_TimeSeriesPopLiveDead(outputPath:str,summaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int) -> Union[pd.DataFrame,None]:
    """
    Population over time of live and dead cells

    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    
    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """

    mcds_ts = pcdl.TimeSeries(outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    for mcds in mcds_ts.get_mcds_list():
        df_cell = mcds.get_cell_df() 
        live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
        dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
        data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
        data_conc = {**data,**dic_params} # concatenate output data and parameters
        if ( mcds.get_time() == 0 ): df = pd.DataFrame([data_conc]) # create the dataframe
        else: df.loc[len(df)] = data_conc # append the dictionary to the dataframe
    # remove replicate output folder
    rmtree( outputPath )
    if (summaryFile): 
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df


# Generic function for creating custom QoIs (Quantities of Interest)
def generic_QoI(outputPath: str, summaryFile: Union[str, None], dic_params: dict, SampleID: int, ReplicateID: int, qoi_funcs: dict, mode: str = 'time_series', RemoveFolder: bool = True, drop_columns:list = []) -> Union[pd.DataFrame, None]:
    """
    Generic function for creating custom QoIs (Quantities of Interest) based on df_cell elements.

    Parameters:
    - outputPath: str -> Path to the PhysiCell output directory.
    - summaryFile: Union[str, None] -> File to store the summary (optional).
    - dic_params: dict -> Dictionary of simulation parameters.
    - SampleID: int -> Unique identifier for the sample.
    - ReplicateID: int -> Unique identifier for the replicate.
    - qoi_funcs: dict -> Dictionary of QoI functions with keys as QoI names and values as functions/lambdas.
    - mode: str -> Mode of operation: 'last_snapshot', 'time_series', or 'summary'.
    - RemoveFolder: bool -> Whether to remove the output folder after processing.
    - drop_columns: list -> List of columns to drop from the DataFrame.

    Returns:
    - pd.DataFrame or None -> DataFrame with the computed QoIs or None if saved to a file.
    """
    try:
        if mode == 'last_snapshot':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_funcs)
            
            # Load the last snapshot
            mcds = pcdl.TimeStep('final.xml', outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            if qoi_funcs is None:
                # Optional: Remove replicate output folder
                if (RemoveFolder): rmtree(outputPath)
                # Entire mcds is returned if drop_columns is empty
                if not drop_columns:
                    return [mcds]
                else:
                    df_cell = mcds.get_cell_df()  # Ensure df_cell is initialized
                    df_cell.drop(columns=drop_columns, inplace=True, errors='ignore')
                return [df_cell]
                    
            else:
                df_cell = mcds.get_cell_df()
                df_subs = mcds.get_conc_df() if needs_microenv else None
                # Compute QoIs using safe function calling
                qoi_data = {name: safe_call_qoi_function(func, df_cell, df_subs) for name, func in qoi_funcs.items()}
                data = {
                    'time': mcds.get_time(),
                    'sampleID': SampleID,
                    'replicateID': ReplicateID,
                    **qoi_data
                }
                data_conc = {**data, **dic_params}
                df = pd.DataFrame([data_conc])

        elif mode == 'time_series':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_funcs)
            
            # Load the time series
            mcds_ts = pcdl.TimeSeries(outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            #  All data is stored as a list of mcds
            if qoi_funcs is None:
                # Optional: Remove replicate output folder
                if (RemoveFolder): rmtree(outputPath)
                # Entire list of mcds is returned if drop_columns is empty
                if not drop_columns:
                    return mcds_ts.get_mcds_list()
                else:
                    df_list = []
                    for mcds in mcds_ts.get_mcds_list():
                        df_cell = mcds.get_cell_df()
                        df_cell.drop(columns=drop_columns, inplace=True, errors='ignore')
                        df_list.append(df_cell)
                    return df_list
            else:
                df_list = []
                for mcds in mcds_ts.get_mcds_list():
                    df_cell = mcds.get_cell_df()
                    df_subs = mcds.get_conc_df() if needs_microenv else None
                    try: 
                        qoi_data = {name: safe_call_qoi_function(func, df_cell, df_subs) for name, func in qoi_funcs.items()}
                    except Exception as e:
                        raise RuntimeError(f"Error computing QoIs in generic_QoI function: {e}")

                    data = {
                        'time': mcds.get_time(),
                        'sampleID': SampleID,
                        'replicateID': ReplicateID,
                        **qoi_data
                    }
                    data_conc = {**data, **dic_params}
                    df_list.append(data_conc)
                df = pd.DataFrame(df_list)
                

        elif mode == 'summary':
            # Check if any function needs microenvironment data
            needs_microenv = check_functions_need_microenv(qoi_funcs)
            
            # Load the time series and summarize across snapshots
            mcds_ts = pcdl.TimeSeries(outputPath, microenv=needs_microenv, graph=False, settingxml=None, verbose=False)
            summary_data = {name: [] for name in qoi_funcs.keys()}
            for mcds in mcds_ts.get_mcds_list():
                df_cell = mcds.get_cell_df()
                df_subs = mcds.get_conc_df() if needs_microenv else None
                for name, func in qoi_funcs.items():
                    summary_data[name].append(safe_call_qoi_function(func, df_cell, df_subs))
            summarized_qois = {name: sum(values) / len(values) for name, values in summary_data.items()}
            data = {
                'sampleID': SampleID,
                'replicateID': ReplicateID,
                **summarized_qois
            }
            data_conc = {**data, **dic_params}
            df = pd.DataFrame([data_conc])

        else:
            raise ValueError(f"Invalid mode: {mode}. Supported modes are 'last_snapshot', 'time_series', and 'summary'.")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error: Required file not found in {outputPath}. {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing QoIs: {e}")

    # Optional: Remove replicate output folder
    if (RemoveFolder): rmtree(outputPath)

    # Save to file or return DataFrame
    if summaryFile:
        df.to_csv(summaryFile, sep='\t', encoding='utf-8')
        return None
    else:
        return df
