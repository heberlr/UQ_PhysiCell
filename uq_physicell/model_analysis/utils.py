import numpy as np
import pandas as pd
from typing import Union

# My local modules
from ..database.ma_db import load_structure
from ..utils.model_wrapper import create_named_function_from_string

def reshape_sa_expanded_data(expanded_data: pd.DataFrame, qoi_columns: list) -> pd.DataFrame:
    """Reshape expanded sensitivity analysis data for pivot table analysis.
    
    This function transforms time-series data from long format (multiple rows per sample)
    to wide format (columns for each time point) to facilitate statistical analysis.
    
    Args:
        expanded_data (pd.DataFrame): DataFrame containing expanded simulation data
            with SampleID, ReplicateID, time, and QoI columns.
        qoi_columns (list): List of quantity of interest column names to reshape.
    
    Returns:
        pd.DataFrame: Reshaped DataFrame with multi-level columns where each QoI
                     and time point becomes a separate column indexed by SampleID
                     and ReplicateID.
    
    Example:
        >>> data = pd.DataFrame({
        ...     'SampleID': [0, 0, 1, 1],
        ...     'ReplicateID': [0, 0, 0, 0],
        ...     'time': [0, 1, 0, 1],
        ...     'cell_count': [100, 150, 80, 120]
        ... })
        >>> reshaped = reshape_sa_expanded_data(data, ['cell_count'])
    """
    try:
        # Ensure QoI columns are numeric
        for qoi in qoi_columns:
            expanded_data[qoi] = pd.to_numeric(expanded_data[qoi], errors='coerce')

        # Create a unique time_id for each time step
        expanded_data['time_id'] = expanded_data.groupby(['SampleID', 'ReplicateID']).cumcount()

        # Pivot the DataFrame to create columns for each QoI and time_id
        reshaped_data = expanded_data.pivot_table(
            index=['SampleID', 'ReplicateID'],
            columns='time_id',
            values=qoi_columns + ['time']
        )

        # Flatten the multi-index columns
        reshaped_data.columns = [
            f"{col[0]}_{int(col[1])}" if col[0] != 'time' else f"time_{int(col[1])}"
            for col in reshaped_data.columns
        ]
        reshaped_data.reset_index(inplace=True)

        return reshaped_data
    except Exception as e:
        raise ValueError(f"Error reshaping expanded data: {e}")

def calculate_qoi_from_sa_db(db_file: str, qoi_functions: str) -> pd.DataFrame:
    """Calculate quantities of interest from sensitivity analysis database results.
    
    This function loads simulation results from a database and applies QoI functions
    to extract meaningful metrics from the time-series data.
    
    Args:
        db_file (str): Path to the SQLite database containing simulation results.
        qoi_functions (str): Dictionary of QoI functions where keys are QoI names
                           and values are lambda functions or string representations.
    
    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID
                     and ReplicateID, with columns for each QoI.
    
    Example:
        >>> qoi_funcs = {
        ...     'final_cells': 'lambda data: data[-1]["cell_count"]',
        ...     'max_growth': 'lambda data: max(d["cell_count"] for d in data)'
        ... }
        >>> qoi_df = calculate_qoi_from_sa_db('study.db', qoi_funcs)
    """
    # Load the database structure
    _, _, _, _, df_output = load_structure(db_file)
    # Recreate QoI functions from their string representations
    recreated_qoi_funcs = {
        qoi_name: create_named_function_from_string(qoi_value, qoi_name)
        for qoi_name, qoi_value in qoi_functions.items()
    }
    df_qois = pd.DataFrame()
    for SampleID in df_output['SampleID'].unique():
        df_sample = df_output[df_output['SampleID'] == SampleID]
        for ReplicateID in df_sample['ReplicateID'].unique():
            mcds_ts_list = df_sample[df_sample['ReplicateID'] == ReplicateID]['Data'].values[0]
            # print(f"SampleID: {SampleID}, ReplicateID: {ReplicateID} - mcds_ts_list: {mcds_ts_list}")
            data = {'SampleID': SampleID, 'ReplicateID': ReplicateID}
            for id_time, mcds in enumerate(mcds_ts_list):
                data[f"time_{id_time}"] = mcds.get_time()
                # cell dataframe
                df_cell = mcds.get_cell_df()
                try: 
                    for qoi_name, qoi_func in recreated_qoi_funcs.items():
                        # Store the QoI value in the data dictionary
                        data[f"{qoi_name}_{id_time}"] =  qoi_func(df_cell)
                except Exception as e:
                    raise RuntimeError(f"Error calculating QoIs for SampleID: {SampleID}, ReplicateID: {ReplicateID} - QoI: {qoi_name}_{id_time}: {e}")
            # Store the data in a DataFrame
            df_qoi_replicate = pd.DataFrame({key: [value] for key, value in data.items()})
            df_qois = pd.concat([df_qois, df_qoi_replicate], ignore_index=True)
    df_qois = df_qois.reset_index(drop=True)
    return df_qois
 
def calculate_qoi_statistics(df_qois_data: pd.DataFrame, qoi_funcs: dict, db_file_path: str) -> pd.DataFrame:
    """Calculate statistical summaries of quantities of interest across replicates.
    
    This function computes mean and standard deviation of QoI values across
    simulation replicates for each parameter sample, enabling uncertainty quantification.
    
    Args:
        df_qois_data (pd.DataFrame): DataFrame containing QoI values with SampleID,
                                   ReplicateID, and QoI columns.
        qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names and
                         values are lambda functions or None.
        db_file_path (str): Path to the database file for context.
    
    Returns:
        pd.DataFrame: DataFrame with statistical summaries (mean, std) of QoIs
                     grouped by SampleID, with columns for each QoI statistic.
    
    Raises:
        ValueError: If no QoI functions are defined or data format is invalid.
    
    Example:
        >>> qoi_funcs = {'cell_count': lambda x: x.sum(), 'growth_rate': None}
        >>> stats_df = calculate_qoi_statistics(qoi_data, qoi_funcs, 'study.db')
        >>> print(stats_df[['cell_count_mean', 'cell_count_std']])
    """
    
    # Check if 'Data' column in df_qois_data is a DataFrame - Case of db generated by custom summary function
    if ('Data' in df_qois_data.columns):
        qoi_columns = list(qoi_funcs.keys())
        if not qoi_columns:
            raise ValueError("Error: No QoI functions defined.")
        if isinstance(df_qois_data['Data'].iloc[0], pd.DataFrame):
            print("Calculating QoIs from DataFrame...")
            try:
                # Extract the consistent 'time' column from the first DataFrame
                time_column = df_qois_data['Data'].iloc[0]['time'].values
                # Flatten the 'Data' column into a single DataFrame with SampleID and ReplicateID
                expanded_data = pd.concat(
                    [
                        pd.DataFrame(data).assign(SampleID=SampleID, ReplicateID=ReplicateID)
                        for (SampleID, ReplicateID), group in df_qois_data.groupby(['SampleID', 'ReplicateID'])
                        for data in group['Data']  # Ensure 'Data' contains DataFrames
                    ],
                    ignore_index=True
                )
                # Dynamically calculate the number of repetitions for the time column
                num_repeats = len(expanded_data) // len(time_column)
                if len(expanded_data) % len(time_column) != 0:
                    raise ValueError("Mismatch between expanded_data rows and time column length.")
                expanded_data['time'] = np.tile(time_column, num_repeats)
                # Reshape the expanded_data to match the expected format
                reshaped_data = reshape_sa_expanded_data(expanded_data, qoi_columns)
                # Assign the reshaped data to df_qois
                df_qois = reshaped_data
            except Exception as e:
                raise ValueError(f"Error calculating QoIs from DataFrame: {e}")
        # Check if 'Data' column in df_qois_data is a series of mcds list - Case of db generated by generic summary function with NO QoI functions
        elif ('Data' in df_qois_data.columns) and isinstance(df_qois_data['Data'].iloc[0], list):
            print("Calculating QoIs from mcds list...")
            try:
                df_qois = calculate_qoi_from_sa_db(db_file_path, qoi_funcs)
            except Exception as e:
                raise ValueError(f"Error calculating QoIs from mcds list: {e}")
            if df_qois.empty:
                raise ValueError("df_qois is empty, unable to generate QoIs from the database.")
        else:
            raise ValueError("Error: Data element is neither Dataframe nor List.")
    # If QoIs are already in the database and 'Data' column is not present
    else:
        print("Calculating QoIs from existing DataFrame...")
        df_qois = df_qois_data

    # Take the mean and MCSE among the replicates and sort the samples
    try:
        # Number of Replicates is equal for all samples
        num_replicates = df_qois['ReplicateID'].nunique()
        time_columns = sorted([col for col in df_qois.columns if col.startswith("time_")])
        print(f"Number of replicates: {num_replicates}")
        df_grouped = df_qois.groupby(['SampleID'])
        df_stds = df_grouped.std(numeric_only=True) # ignores NaN values
        df_stds.drop(columns=['ReplicateID'], inplace=True)
        df_mean = df_grouped.mean(numeric_only=True) # ignores NaN values
        df_mean.drop(columns=['ReplicateID'], inplace=True)
        # Calculate the relative Monte Carlo Standard Error (MCSE)
        df_relative_mcse = df_stds/np.sqrt(num_replicates)
        # Small epsilon to avoid division by zero
        epsilon = 0.01 * np.median(df_mean.abs().to_numpy().flatten())
        # Relative MCSE
        df_relative_mcse = df_relative_mcse.div(df_mean + epsilon)
        # Replace the columns relative to time as the real value from df_mean
        df_relative_mcse[time_columns] = df_mean[time_columns]
    except Exception as e:
        raise ValueError(f"Error taking the mean and MCSE among replicates: {e}")
    return df_mean, df_relative_mcse