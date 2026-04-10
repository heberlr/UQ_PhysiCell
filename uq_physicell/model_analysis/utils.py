import numpy as np
import pandas as pd
from typing import Union

# My local modules
from ..database.ma_db import load_output, load_samples, check_db_consistency
from ..utils.sumstats import recreate_qoi_functions, safe_call_qoi_function

def reshape_sa_expanded_data(expanded_data: pd.DataFrame, qoi_columns: list) -> pd.DataFrame:
    """Reshape expanded sensitivity analysis data for pivot table analysis.

    This function transforms time-series data from long format (multiple rows per sample)
    to wide format (columns for each time point) to facilitate statistical analysis.

    Args:
        expanded_data (pd.DataFrame): DataFrame containing expanded simulation data
            with SampleID, ReplicateID, time, and QoI columns.
        qoi_columns (list): List of quantity of interest column names to reshape.

    Returns:
        pd.DataFrame: Reshaped DataFrame with multi-level columns where each QoI and time point becomes a separate column indexed by SampleID and ReplicateID.

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

def mcds_list_to_qoi_df_for_sa(recreated_qoi_funcs, all_sample_ids, chunk_size, db_file, verbose=False) -> pd.DataFrame:
    """Convert a list of MCDS objects to a DataFrame of quantities of interest for sensitivity analysis.

    This function processes a list of MCDS simulation results, extracting relevant
    quantities of interest (QoIs) at each time point and organizing them into a
    structured DataFrame suitable for sensitivity analysis.

    Args:
        recreated_qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names
                                   and values are callable functions.
        all_sample_ids (list): List of all sample IDs to process.
        chunk_size (int): Number of samples to process in each chunk to manage memory usage.
        db_file (str): Path to the database file containing simulation output.
    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI - columns combined with time points.
    """
    df_qois = pd.DataFrame()
    # Process samples in chunks to avoid memory issues
    for i in range(0, len(all_sample_ids), chunk_size):
        chunk_sample_ids = all_sample_ids[i:i + chunk_size]
        # Load only this chunk of data
        df_output = load_output(db_file, sample_ids=chunk_sample_ids, load_data=True)
        for SampleID in df_output['SampleID'].unique():
            df_sample = df_output[df_output['SampleID'] == SampleID]
            for ReplicateID in df_sample['ReplicateID'].unique():
                mcds_ts_list = df_sample[df_sample['ReplicateID'] == ReplicateID]['Data'].values[0]
                if verbose:
                    print(f"SampleID: {SampleID}, ReplicateID: {ReplicateID} - mcds_ts_list: {mcds_ts_list}")
                data = {'SampleID': SampleID, 'ReplicateID': ReplicateID}
                for id_time, mcds in enumerate(mcds_ts_list):
                    data[f"time_{id_time}"] = mcds.get_time()
                    try:
                        for qoi_name, qoi_func in recreated_qoi_funcs.items():
                            function_result = safe_call_qoi_function(qoi_func, mcds=mcds, list_mcds=mcds_ts_list)
                            if function_result is not None:
                                data[f"{qoi_name}_{id_time}"] = function_result
                        if len(data) == 2: continue  # Skip if no QoI data was computed
                    except Exception as e:
                        raise RuntimeError(f"Error calculating QoIs for SampleID: {SampleID}, ReplicateID: {ReplicateID} - QoI: {qoi_name}_{id_time}: {e}")
                # Store the data in a DataFrame
                df_qoi_replicate = pd.DataFrame({key: [value] for key, value in data.items()})
                df_qois = pd.concat([df_qois, df_qoi_replicate], ignore_index=True)
        # Explicitely free memory
        del mcds_ts_list
        del df_sample
        del df_output
    # Reset index of the final DataFrame
    df_qois = df_qois.reset_index(drop=True)
    return df_qois

def mcds_list_to_qoi_df_long(recreated_qoi_funcs, all_sample_ids, chunk_size, db_file, verbose=False) -> pd.DataFrame:
    """Convert a list of MCDS objects to a DataFrame  of quantities of interest in long format.

    This function processes a list of MCDS simulation results, extracting relevant
    quantities of interest (QoIs) at each time point and organizing them into a long
    structured DataFrame.

    Args:
        recreated_qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names
                                   and values are callable functions.
        all_sample_ids (list): List of all sample IDs to process.
        chunk_size (int): Number of samples to process in each chunk to manage memory usage.
        db_file (str): Path to the database file containing simulation output.
    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI - columns combined with time points.
    """
    # Process samples in chunks to avoid memory issues
    b_column = True
    ls_column = ['SampleID', 'time', 'ReplicateID']
    llo_data = []
    for i in range(0, len(all_sample_ids), chunk_size):
        chunk_sample_ids = all_sample_ids[i:i + chunk_size]
        # Load only this chunk of data
        df_output = load_output(db_file, sample_ids=chunk_sample_ids, load_data=True)
        # Fetch mcds from sample replicate
        for s_sample in sorted(df_output['SampleID'].unique()):
            df_sample = df_output[df_output['SampleID'] == s_sample]
            for s_replicate in sorted(df_sample['ReplicateID'].unique()):
                l_mcds = df_sample[df_sample['ReplicateID'] == s_replicate]['Data'].values[0]
                for mcds in l_mcds:
                    lo_data = [s_sample, mcds.get_time(), s_replicate]
                    try:
                        for s_qoi_name, o_qoi_func in sorted(recreated_qoi_funcs.items()):
                            if verbose:
                                print(f"processing sample replicate qoi: {s_sample} {s_replicate} {s_qoi_name} ...")
                            # Execut qoi function on mcds
                            o_result = safe_call_qoi_function(o_qoi_func, mcds=mcds, list_mcds=l_mcds)
                            # Save results from multi qoi function
                            if type(o_result) in {dict, pd.Series}:
                                for s_key, o_value in sorted(o_result.items()):
                                    if b_column:
                                        ls_column.append(f'{s_key}_{s_qoi_name}')
                                    lo_data.append(o_value)
                            # Save result from single qoi function
                            else:
                                if b_column:
                                    ls_column.append(s_qoi_name)
                                lo_data.append(o_result)
                    # Error handling
                    except Exception as e:
                        raise RuntimeError(f"Error calculating QoIs for SampleID: {s_sample}, ReplicateID: {s_replicate} - QoI: {s_qoi_name}: {e}")
                    # Save row
                    llo_data.append(lo_data)
                    # Update flag
                    b_column = False
        # Explicitely free memory
        del l_mcds
        del df_sample
        del df_output
    # Generate data frame
    df_qois = pd.DataFrame(llo_data, columns=ls_column)
    df_qois = df_qois.sort_values(['SampleID','time','ReplicateID'], ignore_index=True)
    return df_qois

def mcds_list_to_qoi_df_for_calib(recreated_qoi_funcs, all_sample_ids, chunk_size, db_file, verbose=False) -> pd.DataFrame:
    """Convert a list of MCDS objects to a DataFrame of quantities of interest for calibration.

    This function processes a list of MCDS simulation results, extracting relevant
    quantities of interest (QoIs) and organizing them into a structured DataFrame
    suitable for calibration tasks.

    Args:
        recreated_qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names
                                   and values are callable functions.
        all_sample_ids (list): List of all sample IDs to process.
        chunk_size (int): Number of samples to process in each chunk to manage memory usage.
        db_file (str): Path to the database file containing simulation output.
    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI - columns is not combined with time points.
    """
    df_qois = pd.DataFrame()
    # Process samples in chunks to avoid memory issues
    for i in range(0, len(all_sample_ids), chunk_size):
        chunk_sample_ids = all_sample_ids[i:i + chunk_size]
        # Load only this chunk of data
        df_output = load_output(db_file, sample_ids=chunk_sample_ids, load_data=True)
        for SampleID in df_output['SampleID'].unique():
            df_sample = df_output[df_output['SampleID'] == SampleID]
            for ReplicateID in df_sample['ReplicateID'].unique():
                mcds_ts_list = df_sample[df_sample['ReplicateID'] == ReplicateID]['Data'].values[0]
                for id_time, mcds in enumerate(mcds_ts_list):
                    data = {'SampleID': SampleID, 'ReplicateID': ReplicateID, 'time': mcds.get_time()}
                    try:
                        for qoi_name, qoi_func in recreated_qoi_funcs.items():
                            function_result = safe_call_qoi_function(qoi_func, mcds=mcds, list_mcds=mcds_ts_list)
                            if verbose:
                                print(f"processing sample replicate qoi: {SampleID} {ReplicateID} {qoi_name} ...")
                            if function_result is not None:
                                data[qoi_name] = function_result
                        if len(data) == 3: continue  # Skip if no QoI data was computed
                    except Exception as e:
                        raise RuntimeError(f"Error calculating QoIs for SampleID: {SampleID}, ReplicateID: {ReplicateID} - QoI: {qoi_name}: {e}")
                    # Store the data in a DataFrame
                    df_qoi_replicate = pd.DataFrame({key: [value] for key, value in data.items()})
                    df_qois = pd.concat([df_qois, df_qoi_replicate], ignore_index=True)
        # Explicitely free memory
        del mcds_ts_list
        del df_sample
        del df_output
    # Reset index of the final DataFrame
    df_qois = df_qois.reset_index(drop=True)
    return df_qois

def calculate_qoi_from_sa_db(db_file:str, qoi_functions:dict, qoi_def:dict={}, chunk_size:int=10, mode='sa', verbose=False) -> pd.DataFrame:
    """Calculate quantities of interest from sensitivity analysis database results.

    This function loads simulation results from a database in chunks and applies QoI
    functions to extract meaningful metrics from the time-series data. Processing in
    chunks helps avoid excessive memory usage for large databases.

    Args:
        db_file (str): Path to the SQLite database containing simulation results.
        qoi_functions (dict): Dictionary of QoI functions where keys are QoI names
                           and values are lambda functions or string representations.
        qoi_def (dict): first-class object, that can be used in qoi_functions
                        lambda string, mapped to their name.
                        e.g. for a function definition, if the function definition is:
                        def my_func():
                            print('hello world!')
                            return 0
                        then the qoi_def dict would look like this:
                        {'my_func': my_func}
        chunk_size (int, optional): Number of samples to process at a time. Default is 10.
                                   Adjust based on available memory and data size.
        mode:  Specify the form of the result dataframe. Possible modes are
            sa, calib, and long. The default is sa.

    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI.

    Example:
        >>> qoi_funcs = {
        ...     'live_cells': lambda df: len(df[df['dead'] == False]),
        ...     'dead_cells': lambda df: len(df[df['dead'] == True])
        ... }
        >>> qoi_df = calculate_qoi_from_sa_db('study.db', qoi_funcs, chunk_size=20)
    """

    # Load sample IDs to determine what to process
    dic_samples = load_samples(db_file)
    all_sample_ids = sorted(dic_samples.keys())

    # Recreate QoI functions from their string representations
    recreated_qoi_funcs = recreate_qoi_functions(qoi_functions=qoi_functions, qoi_def=qoi_def)
    if mode == 'sa':
        df_qois = mcds_list_to_qoi_df_for_sa(
            recreated_qoi_funcs=recreated_qoi_funcs,
            all_sample_ids=all_sample_ids,
            chunk_size=chunk_size,
            db_file=db_file,
            verbose=verbose,
        )
    elif mode == 'calib':
        df_qois = mcds_list_to_qoi_df_for_calib(
            recreated_qoi_funcs=recreated_qoi_funcs,
            all_sample_ids=all_sample_ids,
            chunk_size=chunk_size,
            db_file=db_file,
            verbose=verbose,
        )
    elif mode == 'long':
        df_qois = mcds_list_to_qoi_df_long(
            recreated_qoi_funcs=recreated_qoi_funcs,
            all_sample_ids=all_sample_ids,
            chunk_size=chunk_size,
            db_file=db_file,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported modes are 'sa' and 'calib'.")

    return df_qois



def calculate_qoi_statistics(df_qois_data: pd.DataFrame, qoi_funcs: dict, db_file_path: str, ignore_db_consistency: bool = False) -> pd.DataFrame:
    """Calculate statistical summaries (mean and relative MCSE) of quantities of interest across replicates.

    This function computes mean and relative Monte Carlo Standard Error (MCSE) of QoI values across
    simulation replicates for each parameter sample, enabling uncertainty quantification.

    Args:
        df_qois_data (pd.DataFrame): DataFrame containing QoI values with SampleID,
                                   ReplicateID, and QoI columns.
        qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names and
                         values are lambda functions or None.
        db_file_path (str): Path to the database file for context.
        ignore_db_consistency (bool): If True, bypasses the database consistency check.
    Returns:
        tuple: A tuple containing:
            - df_mean (pd.DataFrame): DataFrame with statistical summaries (mean) of QoIs grouped by SampleID, with columns for each QoI statistic.
            - df_mcse (pd.DataFrame): DataFrame with relative Monte Carlo Standard Error (MCSE) of QoIs grouped by SampleID, with columns for each QoI statistic.
    Note:
        Relative Monte Carlo Standard Error (MCSE) is calculated as the standard deviation
        of the QoI across replicates divided by the square root of the number of replicates, and then normalized by the mean QoI value to express it as a percentage. This provides insight into the uncertainty of the QoI estimates due to finite sampling in the simulations.
            - < 1% (Excellent): This is the gold standard. If your relative MCSE is under 1%, your mean estimate is highly stable and precise. You can confidently use this metric for sensitivity analysis or publication.
            - 1% to 5% (Good / Acceptable): For stochastic biological simulations like PhysiCell, getting under 5% is generally considered reliable and practical.
            - 5% to 10% (Caution): You can use these metrics to observe broad trends, but small differences between parameters might just be noise. You likely need to run more replicates.
            - > 10% (Unreliable): The metric is too noisy. If you are running 50+ replicates and still have a relative MCSE > 10%, that specific QoI (Quantity of Interest) is likely a poor choice, or your biological system is fundamentally chaotic in that aspect.

    Raises:
        ValueError: If no QoI functions are defined or data format is invalid.

    Example:
        >>> qoi_funcs = {
        ...     'live_cells': lambda df: len(df[df['dead'] == False]),
        ...     'dead_cells': lambda df: len(df[df['dead'] == True])
        ... }
        >>> df_mean, df_mcse = calculate_qoi_statistics(qoi_data, qoi_funcs, 'study.db')
    """
    # Check db consistency
    if not check_db_consistency(db_file_path):
        if ignore_db_consistency:
            print("Warning: Database consistency check failed, but proceeding as per user request.")
        else:
            raise ValueError("Database consistency check failed. Please verify the database integrity.")

    # Check if 'Data' column in df_qois_data is a DataFrame - Case of db generated by custom summary function
    if ('Data' in df_qois_data.columns):
        qoi_columns = list(qoi_funcs.keys())
        if not qoi_columns:
            raise ValueError("Error: No QoI functions defined.")
        if isinstance(df_qois_data['Data'].iloc[0], pd.DataFrame):
            print("Calculating QoIs from DataFrame...")
            # Load the full output data to extract time column and reshape
            df_qois_data = load_output(db_file_path, load_data=True)
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
        elif isinstance(df_qois_data['Data'].iloc[0], list):
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
        epsilon = max(0.01 * np.nanmedian(df_mean.abs().to_numpy().flatten()), 1e-12)
        # Relative MCSE
        df_relative_mcse = df_relative_mcse.div(df_mean + epsilon)
        # Replace the columns relative to time as the real value from df_mean
        df_relative_mcse[time_columns] = df_mean[time_columns]
    except Exception as e:
        raise ValueError(f"Error taking the mean and MCSE among replicates: {e}")
    return df_mean, df_relative_mcse
