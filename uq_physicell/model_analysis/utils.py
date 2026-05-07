import numpy as np
import pandas as pd
from typing import Union

# My local modules
from ..database.ma_db import load_output, load_samples, check_db_consistency
from ..utils.sumstats import recreate_qoi_functions, safe_call_qoi_function

def _reshape_sa_expanded_data(expanded_data: pd.DataFrame, qoi_columns: list) -> pd.DataFrame:
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

def get_qoi_from_db_file(db_file:str, qoi_names:list) -> pd.DataFrame:
    """Extract quantities of interest (QoIs) from a database file containing simulation results.

    Args:
        db_file (str): Path to the SQLite database containing simulation results.
        qoi_names (list): List of QoI names to extract.

    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI.
    """
    # Load the full output data to extract time column and reshape
    df_qois_data = load_output(db_file, load_data=True)
 
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
    reshaped_data = _reshape_sa_expanded_data(expanded_data, qoi_names)
    
    return reshaped_data

def calculate_qoi_from_db_file(db_file:str, qoi_functions:dict, qoi_def:dict={}, chunk_size:int=10, mode='long', verbose=False) -> pd.DataFrame:
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
            sa, calib, and long. The default is long.

    Returns:
        pd.DataFrame: DataFrame with calculated QoI values indexed by SampleID and ReplicateID, with columns for each QoI.

    Example:
        >>> qoi_funcs = {
        ...     'live_cells': lambda df: len(df[df['dead'] == False]),
        ...     'dead_cells': lambda df: len(df[df['dead'] == True])
        ... }
        >>> qoi_df = calculate_qoi_from_db_file('study.db', qoi_funcs, chunk_size=20)
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

def get_mean_std_qois(df_qois: pd.DataFrame, filter_columns: list = []) -> pd.DataFrame:
    """Calculate the mean and standard deviation of quantities of interest (QoIs) across replicates.

    This function computes the mean and standard deviation QoI values for each parameter sample across all replicates, providing a central tendency and variability measure for the QoI estimates.
    
    Args: 
        df_qois (pd.DataFrame): DataFrame containing QoI values with SampleID, ReplicateID, and QoI columns.
        filter_columns (list, optional): List of columns to include in the calculation. If None, all numeric columns are used.
    Returns: tuple: A tuple containing:
        pd.DataFrame: DataFrame containing the mean QoI values for each SampleID, indexed by SampleID and ReplicateID.
        pd.DataFrame: DataFrame containing the standard deviation QoI values for each SampleID, indexed by SampleID and ReplicateID.
    Note:
        The mean QoI values are calculated by averaging the QoI estimates across all replicates for each parameter sample. This provides a central tendency measure for the QoI estimates, which can be used for sensitivity analysis and comparison between different parameter samples.
    """
    df_grouped = df_qois.groupby(['SampleID']+filter_columns)
    df_mean = df_grouped.mean(numeric_only=True) # ignores NaN values
    df_mean.drop(columns=['ReplicateID'], inplace=True)
    df_std = df_grouped.std(numeric_only=True) # ignores NaN values
    df_std.drop(columns=['ReplicateID'], inplace=True)
    return df_mean, df_std

def get_relative_mcse_qois(df_mean: pd.DataFrame, df_std: pd.DataFrame, num_replicates: int, time_columns: list) -> pd.DataFrame:
    """Calculate the relative Monte Carlo Standard Error (MCSE) for QoI estimates.

    This function computes the relative MCSE for each QoI across replicates, providing
    insight into the uncertainty of the QoI estimates due to finite sampling in the simulations.

    Args:
        df_mean (pd.DataFrame): DataFrame containing the mean QoI values for each SampleID, indexed by SampleID and ReplicateID.
        df_std (pd.DataFrame): DataFrame containing the standard deviation QoI values for each SampleID, indexed by SampleID and ReplicateID.
        num_replicates (int): The number of replicates used to calculate the mean and standard deviation of the QoIs.
        time_columns (list): List of column names corresponding to time points in the DataFrame, which should not be included in the MCSE calculation.
    Returns:
        pd.DataFrame: DataFrame containing the relative MCSE values for each QoI, indexed by SampleID and ReplicateID.

    Note:
        Relative Monte Carlo Standard Error (MCSE) is calculated as the standard deviation
        of the QoI across replicates divided by the square root of the number of replicates, and then normalized by the mean QoI value to express it as a percentage. This provides insight into the uncertainty of the QoI estimates due to finite sampling in the simulations.
            - < 1% (Excellent): This is the gold standard. If your relative MCSE is under 1%, your mean estimate is highly stable and precise. You can confidently use this metric for sensitivity analysis or publication.
            - 1% to 5% (Good / Acceptable): For stochastic biological simulations like PhysiCell, getting under 5% is generally considered reliable and practical.
            - 5% to 10% (Caution): You can use these metrics to observe broad trends, but small differences between parameters might just be noise. You likely need to run more replicates.
            - > 10% (Unreliable): The metric is too noisy. If you are running 50+ replicates and still have a relative MCSE > 10%, that specific QoI (Quantity of Interest) is likely a poor choice, or your biological system is fundamentally chaotic in that aspect.
    """
    df_relative_mcse = df_std/np.sqrt(num_replicates)
    # Small epsilon to avoid division by zero
    epsilon = max(0.01 * np.nanmedian(df_mean.abs().to_numpy().flatten()), 1e-12)
    # Relative MCSE
    df_relative_mcse = df_relative_mcse.div(df_mean + epsilon)
    # Replace the columns relative to time as the real value from df_mean
    df_relative_mcse[time_columns] = df_mean[time_columns]
    
    return df_relative_mcse

def get_summary_statistics_qois(df_qois: pd.DataFrame) -> tuple:
    """Calculate summary statistics (mean, standard deviation, and relative MCSE) of quantities of interest.

    Args:
        df_qois (pd.DataFrame): DataFrame containing QoI values with SampleID and ReplicateID columns.

    Returns: tuple: A tuple containing:
        pd.DataFrame: DataFrame with statistical summaries (mean) of QoIs grouped by SampleID, with columns for each QoI statistic.
        pd.DataFrame: DataFrame with standard deviation of QoIs grouped by SampleID, with columns for each QoI statistic.
        pd.DataFrame: DataFrame with relative Monte Carlo Standard Error (MCSE) of QoIs grouped by SampleID, with columns for each QoI statistic.
    """
    time_columns = sorted([col for col in df_qois.columns if col.startswith("time_")])
    if not time_columns:
        if 'time' in df_qois.columns:
            filter_columns = ['time']
        else:
            raise ValueError("No time columns found in the DataFrame.")
    else:
        filter_columns = []
    df_mean, df_std = get_mean_std_qois(df_qois, filter_columns=filter_columns)
    
    
    df_relative_mcse = get_relative_mcse_qois(df_mean, df_std, num_replicates = df_qois['ReplicateID'].nunique(), time_columns=time_columns)

    return df_mean, df_std, df_relative_mcse

def calculate_qoi_statistics(db_file_path: str, qoi_funcs: dict, df_qois_data: pd.DataFrame = None,  ignore_db_consistency: bool = False, qoi_def: dict = None, chunk_size: int = 10) -> tuple:
    """Calculate statistical summaries (mean and relative MCSE) of quantities of interest across replicates.

    This function computes mean and relative Monte Carlo Standard Error (MCSE) of QoI values across
    simulation replicates for each parameter sample, enabling uncertainty quantification.

    Args:
        db_file_path (str): Path to the database file for context.
        qoi_funcs (dict): Dictionary of QoI functions where keys are QoI names and
                         values are lambda functions or None.
        df_qois_data (pd.DataFrame): DataFrame containing QoI values with SampleID,
                                   ReplicateID, and QoI columns. Default is None, in which case the function will attempt to load QoI data from the database.
        ignore_db_consistency (bool): If True, bypasses the database consistency check.
        qoi_def (dict): first-class object, that can be used in qoi_funcs lambda string, mapped to their name.
                        e.g. for a function definition, if the function definition is:
                        def my_func():
                            print('hello world!')
                            return 0
                        then the qoi_def dict would look like this:
                        {'my_func': my_func}
        chunk_size (int, optional): Number of samples to process at a time when loading from the database. Default is 10. Adjust based on available memory and data size.
    Returns:
        tuple: A tuple containing:
            - df_mean (pd.DataFrame): DataFrame with statistical summaries (mean) of QoIs grouped by SampleID, with columns for each QoI statistic.
            - df_std (pd.DataFrame): DataFrame with standard deviation of QoIs grouped by SampleID, with columns for each QoI statistic.
            - df_relative_mcse (pd.DataFrame): DataFrame with relative Monte Carlo Standard Error (MCSE) of QoIs grouped by SampleID, with columns for each QoI statistic.
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
        >>> df_mean, df_std, df_mcse = calculate_qoi_statistics(qoi_data, qoi_funcs, 'study.db')
    """

    if df_qois_data is None:
        print("No QoI data provided, calculating QoIs from the database...")
        try:
            df_qois_data = load_output(db_file_path, load_data=False)
        except Exception as e:
            raise ValueError(f"Error loading output data from database: {e}")
        
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
            print("Extracting QoIs from DataFrame...")
            try:
                df_qois = get_qoi_from_db_file(db_file_path, qoi_columns)
            except Exception as e:
                raise ValueError(f"Error extracting QoIs from DataFrame: {e}")
        # Check if 'Data' column in df_qois_data is a series of mcds list - Case of db generated by generic summary function with NO QoI functions
        elif isinstance(df_qois_data['Data'].iloc[0], list):
            print("Calculating QoIs from mcds list...")
            try:
                df_qois = calculate_qoi_from_db_file(db_file_path, qoi_funcs, qoi_def=qoi_def, chunk_size=chunk_size)
            except Exception as e:
                raise ValueError(f"Error calculating QoIs from mcds list: {e}")
            if df_qois.empty:
                raise ValueError("df_qois is empty, unable to generate QoIs from the database.")
        else:
            raise ValueError("Error: Data element is neither Dataframe nor List.")
    # If QoIs are already in the database and 'Data' column is not present
    else:
        print("Extracting QoIs from existing DataFrame...")
        df_qois = df_qois_data

    # Take the mean and MCSE among the replicates and sort the samples
    try:
        df_mean, df_std, df_relative_mcse = get_summary_statistics_qois(df_qois)
    except Exception as e:
        raise ValueError(f"Error taking the mean, std, and MCSE among replicates: {e}")
    return df_mean, df_std, df_relative_mcse

def apply_pca_to_qois(df_mean: pd.DataFrame, latent_dim: int = 3, seed: int = None):
    """Apply PCA to the selected QoIs.

    This function uses PCA (scikit-learn) as a linear dimensionality reduction technique on the QoI matrix (samples x features).

    Args:
        df_mean (pd.DataFrame): DataFrame with QoI mean values indexed by SampleID.
        latent_dim (int): Dimension of the latent encoding.
        seed (int): Random seed for reproducibility (used if PCA requires it). If None, no seed is set.
    Returns:
        dict: {
            'method': 'pca',
            'encoder_output': np.ndarray (n_samples x latent_dim),
            'reconstruction': np.ndarray (n_samples x n_features),
            'model': PCA object,
            'scaler': StandardScaler object
        }
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_mean.to_numpy(dtype=float))
    n_samples, n_features = X_scaled.shape

    pca = PCA(n_components=min(latent_dim, n_features), random_state=seed)
    enc = pca.fit_transform(X_scaled)
    recon = pca.inverse_transform(enc)
    return {'method': 'pca', 'encoder_output': enc, 'reconstruction': recon, 'model': pca, 'scaler': scaler}

def apply_autoencoder_to_qois(df_mean: pd.DataFrame, latent_dim: int = 2, epochs: int = 200, batch_size: int = 32, verbose: bool = False, seed: int = None):
    """Apply an autoencoder to the selected QoIs.

    This function attempts to use PyTorch to train a small autoencoder on the
    QoI matrix (samples x features). If a seed is provided, results will be reproducible.

    Args:
        df_mean (pd.DataFrame): DataFrame with QoI mean values indexed by SampleID.
        latent_dim (int): Dimension of the latent encoding.
        epochs (int): Training epochs for the PyTorch autoencoder.
        batch_size (int): Batch size for training.
        verbose (bool): Verbosity flag.
        seed (int): Random seed for reproducibility. If None, no seed is set.

    Returns:
        dict: {
            'method': 'torch',
            'encoder_output': np.ndarray (n_samples x latent_dim),
            'reconstruction': np.ndarray (n_samples x n_features),
            'model': trained model object
            'scaler': StandardScaler object
        }
    """
    import numpy as _np
    import random
    from sklearn.preprocessing import StandardScaler

    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        _np.random.seed(seed)

    X = df_mean.sort_index().to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_samples, n_features = X_scaled.shape

    # Try PyTorch first
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        # Set torch seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if verbose:
            print("Using PyTorch autoencoder")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class AE(nn.Module):
            def __init__(self, n_in, n_latent):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(n_in, max(16, n_in//2)),
                    nn.ReLU(),
                    nn.Linear(max(16, n_in//2), n_latent),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(n_latent, max(16, n_in//2)),
                    nn.ReLU(),
                    nn.Linear(max(16, n_in//2), n_in),
                )

            def forward(self, x):
                z = self.encoder(x)
                xrec = self.decoder(z)
                return xrec, z

        model = AE(n_features, latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(tensor_X)
        loader = DataLoader(dataset, batch_size=min(batch_size, n_samples), shuffle=False, generator=torch.Generator().manual_seed(seed) if seed is not None else None)

        model.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                xrec, _ = model(batch)
                loss = loss_fn(xrec, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu().item()) * batch.size(0)
            if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
                print(f"AE epoch {ep+1}/{epochs} loss={epoch_loss / n_samples:.6g}")

        # encode and reconstruct
        model.eval()
        with torch.no_grad():
            X_t = tensor_X.to(device)
            xrec_t, z_t = model(X_t)
            recon = xrec_t.cpu().numpy()
            enc = z_t.cpu().numpy()

        return {'method': 'torch', 'encoder_output': enc, 'reconstruction': recon, 'model': model, 'scaler': scaler}

    except Exception as e:
        if verbose:
            print(f"PyTorch autoencoder unavailable or failed ({e}).")


def regression_accuracy_parameters(df_parameters: pd.DataFrame, encoder_output: np.ndarray) -> pd.DataFrame:
    """Calculate regression accuracy of parameters from the latent encoding.

    This function trains a regression model (e.g., Random Forest) to predict each parameter from the latent encoding and evaluates the R² score for each parameter.

    Args:
        df_parameters (pd.DataFrame): DataFrame containing parameter values indexed by SampleID.
        encoder_output (np.ndarray): Latent encoding of the QoI data (n_samples x latent_dim).
    Returns:
        pd.DataFrame: DataFrame containing R² scores for each parameter, indexed by parameter name.
            - R² ≈ 1.0: Perfect fit. The model explains all the variance in the parameter values from the latent encoding.
            - R² < 0.5: Poor fit. The model does not capture the relationship between the latent encoding and the parameter values well.
            - R² ≈ 0: No fit. The model does not explain any of the variance in the parameter values from the latent encoding, indicating no relationship.
            - R² < 0: Worse than no fit. The model performs worse than simply predicting the mean parameter value for all samples, suggesting a very poor relationship between the latent encoding and the parameter values.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    r2_results = {}
    for param in df_parameters.columns:
        y = df_parameters[param].to_numpy()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(encoder_output, y)
        # Use cross_val_score to get the 'True' R2
        # This prevents the 0.82 from being an overfitted illusion
        scores = cross_val_score(model, encoder_output, y, cv=5, scoring='r2')
        r2_results[param] = scores.mean()

    return pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2_CV'])


def align_params_to_qois(df_params: pd.DataFrame, df_qois: pd.DataFrame) -> pd.DataFrame:
    """Align parameter DataFrame to match the multi-index structure of QoI DataFrame.
    
    If df_qois has a multi-index (SampleID, time) but df_params only has SampleID,
    this function replicates parameter rows for each time point so that both DataFrames
    can be joined on (SampleID, time).
    Args:
        df_params (pd.DataFrame): Parameter DataFrame indexed by SampleID.
        df_qois (pd.DataFrame): QoI DataFrame, possibly with multi-index (SampleID, time).
    Returns:
        pd.DataFrame: Parameter DataFrame aligned to match df_qois index structure.
    """
    # If df_qois has a multi-index with 'time', expand df_params accordingly
    if hasattr(df_qois.index, 'names') and 'time' in df_qois.index.names:
        # Extract unique (SampleID, time) pairs from df_qois
        qois_index = df_qois.index
        
        # Create aligned params by replicating rows for each time point
        expanded_data = []
        new_index_tuples = []
        
        for sample_id, time_val in qois_index:
            if sample_id in df_params.index:
                # Replicate the parameter row for this (SampleID, time) pair
                expanded_data.append(df_params.loc[sample_id].values)
                new_index_tuples.append((sample_id, time_val))
        
        # Create new DataFrame with aligned rows
        df_params_aligned = pd.DataFrame(expanded_data, columns=df_params.columns)
        df_params_aligned.index = pd.MultiIndex.from_tuples(new_index_tuples, names=['SampleID', 'time'])
        
        return df_params_aligned.sort_index()
    else:
        # No time index in df_qois, return df_params as-is
        return df_params

def find_optimal_qoi_set(df_qois: pd.DataFrame, df_params: pd.DataFrame) -> list:
    """
    Use Recursive Feature Elimination with Cross-Validation (RFECV) to find the optimal set of QoIs that best predict the parameters.
    This function trains a regression model (e.g., Random Forest) to predict parameters from the QoIs and uses RFECV to identify the most important QoIs for accurate parameter prediction. The optimal set of QoIs is determined based on the R² score obtained through cross-validation.
    Args:
        df_qois (pd.DataFrame): DataFrame containing QoI mean values indexed by SampleID and time.
        df_params (pd.DataFrame): DataFrame containing parameter values indexed by SampleID and time.
    Returns:
        list: List of optimal QoI names that best predict the parameters based on RFECV analysis.
    """

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold
    from sklearn.metrics import make_scorer, r2_score

    # 1. Define the base model
    base_estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    # 2. Define a custom scoring function that focuses on the worst-predicted parameter (the bottleneck)
    def min_r2_scorer(y_true, y_pred):
        # Calculate R2 for each parameter individually
        scores = r2_score(y_true, y_pred, multioutput='raw_values')
        # Return the worst score (the bottleneck)
        return np.min(scores)

    # 3. Set up RFECV with the custom scorer and cross-validation strategy
    selector = RFECV(
        estimator=base_estimator,
        step=1,
        cv=KFold(5),
        scoring=make_scorer(min_r2_scorer), # Focus on the hardest-to-identify parameter
        min_features_to_select=df_params.shape[1],
        n_jobs=-1
    )

    # 4. Fit the selector to the data
    selector.fit(df_qois.sort_index().to_numpy(), df_params.sort_index().to_numpy())
    
    return selector

def _regression_accuracy_with_weights(df_parameters: pd.DataFrame, encoder_output: np.ndarray, mcse_weights: np.ndarray = None) -> pd.DataFrame:
    """
    Calculate regression accuracy of parameters from latent encoding with optional MCSE-based weighting.
    
    Uses sample_weight inversely proportional to MCSE to give more weight to stable features.
    
    Args:
        df_parameters (pd.DataFrame): DataFrame containing parameter values indexed by SampleID.
        encoder_output (np.ndarray): Latent encoding (n_samples x latent_dim).
        mcse_weights (np.ndarray): Per-sample weights based on information-to-noise ratio. If None, uniform weights.
    
    Returns:
        pd.DataFrame: DataFrame containing weighted R² scores for each parameter.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    r2_results = {}
    for param in df_parameters.columns:
        y = df_parameters[param].to_numpy()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        if mcse_weights is not None:
            # Use sample_weight to favor stable features (passed via fit_params)
            model.fit(encoder_output, y, sample_weight=mcse_weights)
            scores = cross_val_score(model, encoder_output, y, cv=5, scoring='r2', fit_params={'sample_weight': mcse_weights})
        else:
            model.fit(encoder_output, y)
            scores = cross_val_score(model, encoder_output, y, cv=5, scoring='r2')
        
        r2_results[param] = scores.mean()

    return pd.DataFrame.from_dict(r2_results, orient='index', columns=['R2_CV'])

def recursive_feature_elimination(df_qois_mean, df_qois_mcse, df_params, autoencoder_params={}, mcse_threshold=0.10, correlation_threshold=0.95, verbose=False):
    """
    Perform smart feature elimination using balanced Autoencoder-first approach.
    
    **Philosophy (Hybrid):** Removes only pathological features, preserves useful noise:
    1. Remove EXTREME outliers (MCSE > 10% by default, configurable)
    2. Remove highly correlated redundancy (>95%, configurable)  
    3. Apply Autoencoder on cleaned feature set
    4. Use RFECV on latent space to find optimal encoding
    5. Validate with Synthetic Recovery Test
    
    This avoids "noise = non-informative" while still cleaning truly problematic features.
    
    Args:
        df_qois_mean (pd.DataFrame): DataFrame containing mean QoI values indexed by SampleID and time.
        df_qois_mcse (pd.DataFrame): DataFrame containing relative MCSE values for QoIs indexed by SampleID and time.
        df_params (pd.DataFrame): DataFrame containing parameter values indexed by SampleID.
        autoencoder_params (dict): Parameters for autoencoder (latent_dim, epochs, batch_size, seed).
        mcse_threshold (float): Threshold for removing EXTREME noise (default 0.10 = 10%). Only removes pathological features.
        correlation_threshold (float): Threshold for removing redundant correlations (default 0.95 = 95%). Only removes near-duplicates.
        verbose (bool): Print detailed pipeline stages.
    
    Returns:
        dict: {
            'removed_extreme_noise': list of features removed as outliers,
            'removed_redundant': list of highly correlated features removed,
            'cleaned_qois': list of QoIs kept for autoencoder,
            'correlation_matrix': full pairwise correlation matrix,
            'full_autoencoder_results': autoencoder results on cleaned set,
            'full_regression_r2': R² from full set,
            'final_qois': QoI names selected by RFECV,
            'reduced_autoencoder_results': autoencoder on final selected QoIs,
            'reduced_regression_r2': R² from reduced set,
            'synthetic_recovery_test': Full vs Reduced R² comparison,
        }
    """
    import warnings
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, r2_score

    #####################################################
    # Stage 0: Prepare Data
    #####################################################
    df_qois_work = df_qois_mean.copy()
    
    # Identify and fill trajectory QoIs (those with NaN in MCSE)
    trajectory_qois = [col for col in df_qois_mcse.columns if df_qois_mcse[col].isna().any()]
    if trajectory_qois:
        last_per_sample = df_qois_work[trajectory_qois].groupby(level='SampleID').transform('last')
        df_qois_work[trajectory_qois] = df_qois_work[trajectory_qois].fillna(last_per_sample)
        if verbose:
            print(f"✓ Trajectory QoIs filled (NaN→last value per sample): {trajectory_qois}")
    
    # Align parameters
    df_params_aligned = align_params_to_qois(df_params, df_qois_work)

    #####################################################
    # Stage 1: Remove EXTREME Outliers Only (Light Filter)
    #####################################################
    if verbose:
        print(f"\n[STAGE 1] Pathological Noise Removal (MCSE > {mcse_threshold*100}%)")
    
    removed_extreme_noise = [col for col in df_qois_mcse.columns if df_qois_mcse[col].quantile(0.9) > mcse_threshold]
    df_qois_work = df_qois_work.drop(columns=removed_extreme_noise, errors='ignore')
    
    if verbose:
        if removed_extreme_noise:
            print(f"  Removed {len(removed_extreme_noise)} pathological features: {removed_extreme_noise}")
        else:
            print(f"  No features exceed extreme noise threshold (all MCSE ≤ {mcse_threshold*100}%)")

    #####################################################
    # Stage 2: Remove Redundant Correlations (Hard Threshold)
    #####################################################
    if verbose:
        print(f"\n[STAGE 2] Redundancy Removal (Correlation > {correlation_threshold*100}%)")
    
    corr_matrix = df_qois_work.corr().abs()
    removed_redundant = []
    
    # Greedy strategy: for each correlated pair, keep the one with lower median MCSE
    visited = set()
    for col in corr_matrix.columns:
        if col in visited:
            continue
        corr_peers = corr_matrix.index[corr_matrix[col] > correlation_threshold].tolist()
        if len(corr_peers) > 1:
            # Keep the one with best MCSE
            best_feature = min(corr_peers, key=lambda x: df_qois_mcse[x].median())
            for peer in corr_peers:
                if peer != best_feature and peer not in removed_redundant:
                    removed_redundant.append(peer)
                    visited.add(peer)
            visited.add(best_feature)
    
    df_qois_work = df_qois_work.drop(columns=removed_redundant, errors='ignore')
    
    if verbose:
        if removed_redundant:
            print(f"  Removed {len(removed_redundant)} redundant features: {removed_redundant}")
        else:
            print(f"  No near-duplicate features found (correlation ≤ {correlation_threshold*100}%)")

    #####################################################
    # Stage 3: Autoencoder on CLEANED Feature Set
    #####################################################
    if verbose:
        print(f"\n[STAGE 3] Autoencoder on Cleaned Features ({len(df_qois_work.columns)} QoIs)")
    
    dict_ae_full = apply_autoencoder_to_qois(
        df_qois_work,
        latent_dim=autoencoder_params.get('latent_dim', 2),
        epochs=autoencoder_params.get('epochs', 200),
        batch_size=autoencoder_params.get('batch_size', 32),
        seed=autoencoder_params.get('seed', 42),
        verbose=verbose
    )
    
    # Regression on full autoencoder output
    df_r2_full = _regression_accuracy_with_weights(
        df_params_aligned, 
        dict_ae_full['encoder_output'], 
        mcse_weights=None  # Uniform weights for full set
    )
    
    if verbose:
        print(f"  Full AE → Params R² (mean): {df_r2_full['R2_CV'].mean():.4f}")

    #####################################################
    # Stage 4: RFECV on Latent Space to Find Optimal QoI Set
    #####################################################
    if verbose:
        print(f"\n[STAGE 4] RFECV-based Feature Selection on Latent Encoding")
    
    # Define scoring: bottleneck (minimum R² of hardest parameter)
    def min_r2_scorer(y_true, y_pred):
        if y_true.ndim == 1:
            return r2_score(y_true, y_pred)
        else:
            return np.min([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    
    # RFECV just to identify if we can remove more from the cleaned set
    base_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    selector = RFECV(
        base_rf,
        step=1,
        cv=KFold(5),
        scoring=make_scorer(min_r2_scorer),
        min_features_to_select=max(1, len(df_qois_work.columns) // 2),
        n_jobs=-1
    )
    
    # Fit RFECV on cleaned QoI data
    selector.fit(df_qois_work.to_numpy(), df_params_aligned.to_numpy())
    
    selected_qoi_mask = selector.support_
    final_qois = df_qois_work.columns[selected_qoi_mask].tolist()
    
    if verbose:
        removed_by_rfecv = df_qois_work.columns[~selected_qoi_mask].tolist()
        print(f"  RFECV selected {len(final_qois)}/{len(df_qois_work.columns)} QoIs")
        if removed_by_rfecv:
            print(f"  Removed by RFECV: {removed_by_rfecv}")

    #####################################################
    # Stage 5: Synthetic Recovery Test
    #####################################################
    if verbose:
        print(f"\n[STAGE 5] Synthetic Recovery Test (Full vs Reduced)")
    
    # Train on reduced set
    dict_ae_reduced = apply_autoencoder_to_qois(
        df_qois_work[final_qois],
        latent_dim=autoencoder_params.get('latent_dim', 2),
        epochs=autoencoder_params.get('epochs', 200),
        batch_size=autoencoder_params.get('batch_size', 32),
        seed=autoencoder_params.get('seed', 42),
        verbose=False
    )
    
    df_r2_reduced = _regression_accuracy_with_weights(
        df_params_aligned,
        dict_ae_reduced['encoder_output'],
        mcse_weights=None
    )
    
    # Compare
    recovery_test = pd.DataFrame({
        'Full_R2': df_r2_full['R2_CV'],
        'Reduced_R2': df_r2_reduced['R2_CV'],
        'R2_Drop': df_r2_full['R2_CV'] - df_r2_reduced['R2_CV']
    })
    
    if verbose:
        print(f"  Full Set Mean R²: {df_r2_full['R2_CV'].mean():.4f}")
        print(f"  Reduced Set Mean R²: {df_r2_reduced['R2_CV'].mean():.4f}")
        print(f"  Mean R² Drop: {recovery_test['R2_Drop'].mean():.4f}")
        
        acceptable_drops = recovery_test[recovery_test['R2_Drop'] <= 0.05]
        critical_drops = recovery_test[recovery_test['R2_Drop'] > 0.1]
        
        if len(acceptable_drops) == len(recovery_test):
            print(f"  ✓ All parameters recovered well (R² drop ≤ 0.05)")
        elif len(critical_drops) > 0:
            print(f"  ⚠ Critical R² drops (>0.1) for: {critical_drops.index.tolist()}")

    #####################################################
    # Stage 6: Final Output
    #####################################################
    
    return {
        'removed_extreme_noise': removed_extreme_noise,
        'removed_redundant': removed_redundant,
        'cleaned_qois': list(df_qois_work.columns),
        'correlation_matrix': corr_matrix,
        # Full set (on cleaned features)
        'full_autoencoder_results': dict_ae_full,
        'full_regression_r2': df_r2_full,
        # Reduced set (post-RFECV)
        'final_qois': final_qois,
        'reduced_autoencoder_results': dict_ae_reduced,
        'reduced_regression_r2': df_r2_reduced,
        # Validation
        'synthetic_recovery_test': recovery_test,
    }

    
    