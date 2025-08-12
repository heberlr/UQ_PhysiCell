import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .database import load_structure

def normalize_params(df_params, df_search_space) -> pd.DataFrame:
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
    df_merged['ParamValue'] = (df_merged['ParamValue'] - df_merged['Lower_Bound']) / (df_merged['Upper_Bound'] - df_merged['Lower_Bound'])
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

def plot_parameter_space(df_samples:pd.DataFrame, df_param_space:pd.DataFrame, params:dict=None, real_value:dict=None, axis=None):
    """    Plot the parameter space from the samples DataFrame.
    Parameters:
    - df_samples: DataFrame containing the samples.
    - df_param_space: DataFrame defining the search space for each parameter.
    - params: Dictionary with parameter names as keys and their best values as values (optional).
    - real_value: Dictionary with real parameter values to plot (optional).
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Normalize the parameter space
    df_plot = normalize_params(df_samples, df_param_space)

    # Plotting
    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = axis
    sns.scatterplot(df_plot, y='ParamName', x='ParamValue', hue='SampleID', legend=True, ax=ax)

    # Get existing legend handles and labels for Sample ID
    sample_handles, sample_labels = ax.get_legend_handles_labels()
    
    # Create separate lists for special markers
    special_handles = []
    special_labels = []
    
    if real_value:
        real_value_df = pd.DataFrame(real_value.items(), columns=['ParamName', 'ParamValue'])
        real_value_df['SampleID'] = 0  # Add SampleID as 0 for real values
        real_value_norm = normalize_params(real_value_df, df_param_space)
        real_scatter = ax.scatter(real_value_norm['ParamValue'], real_value_norm['ParamName'], 
                                color='blue', label='Real Value', marker='*', s=100, zorder=5)
        special_handles.append(real_scatter)
        special_labels.append('Real Value')
        
    if params:
        for key, param in params.items():
            param_df = pd.DataFrame(param.items(), columns=['ParamName', 'ParamValue'])
            param_df['SampleID'] = id  # Add SampleID as 1 for best parameters
            param_norm = normalize_params(param_df, df_param_space)
            param_scatter = ax.scatter(param_norm['ParamValue'], param_norm['ParamName'], marker='x', s=100, zorder=5)
            special_handles.append(param_scatter)
            special_labels.append(key)
    
    # Create legends dynamically based on what's available
    if sample_handles and special_handles:
        # Both sample data and special markers exist - create two legends
        legend1 = ax.legend(sample_handles, sample_labels, loc='upper right', bbox_to_anchor=(1.2, 1), 
                           fontsize='small', title='Sample ID')
        legend2 = ax.legend(special_handles, special_labels, loc='upper right', bbox_to_anchor=(1.2, 0.6), 
                           fontsize='small', title='Ref. Points')
        ax.add_artist(legend1)
    elif sample_handles:
        # Only sample data exists - single legend
        ax.legend(sample_handles, sample_labels, loc='upper right', bbox_to_anchor=(1.2, 1), 
                 fontsize='small', title='Sample ID')
    elif special_handles:
        # Only special markers exist - single legend
        ax.legend(special_handles, special_labels, loc='upper right', bbox_to_anchor=(1.2, 1), 
                 fontsize='small', title='Ref. Points')

    ax.set_title('Parameter Space')
    ax.set_xlabel('Normalized Parameter Value')
    ax.set_ylabel('')

    if axis is None:
        plt.tight_layout()
        return fig, ax

def plot_parameter_space_db(db_file:str, params:dict=None, real_value:dict=None, axis=None):
    """
    Plot the parameter space from the database file.
    Parameters:
    - db_file: Path to the database file.
    - params: Dictionary with parameter names as keys and their best values as values (optional).
    - real_value: Dictionary with real parameter values to plot (optional).
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output  = load_structure(db_file)
    return plot_parameter_space(df_samples, df_param_space, params, real_value, axis)

def plot_parameter_vs_fitness(df_samples:pd.DataFrame, df_output:pd.DataFrame, parameter_name:str, qoi_name:str, samples_id=None, axis=None):
    """
    Plot the parameter values against the fitness values.
    Parameters:
    - df_samples: DataFrame containing the samples.
    - df_output: DataFrame containing the output of the analysis.
    - parameter_name: Name of the parameter to plot.
    - qoi_name: Name of the QoI to plot against the parameter.
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Sort the parameter values
    df_sorted_params = df_samples[df_samples['ParamName'] == parameter_name].sort_values(by='ParamValue').reset_index()
    # Find the corresponding fitness values for the sorted SampleIDs
    df_sorted_fitness = df_output.set_index('SampleID').loc[df_sorted_params['SampleID']]
    df_sorted_fitness = df_sorted_fitness.reset_index()
    for sample_id in df_sorted_fitness['SampleID']:
        df_sorted_fitness.loc[df_sorted_fitness['SampleID'] == sample_id, 'ObjFunc'] = df_sorted_fitness.loc[df_sorted_fitness['SampleID'] == sample_id, 'ObjFunc'].values[0][qoi_name]
    # print(f"Sorted Parameters:\n{df_sorted_params}")
    # print(f"Sorted Objectives:\n{df_sorted_objectives}")

    # Plotting
    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Parameter vs Objective: {parameter_name} vs {qoi_name}")
    else:
        ax = axis

    ax.scatter(df_sorted_params['ParamValue'], df_sorted_fitness['ObjFunc'], marker='o', c='gray', zorder=1)
    if samples_id:
        ax.scatter(df_sorted_params.loc[df_sorted_params['SampleID'].isin(samples_id)]['ParamValue'], 
                   df_sorted_fitness.loc[df_sorted_fitness['SampleID'].isin(samples_id), 'ObjFunc'], c='red', label='Selected Samples', marker='x', zorder=2)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(f"Fitness({qoi_name})")

    if axis is None:
        plt.tight_layout()
        return fig, ax

def plot_parameter_vs_fitness_db(db_file:str, parameter_name:str, qoi_name:str, axis=None):
    """
    Plot the parameter space against the fitness values from the database file.
    Parameters:
    - db_file: Path to the database file.
    - parameter_name: Name of the parameter to plot.
    - qoi_name: Name of the QoI to plot against the parameter.
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)
    return plot_parameter_vs_fitness(df_samples, df_output, parameter_name, qoi_name, axis)


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

def plot_qoi_param(df_ObsData:pd.DataFrame, df_output:pd.DataFrame, samples_id:list, x_var: str, y_var:str, axis=None):
    """
    Plot the QoI parameter space from the database file.
    Parameters:
    - df_ObsData: Observed QoI DataFrame.
    - df_output: Output DataFrame.
    - samples_id: List of Sample IDs to plot.
    - x_var: Variable to plot on the x-axis.
    - y_var: Variable to plot on the y-axis.
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Load the model results associated with the parameters
    selected_outputs = df_output[df_output['SampleID'].isin(samples_id)]
    print(f"Sample ID: {samples_id}")
    print(f"Objective Function Values:\n{selected_outputs['ObjFunc'].values[0]}")
    print(f"Noise of Objective Function:\n{selected_outputs['Noise_Std'].values[0]}")
        
    if axis is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = axis

    # Plot observed data if available
    sns.lineplot(df_ObsData, x=x_var, y=y_var, color='red', label='Observed QoI', linewidth=3, ax=ax)

    all_df_data = pd.DataFrame()
    # Plot each QoI against the model results associated with the dic_param
    for sample_id in samples_id:
        dic_data = df_output[df_output['SampleID'] == sample_id]['Data'].values[0]
        for rep_id, output in dic_data.items():
            df_data = pd.DataFrame(output, columns=[x_var, y_var])
            df_data['SampleID'] = sample_id
            df_data['replicateID'] = rep_id
            if all_df_data.empty: all_df_data = df_data.copy()
            else: all_df_data = pd.concat([all_df_data, df_data], ignore_index=True)

    # Plot PhysiCell replicates with only one legend entry using seaborn
    # Add formatted SampleID for better legend display
    all_df_data['SampleID_formatted'] = all_df_data['SampleID'].apply(lambda x: f'SampleID: {x}')
    sns.lineplot(data=all_df_data, x=x_var, y=y_var, ax=ax,
        hue='SampleID_formatted', units='replicateID', dashes=(4,2), estimator=None)

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.legend()
    
    if axis is None:
        plt.tight_layout()  
        return fig, ax

def plot_qoi_param_db(db_file:str, samples_id:list, x_var: str=None, y_var:str=None, axis=None):
    """
    Plot the QoI parameter space from the database file.
    Parameters:
    - db_file: Path to the database file.
    - samples_id: List of Sample IDs to plot.
    - x_var: Variable to plot on the x-axis (optional).
    - y_var: Variable to plot on the y-axis (optional).
    - axis: Matplotlib axis to plot on (optional).
    Returns:
    - Matplotlib figure and axis if axis is None, otherwise returns the axis.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)
    df_ObsData = get_observed_qoi(df_metadata['ObsData_Path'].values[0], df_qois)
    return plot_qoi_param(df_ObsData, df_output, samples_id, x_var, y_var, axis)