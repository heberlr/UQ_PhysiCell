import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# My local modules
from .utils import normalize_params_df, get_observed_qoi
from ..database.bo_db import load_structure

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
    df_plot = normalize_params_df(df_samples, df_param_space)

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
        real_value_norm = normalize_params_df(real_value_df, df_param_space)
        real_scatter = ax.scatter(real_value_norm['ParamValue'], real_value_norm['ParamName'], 
                                color='blue', label='Real Value', marker='*', s=100, zorder=5)
        special_handles.append(real_scatter)
        special_labels.append('Real Value')
        
    if params:
        for key, param in params.items():
            param_df = pd.DataFrame(param.items(), columns=['ParamName', 'ParamValue'])
            param_df['SampleID'] = id  # Add SampleID as 1 for best parameters
            param_norm = normalize_params_df(param_df, df_param_space)
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