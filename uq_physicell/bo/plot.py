import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .database import load_structure

def normalize_params(df_params, df_search_space):
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

def extract_best_parameters(db_file:str) -> dict:
    """
    Extract the best parameters from the database file based on the maximum hypervolume.
    Parameters:
    - db_file: Path to the database file.
    Returns:
    - Dictionary with parameter names as keys and their best values as values.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)

    # Find the maximum hypervolume and its corresponding iteration ID (if multiple, take the first)
    max_hypervolume = df_gp_models['Hypervolume'].max()
    best_iteration = df_gp_models[df_gp_models['Hypervolume'] == max_hypervolume]['IterationID'].min()

    # Get the parameters for the best iteration
    best_params = df_samples[df_samples['IterationID'] == best_iteration].set_index('ParamName')['ParamValue'].to_dict()

    return best_params, best_iteration

def plot_parameter_space(db_file:str, best_param:dict=None, real_value:dict=None):
    """
    Plot the parameter space from the database file.
    Parameters:
    - db_file: Path to the database file.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output  = load_structure(db_file)

    # Normalize the parameter space
    df_plot = normalize_params(df_samples, df_param_space)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(df_plot, y='ParamName', x='ParamValue', hue='SampleID', legend=True)
    
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
        
    if best_param:
        best_param_df = pd.DataFrame(best_param.items(), columns=['ParamName', 'ParamValue'])
        best_param_df['SampleID'] = 1  # Add SampleID as 1 for best parameters
        best_param_norm = normalize_params(best_param_df, df_param_space)
        best_scatter = ax.scatter(best_param_norm['ParamValue'], best_param_norm['ParamName'], 
                                color='red', label='Max Value', marker='x', s=100, zorder=5)
        special_handles.append(best_scatter)
        special_labels.append('Max Value')
    
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
    plt.tight_layout()
    plt.show()

def plot_qoi_param(db_file:str, iteration:int, obs_qoi:dict={},x_vars_dict:dict={}, y_vars_dict:dict={}):
    """
    Plot the QoI parameter space from the database file.
    Parameters:
    - db_file: Path to the database file.
    - iteration: The iteration number to plot the QoI parameters for.
    """
    # Load the database structure
    df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)


    # Load the model results associated with the parameters
    sample_id = df_samples[df_samples['IterationID'] == iteration]['SampleID'].values[0]
    selected_output = df_output[df_output['SampleID'] == sample_id]
    print(f"Sample ID: {sample_id}")
    print(f"Iteration: {iteration}")
    print(f"Objective Function Values:\n{selected_output['ObjFunc'].values[0]}")
    print(f"Noise of Objective Function:\n{selected_output['Noise_Std'].values[0]}")

    # Select the x_var_list and y_var_list from the output data
    if not x_vars_dict and not y_vars_dict:
        # Load the Observed QoI values
        df_obs_qoi = pd.read_csv(df_metadata['ObsData_Path'].values[0])
        dic_columns = df_qois.set_index('ObsData_Column')['QoI_Name'].to_dict()
        obs_qoi = df_obs_qoi.rename(columns=dic_columns)
        # Create x_vars_dict and y_vars_dict based on the observed QoI columns
        x_vars_dict = { 'model': ["Time"]* len(dic_columns), 'data': ["time"]* len(dic_columns)}
        y_vars_dict = { 'model': [], 'data': []}
        for qoi_name in dic_columns.values():
            y_vars_dict['model'].append(qoi_name)
            y_vars_dict['data'].append(qoi_name)

    # Plot each QoI against the model results associated with the dic_param
    for model_x_var, model_y_var, data_x_var, data_y_var in zip(x_vars_dict['model'], y_vars_dict['model'], x_vars_dict['data'], y_vars_dict['data']):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=obs_qoi[data_x_var], y=obs_qoi[data_y_var], ax=ax, color='red', label='Observed QoI')

        # Plot PhysiCell replicates with only one legend entry using seaborn
        sns.lineplot(data=selected_output['Data'].values[0], x=model_x_var, y=model_y_var, ax=ax,
                    color='blue', units='replicateID', estimator=None)
        # Add a single legend entry for PhysiCell manually
        ax.plot([], [], color='blue', label='PhysiCell')

        ax.set_title(f"QoI: {data_y_var}")
        ax.set_xlabel(data_x_var)
        ax.set_ylabel("Value")
        ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    db_path = "examples/virus-mac-new/BO_calibration.db"  # Path to the database file
    dic_real_value = {
        'mac_phag_rate_infected': 1.0,
        'mac_motility_bias': 0.15,
        'epi2infected_sat': 0.1,
        'epi2infected_hfm': 0.4
    }  # Example real values for the parameters
    best_param, best_iteration = extract_best_parameters(db_path)
    print("Best Parameters:", best_param)
    plot_parameter_space(db_path, best_param=best_param, real_value=dic_real_value)
    plot_qoi_param(db_path, best_iteration)