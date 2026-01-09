def get_global_SA_parameters(db_file):
    from ..database.ma_db import load_parameter_space, load_samples
    df_parameter_space = load_parameter_space(db_file)
    global_SA_parameters = {'samples': load_samples(db_file)}
    for id, param in enumerate(df_parameter_space['ParamName']):
        global_SA_parameters[param] = {"lower_bound": df_parameter_space['lower_bound'].iloc[id],
                                                            "upper_bound": df_parameter_space['upper_bound'].iloc[id],
                                                            "ref_value": df_parameter_space['ref_value'].iloc[id]}
        perturbation = df_parameter_space['perturbation'].iloc[id]
        try:
            global_SA_parameters[param]["perturbation"] = float(perturbation)
        except Exception as e:
            print(f"Warning: Could not convert perturbation ({perturbation}) to float for parameter {param}.")
            # Calculate perturbation as percentage based on bounds and ref_value
            global_SA_parameters[param]["perturbation"] = 100.0 * (df_parameter_space['upper_bound'].iloc[id]/df_parameter_space['ref_value'].iloc[id] - 1.0)
    return global_SA_parameters

def get_local_SA_parameters(db_file):
    from ..database.ma_db import load_parameter_space, load_samples
    df_parameter_space = load_parameter_space(db_file)
    local_SA_parameters = {'samples': load_samples(db_file)}
    for id, param in enumerate(df_parameter_space['ParamName']):
        if type(df_parameter_space['perturbation'].iloc[id]) == list:
            df_parameter_space['perturbation'].iloc[id] = [float(x) for x in df_parameter_space['perturbation'].iloc[id]]
        local_SA_parameters[param] = {"ref_value": df_parameter_space['ref_value'].iloc[id], 
                                                    "perturbation": df_parameter_space['perturbation'].iloc[id]}
    return local_SA_parameters

def plot_qoi_over_time(df_plot, selected_qoi, ax):
    import pandas as pd
    import seaborn as sns
    # Identify the relevant columns
    qoi_columns = sorted([col for col in df_plot.columns if col.startswith(selected_qoi)])
    time_columns = sorted([col for col in df_plot.columns if col.startswith("time_")])
    # Prepare the data for seaborn
    plot_data = pd.DataFrame({
        "Time": df_plot[time_columns].values.flatten(),
        selected_qoi: df_plot[qoi_columns].values.flatten(),
        "SampleID": df_plot.index.repeat(len(qoi_columns))
    })
    # If just one time point, use swarmplot, else use lineplot
    if len(time_columns) == 1:
        sns.swarmplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)
    else:
        sns.lineplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(selected_qoi)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(title="Sample Index")

def plot_global_sa_results(global_SA_parameters, sa_method, qoi_time_values, sa_results, selected_qoi, selected_sm, ax):
    import pandas as pd
    import seaborn as sns
    # This is necessary because Sobol method does not return the names of the parameters
    param_names = [key for key in global_SA_parameters.keys() if key != "samples"]
    plot_data = pd.DataFrame([
        {
            "Time": qoi_time_values[time_label],
            "Sensitivity Index": sa_results[selected_qoi][time_label][selected_sm][param_id],
            "Parameter": param
        }
        for time_label in sa_results[selected_qoi].keys()
        for param_id, param in enumerate(param_names)
    ])
    # print(plot_data)
    # Sort Parameters by the maximum Sensitivity Index in descending order
    parameter_order = (
        plot_data.groupby("Parameter")["Sensitivity Index"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    custom_palette = sns.color_palette("tab20", len(plot_data["Parameter"].unique()))
    # If just one time point, use barplot, else use lineplot
    if len(qoi_time_values) == 1:
        sns.barplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)
    else:
        sns.lineplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)                
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"Sensitivity Measure ({selected_sm})")
    ax.set_title(f"Global SA - {sa_method}", fontsize=8)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)

def plot_local_sa_results(sa_method, qoi_time_values, sa_results, selected_qoi, ax):
    import pandas as pd
    import seaborn as sns
    plot_data = pd.DataFrame([
        {
            "Time": qoi_time_values[time_label],
            "Sensitivity Index": sa_results[selected_qoi][time_label][param],
            "Parameter": param
        }
        for time_label in sa_results[selected_qoi].keys()
        for param in sa_results[selected_qoi][time_label].keys()
    ])
    # print(plot_data)
    # Sort Parameters by the maximum Sensitivity Index in descending order
    parameter_order = (
        plot_data.groupby("Parameter")["Sensitivity Index"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    custom_palette = sns.color_palette("tab20", len(plot_data["Parameter"].unique()))
    sns.lineplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"Local SA - {sa_method}")
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)
    