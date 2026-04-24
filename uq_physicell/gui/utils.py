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
        perturbation = df_parameter_space['perturbation'].iloc[id]
        if isinstance(perturbation, list):
            perturbation = [float(x) for x in perturbation]
        local_SA_parameters[param] = {"ref_value": df_parameter_space['ref_value'].iloc[id], 
                                                    "perturbation": perturbation}
    return local_SA_parameters

def convert_df_to_df_over_time(df_summary, selected_qoi):
    import pandas as pd
    import re
    # Identify the relevant columns
    qoi_pattern = re.compile(rf"^{re.escape(selected_qoi)}_(\d+)$")
    qoi_columns = sorted([
        col for col in df_summary.columns
        if col == selected_qoi or qoi_pattern.match(col)
    ])
    # Extract time IDs from the QoI columns - (Some QoIs may not have multiple time points, e.g., cumulative values over time)
    time_ids = [col.split(f"{selected_qoi}_")[-1] for col in qoi_columns]
    time_columns = sorted([f"time_{time_id}" for time_id in time_ids])
    # Prepare the data for seaborn
    plot_data = pd.DataFrame({
        "Time": df_summary[time_columns].values.flatten(),
        selected_qoi: df_summary[qoi_columns].values.flatten(),
        "SampleID": df_summary.index.repeat(len(qoi_columns))
    })
    return plot_data

def plot_qoi_over_time(df_plot, selected_qoi, ax, plot_mcse_range=False):
    from matplotlib.patches import Patch
    import numpy as np
    import seaborn as sns
    # Prepare the data for seaborn
    if 'time' in df_plot.index.names:
        df_plot = df_plot.reset_index()
        df_plot.rename(columns={'time': 'Time'}, inplace=True)
        plot_data = df_plot[['Time', selected_qoi, 'SampleID']].dropna()
    else:
        plot_data = convert_df_to_df_over_time(df_plot, selected_qoi)
    # If just one time point, use swarmplot, else use lineplot
    if len(plot_data["Time"].unique()) == 1:
        sns.swarmplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)
    else:
        sns.lineplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)

    # Plot MCSE range if requested
    if plot_mcse_range:
        # Use fixed MCSE intervals and draw only the bands that overlap the plotted QoI scale.
        _, y_max = ax.get_ylim()
        y_min = 0.0
        ranges = [
            (0.0, 0.01, 'green'),
            (0.01, 0.05, 'blue'),
            (0.05, 0.1, 'yellow'),
            (0.1, 1.0, 'red')
        ]
        for lower, upper, color in ranges:
            clipped_lower = max(lower, y_min)
            clipped_upper = min(upper, y_max)
            if clipped_upper > clipped_lower:
                ax.axhspan(clipped_lower, clipped_upper, color=color, alpha=0.1)
        ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(selected_qoi)
    # Add sample legend and, when requested, a dedicated MCSE legend outside the plot.
    sample_legend = ax.get_legend()
    if sample_legend is not None:
        ax.add_artist(sample_legend)

    if plot_mcse_range:
        mcse_handles = [
            Patch(facecolor='green', alpha=0.3, label='Excelent (<1%)'),
            Patch(facecolor='blue', alpha=0.3, label='Acceptable ([1%,5%])'),
            Patch(facecolor='yellow', alpha=0.3, label='Cautionary ([5%,10%])'),
            Patch(facecolor='red', alpha=0.3, label='Unreliable (>10%)')
        ]
        # Place MCSE legend below the plot, then call tight_layout to adjust figure
        mcse_legend = ax.legend(
            handles=mcse_handles,
            title='Ranges of MCSE relative to mean',
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=4,
            fontsize=8,
            frameon=True
        )
        ax.add_artist(mcse_legend)


def plot_global_sa_results(global_SA_parameters, sa_method, qoi_time_values, sa_results, selected_qoi, selected_sm, ax, parameter_order=None):
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
    if parameter_order is None:
        parameter_order = (
            plot_data.groupby("Parameter")["Sensitivity Index"]
            .max()
            .sort_values(ascending=False)
            .index
        )
    custom_palette = sns.color_palette("tab20", len(plot_data["Parameter"].unique()))
    # If just one time point, use barplot, else use lineplot
    if len(sa_results[selected_qoi].keys()) == 1:
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
        # If just one time point, use barplot, else use lineplot
    if len(sa_results[selected_qoi].keys()) == 1:
        sns.barplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)
    else:
        sns.lineplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)
    ax.set_xlabel("Time (min)")
    ax.set_title(f"Local SA - {sa_method}")
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)


def plot_cells_2D(df_cells, color_dic=None, ax=None, scale_bar=False, bar_size=200, axes_visible=False, feature='cell_type', cmap='viridis', vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import pandas as pd
    from matplotlib.collections import PatchCollection
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    if ax is None:
        raise ValueError("'ax' must be provided.")

    patches = []
    for index, row in df_cells.iterrows():
        circle = plt.Circle((row['position_x'], row['position_y']), row['radius'])
        patches.append(circle)

    collection = PatchCollection(patches, edgecolors='black', linewidths=0.5)
    
    if color_dic is not None:
        facecolors = [color_dic[ct] for ct in df_cells[feature]]
        collection.set_facecolors(facecolors)
    elif pd.api.types.is_numeric_dtype(df_cells[feature]):
        feature_values = df_cells[feature].astype(float).to_numpy()
        min_value = feature_values.min() if vmin is None else vmin
        max_value = feature_values.max() if vmax is None else vmax
        if min_value == max_value:
            max_value = min_value + 1e-12
        norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
        collection.set_array(feature_values)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
    else:
        unique_values = list(df_cells[feature].dropna().unique())
        cmap_obj = plt.get_cmap(cmap, max(len(unique_values), 1))
        local_color_dic = {value: cmap_obj(idx) for idx, value in enumerate(unique_values)}
        facecolors = [local_color_dic[value] for value in df_cells[feature]]
        collection.set_facecolors(facecolors)
    ax.add_collection(collection)
    ax.set_aspect('equal')
    ax.autoscale_view()
    # Remove spines
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(axes_visible)
    # Remove the tick marks as well:
    ax.tick_params(left=axes_visible, labelleft=axes_visible, bottom=axes_visible, labelbottom=axes_visible)
    if scale_bar:
        asb = AnchoredSizeBar(ax.transData,
                      bar_size,
                      f"{bar_size} μm",
                      loc="lower right",
                      pad=0.1, borderpad=0.5, sep=5, size_vertical=20,
                      frameon=False)
        ax.add_artist(asb)
    return collection