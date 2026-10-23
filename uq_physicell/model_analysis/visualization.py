import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import re

def convert_df_to_df_over_time(df_summary, selected_qoi) -> pd.DataFrame:
    """Reshape a wide-format QoI summary DataFrame into long format for a single QoI.

    Converts columns formatted as '{selected_qoi}_{time_index}' / 'time_{time_index}'
    (one column per timestep, indexed by SampleID) into three long-format columns
    ('Time', selected_qoi, 'SampleID') suitable for seaborn plotting.

    Args:
        df_summary (pd.DataFrame): Wide-format QoI summary, indexed by SampleID, with
            columns '{selected_qoi}_{time_index}' (or just selected_qoi, if there is a
            single time point) and matching 'time_{time_index}' columns.
        selected_qoi (str): Name of the QoI to extract.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns 'Time', selected_qoi, and 'SampleID'.
    """
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

def get_mcse_legend_handles() -> list[Patch]:
    """Build the legend handles for the standard relative-MCSE reliability bands.

    Used internally by plot_qoi_over_time (when plot_mcse_range=True) and can be reused
    directly when building a combined legend across multiple subplots.

    Returns:
        list[matplotlib.patches.Patch]: One Patch per MCSE band (Excellent <1%,
        Acceptable [1%,5%], Cautionary [5%,10%], Unreliable >10%), each already carrying
        its color and label.
    """
    return [
        Patch(facecolor='green', alpha=0.3, label='Excelent (<1%)'),
        Patch(facecolor='blue', alpha=0.3, label='Acceptable ([1%,5%])'),
        Patch(facecolor='yellow', alpha=0.3, label='Cautionary ([5%,10%])'),
        Patch(facecolor='red', alpha=0.3, label='Unreliable (>10%)')
    ]

def plot_qoi_over_time(df_plot, selected_qoi, ax, plot_mcse_range=False, show_legend=True) -> None:
    """Plot one QoI over time, one line per SampleID, on a given matplotlib axis.

    Accepts either a long-format DataFrame with a 'time' index level (as returned by
    calculate_qoi_statistics) or a wide-format summary DataFrame (converted internally
    via convert_df_to_df_over_time). Uses a swarm plot instead of a line plot when the
    data has a single time point.

    Args:
        df_plot (pd.DataFrame): QoI data, either long-format with 'SampleID'/'time' index
            levels, or wide-format with '{selected_qoi}_{time_index}' columns.
        selected_qoi (str): Name of the QoI to plot.
        ax (matplotlib.axes.Axes): Axis to draw on.
        plot_mcse_range (bool, optional): If True, overlay the standard relative-MCSE
            reliability bands (see get_mcse_legend_handles) as colored horizontal spans.
            Intended for use with a relative-MCSE DataFrame as df_plot. Defaults to False.
        show_legend (bool, optional): If False, suppress the per-axes sample legend and
            MCSE legend instead of drawing them — use this when combining several subplots'
            legends into a single shared legend for the whole figure. Defaults to True.

    Returns:
        None. The plot is drawn in place on ax.
    """
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
    # Set show_legend=False to suppress per-axes legends, e.g. when combining them
    # into a single shared legend for the whole figure.
    sample_legend = ax.get_legend()
    if sample_legend is not None:
        if show_legend:
            ax.add_artist(sample_legend)
        else:
            sample_legend.remove()

    if plot_mcse_range:
        mcse_handles = get_mcse_legend_handles()
        if show_legend:
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



def plot_global_sa_results(param_names, sa_method, qoi_time_values, sa_results, selected_qoi, selected_sm, ax, selected_ci=None) -> None:
    """Plot global sensitivity indices (e.g. Sobol S1/ST) over time for one QoI, on a given axis.

    Draws one line (or, for a single time point, one bar) per parameter, showing how
    the selected sensitivity measure evolves over time. Intended to be called with the
    output of get_sa_results.

    Args:
        param_names (list[str]): Parameter names, in the order used to index sa_results.
        sa_method (str): Name of the sensitivity analysis method, used only for the title.
        qoi_time_values (dict): Mapping of time labels to their (numeric) time values, as
            returned by get_sa_results.
        sa_results (dict): Nested results as returned by get_sa_results, structured as
            {qoi_name: {time_label: {sensitivity_measure: [value_per_param]}}}.
        selected_qoi (str): QoI to plot (a key of sa_results).
        selected_sm (str): Sensitivity measure to plot (e.g. 'S1', 'ST'), a key of
            sa_results[selected_qoi][time_label].
        ax (matplotlib.axes.Axes): Axis to draw on.
        selected_ci (str): Confidence interval to plot (e.g. 'S1_conf', 'ST_conf'), a key of sa_results[selected_qoi][time_label][selected_sm].

    Returns:
        None. The plot is drawn in place on ax.
    """
    plot_data = pd.DataFrame([
        {
            "Time": qoi_time_values[time_label],
            # None (not the Sensitivity Index itself) when no CI was requested --
            # SALib's *_conf is a half-width margin in S1's own units, not a
            # substitute value, so mixing them here was never a valid quantity.
            "Confidence Interval": None if selected_ci is None else sa_results[selected_qoi][time_label][selected_ci][param_id],
            "Sensitivity Index": sa_results[selected_qoi][time_label][selected_sm][param_id],
            "Parameter": param
        }
        for time_label in sa_results[selected_qoi].keys()
        for param_id, param in enumerate(param_names)
    ])
    custom_palette = sns.color_palette("tab20", len(plot_data["Parameter"].unique()))
    color_by_param = dict(zip(param_names, custom_palette))
    # If just one time point, use barplot, else use lineplot. seaborn's own
    # errorbar= only ever aggregates *repeated* observations within an (x, hue)
    # group -- with a single precomputed point per (Time, Parameter) here, it
    # never even calls a custom callable (verified: passing one fires zero
    # times), so the CI has to be drawn manually from the "Confidence Interval"
    # column instead, right after each seaborn call, only when selected_ci was
    # requested: whisker-style error bars with clearly visible caps on the bar
    # plot, a shaded confidence band ("shadow") around each line otherwise.
    if len(sa_results[selected_qoi].keys()) == 1:
        sns.barplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=param_names, errorbar=None)
        if selected_ci is not None:
            # Bars are dodged per hue, so align each error bar with its bar's
            # actual (patch) x-position rather than the shared "Time" value.
            for patch, param in zip(ax.patches, param_names):
                x_center = patch.get_x() + patch.get_width() / 2
                row = plot_data.loc[plot_data["Parameter"] == param].iloc[0]
                ax.errorbar(x_center, row["Sensitivity Index"], yerr=row["Confidence Interval"],
                            fmt='none', ecolor='black', elinewidth=1.5, capsize=5, capthick=1.5, zorder=3)
    else:
        sns.lineplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=param_names, errorbar=None)
        if selected_ci is not None:
            for param in param_names:
                sub = plot_data[plot_data["Parameter"] == param]
                ax.fill_between(sub["Time"], sub["Sensitivity Index"] - sub["Confidence Interval"],
                                 sub["Sensitivity Index"] + sub["Confidence Interval"],
                                 color=color_by_param[param], alpha=0.2, linewidth=0, zorder=1)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"Sensitivity Measure ({selected_sm})")
    ax.set_title(f"Global SA - {sa_method}", fontsize=8)
    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)

def plot_sobol_convergence(N_values: list, convergence: dict, qoi_names: list, param_names: list, axes, xlim: tuple = None, ylim: tuple = None) -> None:
    """Plot Sobol total-order index (ST) vs. sample size N, one panel per QoI.

    Draws one line per parameter with a shaded confidence-interval ribbon (from
    SALib's ST_conf) on a log2 x-axis, so equal spacing between successive
    doublings of N makes a flattening (converged) trend visually obvious.
    Intended to be called with the output of get_sobol_convergence.

    Args:
        N_values (list): Sobol base sample sizes, in the same order used to build convergence.
        convergence (dict): Nested results as returned by get_sobol_convergence,
            structured as {qoi_name: {param_name: {'ST': [...], 'ST_conf': [...]}}}.
        qoi_names (list): QoIs to plot, one subplot each, in the given order.
        param_names (list): Parameter names, in a fixed order for consistent coloring.
        axes: Sequence of matplotlib axes, one per qoi_names entry.
        xlim (tuple, optional): The x-axis limits for the plot.
        ylim (tuple, optional): The y-axis limits for the plot.

    Returns:
        None. The plots are drawn in place on axes.
    """
    colors = sns.color_palette("tab20", len(param_names))
    for ax, qoi in zip(axes, qoi_names):
        for pid, param in enumerate(param_names):
            st = np.array(convergence[qoi][param]['ST'])
            conf = np.array(convergence[qoi][param]['ST_conf'])
            ax.plot(N_values, st, marker='o', color=colors[pid % len(colors)], label=param)
            ax.fill_between(N_values, st - conf, st + conf, color=colors[pid % len(colors)], alpha=0.2, linewidth=0)
        if st.size == 0 or np.all(np.isnan(st)):  # Check if all values are NaN or empty for this qoi and parameter combination
            ax.remove()  # Remove the empty subplot if no data for this qoi and parameter combination
            continue  # Skip if no data for this qoi and parameter combination
        else:
            # Plot a horizontal green band from 0 to 1 to indicate the valid range of Sobol indices
            ax.axhspan(0, 1, alpha=0.2, color='gray', zorder=0)
            ax.set_xscale('log', base=2)
            ax.set_xticks(N_values)
            ax.set_xticklabels(N_values)
            ax.set_xlabel('N (Sobol base sample size)')
            ax.set_ylabel('Total-order Sobol index (ST)')
            ax.set_title(qoi)
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
    if len(axes) > 0:
        axes[0].legend(loc='best', fontsize=8)

def plot_local_sa_results(sa_method, qoi_time_values, sa_results, selected_qoi, ax) -> None:
    """Plot local (OAT) sensitivity indices over time for one QoI, on a given axis.

    Draws one line (or, for a single time point, one bar) per parameter, ordered by
    descending maximum sensitivity so the most influential parameters are listed first
    in the legend. Intended to be called with the output of get_sa_results(method="OAT").

    Args:
        sa_method (str): Name of the sensitivity analysis method, used only for the title.
        qoi_time_values (dict): Mapping of time labels to their (numeric) time values, as
            returned by get_sa_results.
        sa_results (dict): Nested results as returned by get_sa_results, structured as
            {qoi_name: {time_label: {parameter_name: sensitivity_value}}}.
        selected_qoi (str): QoI to plot (a key of sa_results).
        ax (matplotlib.axes.Axes): Axis to draw on.

    Returns:
        None. The plot is drawn in place on ax.
    """
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


def plot_cells_2D(df_cells, color_dic=None, ax=None, scale_bar=False, bar_size=200, axes_visible=False, feature='cell_type', cmap='viridis', vmin=None, vmax=None) -> PatchCollection:
    """Plot a 2D snapshot of cells as circles, colored by a categorical or numeric feature.

    Each cell is drawn as a circle at (position_x, position_y) with its actual radius.
    Coloring is either an explicit lookup (color_dic), a continuous colormap (when
    feature is numeric), or an automatically assigned categorical colormap (otherwise).

    Args:
        df_cells (pd.DataFrame): Cell data with 'position_x', 'position_y', 'radius',
            and the column named by feature (e.g. a pcdl cell DataFrame for one timestep).
        color_dic (dict, optional): Mapping from feature value to a matplotlib color.
            Takes precedence over cmap-based coloring when provided. Defaults to None.
        ax (matplotlib.axes.Axes): Axis to draw on. Required — raises ValueError if None.
        scale_bar (bool, optional): If True, draw a physical-scale bar. Defaults to False.
        bar_size (float, optional): Length of the scale bar, in the same units as
            position_x/position_y (typically microns). Defaults to 200.
        axes_visible (bool, optional): If True, keep the axis spines and tick marks
            visible instead of hiding them. Defaults to False.
        feature (str, optional): Column of df_cells used for coloring when color_dic is
            not given. Defaults to 'cell_type'.
        cmap (str, optional): Matplotlib colormap name used when color_dic is not given.
            Defaults to 'viridis'.
        vmin (float, optional): Lower bound for color normalization when feature is
            numeric. Defaults to the data minimum.
        vmax (float, optional): Upper bound for color normalization when feature is
            numeric. Defaults to the data maximum.

    Returns:
        matplotlib.collections.PatchCollection: The collection of cell circles added to ax.
    """
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