import numpy as np
import os
import json

# All the specific classes we need
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# My local modules
from uq_physicell.database.bo_db import load_parameter_space, load_metadata, load_qois, load_gp_models, migrate_bo_database


def add_db_files_to_table(main_window):
    files, _ = QFileDialog.getOpenFileNames(main_window, "Select Calibration .db Files", "", "Database Files (*.db)")
    if not files:
        return
    for file in files:
        row_position = main_window.db_table.rowCount()
        # A model's id is permanent once assigned and never reused or renumbered,
        # unlike deriving it from the current row count/position: as soon as any
        # earlier row has been removed, "row_position + 1" collides with a key
        # still present in dic_table (removal only relabels the visible Model
        # column text, it never touched dic_table's keys), silently overwriting
        # that other model's stored data, or later KeyError-ing on a lookup that
        # assumed the keys stay contiguous.
        model_id = main_window._next_model_id
        main_window._next_model_id += 1
        main_window.dic_table[model_id] = {"Database_path": file}
        try:
            migrate_bo_database(file)  # upgrade an older calibration .db to the current schema
            main_window.dic_table[model_id]["Num_Params"] = load_parameter_space(file).shape[0]
            main_window.dic_table[model_id]["Score"] = np.max(load_gp_models(file)['Score'])
        except Exception as e:
            main_window.dic_table.pop(model_id, None)
            QMessageBox.warning(main_window, "DB File not supported", f"Error occurred while loading parameters and scores from {os.path.basename(file)}: {e}")
            return
        if row_position > 0:
            # Check consistency against whichever file is currently listed in the
            # previous row -- found via that row's own stored model_id, not
            # "model_id - 1", which is not guaranteed to still exist once any row
            # has ever been removed.
            prev_model_id = main_window.db_table.item(row_position - 1, 0).data(Qt.UserRole)
            file_prev = main_window.dic_table[prev_model_id]["Database_path"]
            try:
                consistency_warnings = check_consistency_dbs(load_metadata(file_prev), load_metadata(file), load_qois(file_prev), load_qois(file))
            except ValueError as e:
                main_window.dic_table.pop(model_id, None)
                QMessageBox.warning(main_window, "Inconsistent DB Files", f"Database file {os.path.basename(file)} has inconsistent QoIs or BO setup compared to previous files: {e}")
                return
            if consistency_warnings:
                QMessageBox.warning(
                    main_window, "Model comparison caveats",
                    f"{os.path.basename(file)} was added, but note:\n\n" + "\n\n".join(f"• {w}" for w in consistency_warnings),
                )

        main_window.db_table.insertRow(row_position)
        # Model number: displays the permanent model_id (not necessarily a dense
        # 1..N sequence once rows have been removed), also stashed as the item's
        # UserRole data so remove_db_row can pop the right dic_table entry.
        model_item = QTableWidgetItem(str(model_id))
        model_item.setData(Qt.UserRole, model_id)
        main_window.db_table.setItem(row_position, 0, model_item)
        # Database name
        main_window.db_table.setItem(row_position, 1, QTableWidgetItem(os.path.basename(file)))
        # Number of Parameters
        main_window.db_table.setItem(row_position, 2, QTableWidgetItem(str(main_window.dic_table[model_id]["Num_Params"])))
        # Score (hypervolume for multi-objective, best fitness for single-objective)
        main_window.db_table.setItem(row_position, 3, QTableWidgetItem(f"{main_window.dic_table[model_id]['Score']:.4e}"))
        # Add remove button. Captures the button itself, not the row index: an
        # index captured at connect time goes stale the moment an earlier row is
        # removed (QTableWidget shifts remaining rows up, but a plain Python
        # closure has no way to know that), which would remove the wrong row.
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda _, btn=remove_btn: remove_db_row(main_window, btn))
        main_window.db_table.setCellWidget(row_position, 4, remove_btn)

def remove_db_row(main_window, remove_btn):
    # Resolve the button's current row at click time rather than trusting a row
    # index captured when it was created (see the comment in add_db_files_to_table).
    for row in range(main_window.db_table.rowCount()):
        if main_window.db_table.cellWidget(row, 4) is remove_btn:
            model_item = main_window.db_table.item(row, 0)
            model_id = model_item.data(Qt.UserRole) if model_item is not None else None
            main_window.db_table.removeRow(row)
            main_window.dic_table.pop(model_id, None)
            return

# BO_Options entries that define the objective / score itself. A mismatch makes the
# scores meaningless to compare, so it is a hard error. Everything else (parameter
# space, iteration budget, initial samples, workers, replicates, fixed params, ...)
# is allowed to differ between the models being compared.
_SCORE_DEFINING_OPTIONS = ("use_exponential_fitness", "use_correlated_gp", "ref_point", "custom_aggregation_func")


def check_consistency_dbs(df_metadata_prev, df_metadata, df_qois_prev, df_qois):
    """
    Check that two calibration databases produce comparable scores for model selection.

    Hard-compared (raise ValueError): the score-defining BO_Options entries
    (use_exponential_fitness, use_correlated_gp, ref_point, custom_aggregation_func)
    and the full QoIs table (names, observed-data columns, distance functions and weights) --
    but only when BO_Options is actually available for both databases (see below).

    Soft-compared (returned as warnings, not fatal): ObsData_Path, summary_function,
    botorch_version, and a missing BO_Options itself - differences here *may* still
    allow a fair comparison but are worth flagging to the user.

    A database created before BO_Options was tracked has it as NULL after migration,
    with no way to recover the original settings -- treating that as a hard error
    would permanently lock every pre-upgrade calibration run out of this comparison
    tab. Warn instead: the score-defining settings simply couldn't be verified.

    NOT compared: parameter space, iteration budget, worker/replicate counts, fixed
    params or any other tuning knob.

    Returns:
        list[str]: soft-warning messages (empty if none).

    Raises:
        ValueError: if the databases are not comparable.
    """
    def _bo_options(df):
        if 'BO_Options' not in df.columns or df['BO_Options'].iloc[0] is None:
            return None
        return json.loads(df['BO_Options'].iloc[0])

    opts_prev = _bo_options(df_metadata_prev)
    opts_current = _bo_options(df_metadata)

    consistency_warnings = []
    if opts_prev is None or opts_current is None:
        consistency_warnings.append(
            "BO setup (BO_Options) is missing for one or both databases (created before "
            "this was tracked) -- score-defining settings "
            f"({', '.join(_SCORE_DEFINING_OPTIONS)}) could not be verified as consistent "
            "between these calibrations; compare their scores with that in mind."
        )
    else:
        for key in _SCORE_DEFINING_OPTIONS:
            if opts_prev.get(key) != opts_current.get(key):
                raise ValueError(
                    f"Inconsistent '{key}' between calibrations "
                    f"(previous: {opts_prev.get(key)}, current: {opts_current.get(key)}); "
                    f"their scores are not comparable."
                )

    qois_prev = df_qois_prev.sort_values('QoI_Name').reset_index(drop=True)
    qois_current = df_qois.sort_values('QoI_Name').reset_index(drop=True)
    if not qois_current.equals(qois_prev):
        raise ValueError(f"Inconsistent QoIs between files.\nPrevious:\n{qois_prev}\nCurrent:\n{qois_current}")

    # Soft checks - flag but do not block the comparison
    def _meta(df, col):
        return df[col].iloc[0] if col in df.columns and len(df) else None

    prev_obs, cur_obs = _meta(df_metadata_prev, 'ObsData_Path'), _meta(df_metadata, 'ObsData_Path')
    if prev_obs != cur_obs:
        consistency_warnings.append(
            f"Observed-data path differs (previous: {prev_obs}, current: {cur_obs}). "
            f"Scores are only comparable if the underlying data is identical."
        )
    if opts_prev is not None and opts_current is not None and opts_prev.get('summary_function') != opts_current.get('summary_function'):
        consistency_warnings.append(
            f"summary_function differs (previous: {opts_prev.get('summary_function')}, "
            f"current: {opts_current.get('summary_function')})."
        )
    prev_bt, cur_bt = _meta(df_metadata_prev, 'botorch_version'), _meta(df_metadata, 'botorch_version')
    if prev_bt != cur_bt:
        consistency_warnings.append(
            f"botorch version differs (previous: {prev_bt}, current: {cur_bt}); "
            f"hypervolume computation may not be identical."
        )
    return consistency_warnings


def plot_complexity_vs_score(main_window):
    if main_window.db_table.rowCount() == 0:
        QMessageBox.warning(main_window, "No DB Files", "Please load at least one calibration .db file.")
        return

    # Plot Comparison
    main_window.comparison_figure.clear()
    ax = main_window.comparison_figure.add_subplot(111)
    max_param = 1; min_param = 0
    for model_id, dict_info in main_window.dic_table.items():
        ax.scatter(dict_info["Num_Params"], dict_info["Score"], label=f"Model {model_id}")
        if dict_info["Num_Params"] > max_param:
            max_param = dict_info["Num_Params"]
        if dict_info["Num_Params"] < min_param:
            min_param = dict_info["Num_Params"]
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Score')
    # Integer ticks from min_param to max_param, so a single loaded model still shows a meaningful scale
    ax.set_xticks(range(min_param, max_param + 2, 1))
    ax.set_xlim(min_param - 0.5, max_param + 0.5)
    ax.legend()
    main_window.comparison_canvas.draw()


def create_tab4(main_window):
    layout_tab4 = QVBoxLayout()
    # Title
    load_dbs_label = QLabel("<b>Load Multiple Bayesian Optimization (BO) Calibration .db Files</b>")
    load_dbs_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(load_dbs_label)

    # Load button
    load_db_btn = QPushButton("Add Calibration .db Files")
    load_db_btn.setStyleSheet("background-color: lightgreen; color: black")
    load_db_btn.clicked.connect(lambda: add_db_files_to_table(main_window))
    layout_tab4.addWidget(load_db_btn)

    # Process and plot button
    process_btn = QPushButton("Plot Complexity vs. Score")
    process_btn.setStyleSheet("background-color: lightgreen; color: black")
    process_btn.clicked.connect(lambda: plot_complexity_vs_score(main_window))
    layout_tab4.addWidget(process_btn)

    # Table for models
    main_window.dic_table = {}
    main_window._next_model_id = 1  # ever-incrementing, never reused/renumbered (see add_db_files_to_table)
    main_window.db_table = QTableWidget()
    main_window.db_table.setColumnCount(5)
    main_window.db_table.setHorizontalHeaderLabels(["Model", "Database", "Number of Parameters", "Best Score", "Remove"])
    header = main_window.db_table.horizontalHeader()
    for col in range(main_window.db_table.columnCount()):
        header.setSectionResizeMode(col, QHeaderView.Stretch)
    layout_tab4.addWidget(main_window.db_table)
     # Set gray background for header
    header = main_window.db_table.horizontalHeader()
    header.setStyleSheet("QHeaderView::section { background-color: lightgray; color: black; font-weight: bold; }")

    # Matplotlib Figure for Comparison plot
    plot_label = QLabel("<b>Model Comparison - Complexity vs. Score</b>")
    plot_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(plot_label)
    main_window.comparison_figure = Figure(figsize=(5,3), layout="tight")
    main_window.comparison_canvas = FigureCanvas(main_window.comparison_figure)
    layout_tab4.addWidget(main_window.comparison_canvas)
    return layout_tab4