import numpy as np
import os

# All the specific classes we need
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# My local modules
from uq_physicell.database.bo_db import load_structure


def calculate_aic(log_fitness, num_params):
    # Akaike Information Criterion: AIC = 2k - 2ln(fitness)
    return 2 * num_params - 2 * log_fitness


def add_db_files_to_table(main_window):
    files, _ = QFileDialog.getOpenFileNames(main_window, "Select Calibration .db Files", "", "Database Files (*.db)")
    if not files:
        return
    for f in files:
        row_position = main_window.db_table.rowCount()
        main_window.db_table.insertRow(row_position)
        # Model number (auto-increment)
        main_window.db_table.setItem(row_position, 0, QTableWidgetItem(str(row_position + 1)))
        # Database name
        main_window.db_table.setItem(row_position, 1, QTableWidgetItem(os.path.basename(f)))
        # Number of Parameters (empty, to be filled later)
        main_window.db_table.setItem(row_position, 2, QTableWidgetItem(""))
        # Log Likelihood (empty, to be filled later)
        main_window.db_table.setItem(row_position, 3, QTableWidgetItem(""))
        # Store full path for later use
        main_window.db_table.setItem(row_position, 4, QTableWidgetItem(f))
        # Add remove button
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda _, r=row_position: remove_db_row(main_window, r))
        main_window.db_table.setCellWidget(row_position, 5, remove_btn)

def remove_db_row(main_window, row):
    main_window.db_table.removeRow(row)
    # Re-number Model column
    for i in range(main_window.db_table.rowCount()):
        main_window.db_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))

def process_db_files_and_plot(main_window):
    if main_window.db_table.rowCount() == 0:
        QMessageBox.warning(main_window, "No DB Files", "Please load at least one calibration .db file.")
        return
    dic_aic_results = {}
    for idx in range(main_window.db_table.rowCount()):
        db_file = main_window.db_table.item(idx, 4).text()
        try:
            df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(db_file)
            num_params = df_param_space.shape[0]
            max_log_fitness = np.log(np.max(df_gp_models['Hypervolume'])) # This is a heuristic AIC-style criterion, not a formal information criterion.
            aic = calculate_aic(max_log_fitness, num_params)
            # Fill table
            main_window.db_table.setItem(idx, 2, QTableWidgetItem(str(num_params)))
            main_window.db_table.setItem(idx, 3, QTableWidgetItem(str(max_log_fitness)))
            # Fill dictionary
            dic_aic_results[f"Model {main_window.db_table.item(idx, 0).text()}"] = aic
        except Exception as e:
            QMessageBox.critical(main_window, "Error", f"Error processing {os.path.basename(db_file)}: {e}")
            return
    # Plot AIC results
    main_window.aic_figure.clear()
    ax = main_window.aic_figure.add_subplot(111)
    min_aic = min(dic_aic_results.values())
    relative_likelihoods = {k: np.exp((min_aic - v) / 2) for k, v in dic_aic_results.items()}
    ax.bar(relative_likelihoods.keys(), relative_likelihoods.values(), color='skyblue')
    ax.set_ylabel('Relative likelihood (AIC)')
    ax.set_title('Model Comparison by AIC')
    main_window.aic_canvas.draw()


def create_tab4(main_window):
    layout_tab4 = QVBoxLayout()
    load_dbs_label = QLabel("<b>Load Multiple Calibration .db Files</b>")
    load_dbs_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(load_dbs_label)

    # Load button
    load_db_btn = QPushButton("Add Calibration .db Files")
    load_db_btn.setStyleSheet("background-color: lightgreen; color: black")
    load_db_btn.clicked.connect(lambda: add_db_files_to_table(main_window))
    layout_tab4.addWidget(load_db_btn)

    # Process and plot button
    process_btn = QPushButton("Calculate Akaike Criterion and Plot")
    process_btn.setStyleSheet("background-color: lightgreen; color: black")
    process_btn.clicked.connect(lambda: process_db_files_and_plot(main_window))
    layout_tab4.addWidget(process_btn)

    # Table for models
    main_window.db_table = QTableWidget()
    main_window.db_table.setColumnCount(6)
    main_window.db_table.setHorizontalHeaderLabels(["Model", "Database", "Number of Parameters", "Log Fitness", "DB Path (hidden)", "Remove"])
    main_window.db_table.setColumnHidden(4, True)  # Hide DB path column
    header = main_window.db_table.horizontalHeader()
    for col in range(main_window.db_table.columnCount()):
        header.setSectionResizeMode(col, QHeaderView.Stretch)
    layout_tab4.addWidget(main_window.db_table)
     # Set gray background for header
    header = main_window.db_table.horizontalHeader()
    header.setStyleSheet("QHeaderView::section { background-color: lightgray; color: black; font-weight: bold; }")

    # Matplotlib Figure for AIC plot
    plot_label = QLabel("<b>Information Criteria Plot</b>")
    plot_label.setAlignment(Qt.AlignCenter)
    layout_tab4.addWidget(plot_label)
    main_window.aic_figure = Figure(figsize=(5,3))
    main_window.aic_canvas = FigureCanvas(main_window.aic_figure)
    layout_tab4.addWidget(main_window.aic_canvas)
    return layout_tab4