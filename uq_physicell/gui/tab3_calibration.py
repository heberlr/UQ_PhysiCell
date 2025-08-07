from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QDialog, QFileDialog, QListWidget, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns

from uq_physicell.bo.database import load_structure
from uq_physicell.bo.plot import (
    analyze_pareto_results, get_observed_qoi, plot_parameter_space, plot_qoi_param, plot_parameter_vs_fitness
)

def create_tab3(main_window):
    # Add methods to the main_window instance
    main_window.load_obs_data = load_obs_data
    main_window.load_bo_database = load_bo_database
    main_window.match_qois = match_qois
    main_window.run_calibration = run_calibration
    main_window.plot_calibration_results = plot_calibration_results

    layout_tab3 = QVBoxLayout()

    ###########################################
    # Load observational data section
    ###########################################
    main_window.load_obs_label = QLabel("<b>Load Files</b>")
    main_window.load_obs_label.setAlignment(Qt.AlignCenter)
    layout_tab3.addWidget(main_window.load_obs_label)
    # Horizontal layout for loading files
    load_obs_hbox = QHBoxLayout()
    # Model structure button and input
    load_obs_hbox.addWidget(QLabel("Model structure:"))
    main_window.load_ini_button_tab3 = QPushButton("Load .ini File")
    main_window.load_ini_button_tab3.setStyleSheet("background-color: lightgreen; color: black")
    main_window.load_ini_button_tab3.clicked.connect(lambda: main_window.load_ini_file(main_window))
    load_obs_hbox.addWidget(main_window.load_ini_button_tab3)
    main_window.ini_file_name_input = QLineEdit()
    main_window.ini_file_name_input.setPlaceholderText("Model structure file path")
    main_window.ini_file_name_input.setEnabled(False)
    load_obs_hbox.addWidget(main_window.ini_file_name_input)
    # Add a horizontal spacer (scape)
    load_obs_hbox.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Minimum))
    # Observational data button and input
    load_obs_hbox.addWidget(QLabel("Obs. data:"))
    main_window.load_obs_button = QPushButton("Load Observational Data (.csv)")
    main_window.load_obs_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.load_obs_button.clicked.connect(lambda: main_window.load_obs_data(main_window))
    load_obs_hbox.addWidget(main_window.load_obs_button)
    main_window.obs_file_name_input = QLineEdit()
    main_window.obs_file_name_input.setPlaceholderText("Observational data file path")
    main_window.obs_file_name_input.setEnabled(False)
    load_obs_hbox.addWidget(main_window.obs_file_name_input)
    # Add the horizontal layout to the main layout
    layout_tab3.addLayout(load_obs_hbox)

    ###########################################
    # Setup the calibration problem
    ###########################################
    main_window.qoi_obs_label = QLabel("<b>Setting up the Calibration problem</b>")
    main_window.qoi_obs_label.setAlignment(Qt.AlignCenter)
    layout_tab3.addWidget(main_window.qoi_obs_label)
    #Vertical layout for setuping the calibration problem
    vbox_qoi_tab3 = QVBoxLayout()
    # Horizontal layout for defining QoIs
    hbox_inout_tab3 = QHBoxLayout()
    # Define parameters
    hbox_inout_tab3.addWidget(QLabel("Define Parameter(s):"))
    main_window.define_param_button_tab3 = QPushButton("Model Parameter(s)")
    main_window.define_param_button_tab3.setStyleSheet("background-color: lightgreen; color: black")
    main_window.define_param_button_tab3.setEnabled(False)
    main_window.define_param_button_tab3.clicked.connect(lambda: open_param_definition_window(main_window))
    hbox_inout_tab3.addWidget(main_window.define_param_button_tab3)
    # Add a horizontal spacer (scape)
    hbox_inout_tab3.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))
    # Define QoIs
    hbox_inout_tab3.addWidget(QLabel("Define QoI(s):"))
    main_window.define_qoi_button_tab3 = QPushButton("Model QoI(s)")
    main_window.define_qoi_button_tab3.setEnabled(False)
    main_window.define_qoi_button_tab3.setStyleSheet("background-color: lightgreen; color: black")
    main_window.define_qoi_button_tab3.clicked.connect(lambda: open_qoi_definition_window(main_window))
    hbox_inout_tab3.addWidget(main_window.define_qoi_button_tab3)
    # Add a horizontal spacer (scape)
    hbox_inout_tab3.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))
    # Map QoIs to observational data
    hbox_inout_tab3.addWidget(QLabel("Map Model QoI(s) to Observational Data:"))
    main_window.map_qois_button = QPushButton("Map QoI(s)")
    main_window.map_qois_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.map_qois_button.setEnabled(False)
    main_window.map_qois_button.clicked.connect(lambda: main_window.match_qois(main_window))
    hbox_inout_tab3.addWidget(main_window.map_qois_button)
    # Add horizondal layout to the vertical layout
    vbox_qoi_tab3.addLayout(hbox_inout_tab3)
    
    # Select calibration method
    hbox_calib_tab3 = QHBoxLayout()
    hbox_calib_tab3.addWidget(QLabel("Method:"))
    # Combobox for calibration method
    main_window.calibration_method_combo = QComboBox()
    main_window.calibration_method_combo.addItems(["Bayesian Optimization", "Bayesian Inference"])
    main_window.calibration_method_combo.setEnabled(False)
    hbox_calib_tab3.addWidget(main_window.calibration_method_combo)
    # Select distance metric - Add weights later
    hbox_calib_tab3.addWidget(QLabel("Distance Function:"))
    # Combobox for distance metric
    main_window.distance_metric_combo = QComboBox()
    main_window.distance_metric_combo.addItems(["Euclidean", "Manhattan", "Cosine", "Minkowski"])
    main_window.distance_metric_combo.setEnabled(False)
    hbox_calib_tab3.addWidget(main_window.distance_metric_combo)
    # File name for calibration results
    hbox_calib_tab3.addWidget(QLabel("Calibration Results File:"))
    # LineEdit for file name
    main_window.calibration_file_input = QLineEdit()
    main_window.calibration_file_input.setPlaceholderText("Enter file name to store calibration results")
    main_window.calibration_file_input.setEnabled(False)
    hbox_calib_tab3.addWidget(main_window.calibration_file_input)
    # Add horizondal layout to the vertical layout
    vbox_qoi_tab3.addLayout(hbox_calib_tab3)
    
    # Add the vertical layout to the main layout
    layout_tab3.addLayout(vbox_qoi_tab3)

    ###########################################
    # Run and plot calibration results
    ###########################################
    calibration_buttons_hbox = QHBoxLayout()

    main_window.run_calibration_button = QPushButton("Run Calibration")
    main_window.run_calibration_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.run_calibration_button.setEnabled(False)
    main_window.run_calibration_button.clicked.connect(lambda: main_window.run_calibration(main_window))
    calibration_buttons_hbox.addWidget(main_window.run_calibration_button)

    main_window.plot_calibration_button = QPushButton("Plot Results")
    main_window.plot_calibration_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_calibration_button.setEnabled(False)
    main_window.plot_calibration_button.clicked.connect(lambda: main_window.plot_calibration_results(main_window))
    calibration_buttons_hbox.addWidget(main_window.plot_calibration_button)

    layout_tab3.addLayout(calibration_buttons_hbox)

    ###########################################
    # Output section
    ###########################################
    main_window.output_label_tab3 = QLabel("<b>Output</b>")
    main_window.output_label_tab3.setAlignment(Qt.AlignCenter)
    layout_tab3.addWidget(main_window.output_label_tab3)

    main_window.output_text_tab3 = QTextEdit()
    main_window.output_text_tab3.setReadOnly(True)
    main_window.output_text_tab3.setMinimumHeight(100)
    layout_tab3.addWidget(main_window.output_text_tab3)

    return layout_tab3


def load_obs_data(main_window):
    # Load observational data from a .csv file
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(main_window, "Select Observational Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_path:
        try:
            main_window.obs_data = pd.read_csv(file_path)
            main_window.obs_file_name_input.setText(file_path)
            main_window.define_qoi_button_tab3.setEnabled(True)
            main_window.map_qois_button.setEnabled(True)
            main_window.distance_metric_combo.setEnabled(True)
            main_window.calibration_method_combo.setEnabled(True)
            main_window.calibration_file_input.setEnabled(True)
            main_window.run_calibration_button.setEnabled(True)
            main_window.plot_calibration_button.setEnabled(True)
            main_window.output_text_tab3.append(f"Loaded observational data from {file_path}")
        except Exception as e:
            main_window.output_text_tab3.append(f"Error loading observational data: {e}")

def load_bo_database(main_window):
    # Missing implementation for loading Bayesian Optimization database
    print("Loading Bayesian Optimization database is not implemented yet.")
    try:
        # Load the database structure
        main_window.output_text_tab3.append(f"Loading Bayesian Optimization database {main_window.bo_file_path} ...")
        df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output = load_structure(main_window.bo_file_path)

        # Load the .ini file if it exists
        main_window.load_ini_file(main_window, df_metadata['Ini_File_Path'].values[0], df_metadata['StructureName'].values[0])
        print(df_metadata)
        # Set the widgets accordingly
        main_window.ini_file_name_input.setText(df_metadata['Ini_File_Path'].values[0])
        main_window.ini_file_name_input.setEnabled(False)
        main_window.obs_file_name_input.setText(df_metadata['ObsData_Path'].values[0])
        main_window.obs_file_name_input.setEnabled(False)
        main_window.calibration_file_input.setText(main_window.bo_file_path)  # Store the path of the
        main_window.calibration_file_input.setEnabled(False)
        # Activate plot results button
        main_window.plot_calibration_button.setEnabled(True)

        # Save the loaded data to the main_window
        main_window.df_obs_data = get_observed_qoi(df_metadata['ObsData_Path'].values[0], df_qois)
        main_window.df_metadata = df_metadata
        main_window.df_param_space = df_param_space
        main_window.df_qois = df_qois
        main_window.df_gp_models = df_gp_models
        main_window.df_samples = df_samples
        main_window.df_output = df_output

    except Exception as e:
        main_window.output_text_tab3.append(f"Error loading Bayesian Optimization database: {e}")

def match_qois(main_window):
    # Define QoIs and correlate them with observational data
    qoi_window = QDialog(main_window)
    qoi_window.setWindowTitle("Define QoIs and Correlate with Observational Data")
    qoi_window.setMinimumSize(600, 500)

    layout = QVBoxLayout()

    # Instructions
    instructions_label = QLabel("Define QoIs and correlate them with observational data columns:")
    layout.addWidget(instructions_label)

    # QoI and observational data correlation
    qoi_obs_hbox = QHBoxLayout()
    qoi_label = QLabel("Select QoI:")
    qoi_obs_hbox.addWidget(qoi_label)
    qoi_combo = QComboBox()
    qoi_combo.addItems(main_window.qoi_funcs.keys())
    qoi_obs_hbox.addWidget(qoi_combo)

    obs_label = QLabel("Select Observational Data Column:")
    qoi_obs_hbox.addWidget(obs_label)
    obs_combo = QComboBox()
    obs_combo.addItems(main_window.obs_data.columns)
    qoi_obs_hbox.addWidget(obs_combo)

    layout.addLayout(qoi_obs_hbox)

    # Add correlation button
    correlate_button = QPushButton("Correlate QoI with Observational Data")
    correlate_button.setStyleSheet("background-color: lightgreen; color: black")
    layout.addWidget(correlate_button)

    # Selected correlations
    selected_correlations_label = QLabel("<b>Selected Correlations</b>")
    selected_correlations_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(selected_correlations_label)

    selected_correlations_list = QListWidget()
    layout.addWidget(selected_correlations_list)

    # Add correlation logic
    def add_correlation():
        qoi = qoi_combo.currentText()
        obs_col = obs_combo.currentText()
        selected_correlations_list.addItem(f"{qoi} -> {obs_col}")

    correlate_button.clicked.connect(add_correlation)

    # Close button
    close_button = QPushButton("Close")
    close_button.setStyleSheet("background-color: lightgreen; color: black")
    close_button.clicked.connect(qoi_window.accept)
    layout.addWidget(close_button)

    qoi_window.setLayout(layout)
    qoi_window.exec_()


def run_calibration(main_window):
    # Placeholder for running calibration
    main_window.output_text_tab3.append("Running calibration... (this is a placeholder)")


def plot_calibration_results(main_window):
    main_window.output_text_tab3.append("Plot Bayesian Optimization Results...")
    # Create a dialog window for plotting QoIs
    plot_bo_window = QDialog(main_window)
    plot_bo_window.setWindowTitle("Bayesian Optimization Results")
    plot_bo_window.setGeometry(100, 100, 800, 600)
    # Create a layout for the dialog
    layout = QVBoxLayout(plot_bo_window)
    # Add plot mode combo box
    plot_mode_hbox = QHBoxLayout()
    plot_mode_label = QLabel("Select Plot Mode:")
    plot_mode_combo = QComboBox()
    plot_mode_combo.addItems(["Parameter Space", "Parameter vs Fitness", "QoI"])
    plot_mode_hbox.addWidget(plot_mode_label)
    plot_mode_hbox.addWidget(plot_mode_combo)
    layout.addLayout(plot_mode_hbox)
    # Add input for parameter to plot
    plot_param_hbox = QHBoxLayout()
    plot_param_label = QLabel("Select Parameter to plot:")
    plot_param_combo = QComboBox()
    plot_param_combo.addItems(main_window.df_param_space['ParamName'].tolist())
    plot_param_hbox.addWidget(plot_param_label)
    plot_param_hbox.addWidget(plot_param_combo)
    layout.addLayout(plot_param_hbox)
    # Add input for the QoI to plot
    plot_qoi_hbox = QHBoxLayout()
    plot_qoi_label = QLabel("Select QoI to plot:")
    plot_qoi_combo = QComboBox()
    plot_qoi_combo.addItems(list(main_window.df_qois['QoI_Name'].tolist()))
    plot_qoi_hbox.addWidget(plot_qoi_label)
    plot_qoi_hbox.addWidget(plot_qoi_combo)
    layout.addLayout(plot_qoi_hbox)
    
    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(5, 3))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)

    def update_plot_bo():
        # Clear the previous plot
        figure.clear()
        selected_mode = plot_mode_combo.currentText()
        # Comprehensive Pareto analysis
        pareto_data = analyze_pareto_results(main_window.df_qois, main_window.df_samples, main_window.df_output)
        try:
            ax = figure.add_subplot(111)
            if selected_mode == "Parameter Space":
                plot_param_combo.setEnabled(False)
                plot_qoi_combo.setEnabled(False)
                pareto_points = {f"Pareto Point {i}": param for i, param in enumerate(pareto_data['pareto_front']['parameters'])}
                plot_parameter_space(main_window.df_samples, main_window.df_param_space, params=pareto_points, axis=ax)
            elif selected_mode == "Parameter vs Fitness":
                plot_param_combo.setEnabled(True)
                plot_qoi_combo.setEnabled(True)
                selected_param = plot_param_combo.currentText()
                selected_qoi = plot_qoi_combo.currentText()
                plot_parameter_vs_fitness(main_window.df_samples, main_window.df_output, 
                                            parameter_name=selected_param, qoi_name=selected_qoi, 
                                            samples_id=pareto_data['pareto_front']['sample_ids'], axis=ax)
            elif selected_mode == "QoI":
                plot_param_combo.setEnabled(False)
                plot_qoi_combo.setEnabled(True)
                selected_qoi = plot_qoi_combo.currentText()
                plot_qoi_param(main_window.df_obs_data, main_window.df_output, 
                               samples_id=pareto_data['pareto_front']['sample_ids'], x_var='time', y_var=selected_qoi, axis=ax)
            # Set the layout to be constrained
            figure.set_constrained_layout(True)
            canvas.draw()
        except Exception as e:
            main_window.output_text_tab3.append(f"Error updating plot: {e}")
            # Create a simple error plot
            try:
                ax = figure.add_subplot(111)
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
                canvas.draw()
            except:
                pass

    # Connect the combo box to update the plot
    plot_mode_combo.currentIndexChanged.connect(update_plot_bo)
    plot_param_combo.currentIndexChanged.connect(update_plot_bo)
    plot_qoi_combo.currentIndexChanged.connect(update_plot_bo)
    # Set the default selected qoi and update the plot
    plot_mode_combo.setCurrentIndex(0)
    update_plot_bo()
    # Show the dialog
    plot_bo_window.exec_()


    