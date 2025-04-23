import os, sys
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QTextEdit, QDialog, QFileDialog, QInputDialog, QListWidget, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# SA library
from SALib.sample import fast_sampler, ff, finite_diff, latin, sobol, morris
from SALib.analyze import fast as fast_analyze, rbd_fast as rbd_fast_analyze, ff as ff_analyze, pawn as pawn_analyze, dgsm as dgsm_analyze, hdmr as hdmr_analyze, rsa as rsa_analyze, discrepancy as discrepancy_analyze, delta as delta_analyze, sobol as sobol_analyze, morris as morris_analyze

# My local modules
from uq_physicell.SA_script import run_sa_simulations
from uq_physicell.SA_utils import load_db_structure, OAT_analyze, extract_qoi_from_db

def create_tab2(main_window):
    # Add the following methods to the main_window instance
    main_window.update_output_tab2 = update_output_tab2
    main_window.load_db_file = load_db_file
    main_window.update_analysis_type = update_analysis_type
    main_window.update_local_inputs = update_local_inputs
    main_window.update_local_SA_reference = update_local_SA_reference
    main_window.update_local_SA_perturbations = update_local_SA_perturbations
    main_window.update_global_inputs = update_global_inputs
    main_window.update_global_SA_reference = update_global_SA_reference
    main_window.update_global_SA_range_percentage = update_global_SA_range_percentage
    main_window.sample_parameters = sample_parameters
    main_window.plot_samples = plot_samples
    main_window.update_global_sampler_options = update_global_sampler_options
    main_window.run_simulations = run_simulations
    main_window.run_analysis = run_analysis
    main_window.plot_sa_results = plot_sa_results

    layout_tab2 = QVBoxLayout()

    ###########################################
    # Buttons to load .ini file and .db file
    ###########################################
    main_window.load_ini_label = QLabel("<b>Load .ini and .db Files</b>")
    main_window.load_ini_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.load_ini_label)
    load_button_hbox = QHBoxLayout()
    main_window.load_ini_button = QPushButton("Load .ini File")
    main_window.load_ini_button.clicked.connect(lambda: main_window.load_ini_file(main_window))
    load_button_hbox.addWidget(main_window.load_ini_button)
    main_window.load_db_button = QPushButton("Load .db File")
    main_window.load_db_button.clicked.connect(lambda: main_window.load_db_file(main_window))
    load_button_hbox.addWidget(main_window.load_db_button)
    layout_tab2.addLayout(load_button_hbox)

    # Separator line
    layout_tab2.addWidget(QLabel("<hr>"))

    ###########################################
    # Dropdown for sensitivity analysis type
    ###########################################
    main_window.analysis_sample_label = QLabel("<b>Sensitivity Analysis Type and Parameter Sampling</b>")
    main_window.analysis_sample_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.analysis_sample_label)
    main_window.analysis_type_hbox = QHBoxLayout()
    main_window.analysis_type_label = QLabel("Select Sensitivity Analysis Type:")
    main_window.analysis_type_hbox.addWidget(main_window.analysis_type_label)
    main_window.analysis_type_dropdown = QComboBox()
    main_window.analysis_type_dropdown.addItems(["Local", "Global"])
    main_window.analysis_type_dropdown.setEnabled(False)
    main_window.analysis_type_dropdown.currentIndexChanged.connect(lambda: main_window.update_analysis_type(main_window))
    main_window.analysis_type_hbox.addWidget(main_window.analysis_type_dropdown)
    layout_tab2.addLayout(main_window.analysis_type_hbox)

    ###########################################
    # Hbox for LOCAL sentivity analysis method
    ###########################################
    main_window.localSA_param_hbox = QHBoxLayout()
    # Select parameter for local SA
    main_window.local_param_label = QLabel("Select Parameter:")
    main_window.local_param_label.setVisible(False)
    layout_tab2.addWidget(main_window.local_param_label)
    main_window.local_param_combo = QComboBox()
    main_window.local_param_combo.setVisible(False)
    main_window.localSA_param_hbox.addWidget(main_window.local_param_combo)
    # Reference value input
    main_window.local_ref_value_label = QLabel("Reference Value:")
    main_window.local_ref_value_label.setVisible(False)
    main_window.localSA_param_hbox.addWidget(main_window.local_ref_value_label)
    main_window.local_ref_value_input = QLineEdit()
    main_window.local_ref_value_input.setValidator(QDoubleValidator())
    main_window.local_ref_value_input.setVisible(False)
    main_window.localSA_param_hbox.addWidget(main_window.local_ref_value_input)
    # Percentage perturbations input
    main_window.local_perturb_label = QLabel("Percentage Perturbation(s) (+/-):")
    main_window.local_perturb_label.setVisible(False)
    main_window.localSA_param_hbox.addWidget(main_window.local_perturb_label)
    main_window.local_perturb_input = QLineEdit()
    main_window.local_perturb_input.setVisible(False)
    main_window.localSA_param_hbox.addWidget(main_window.local_perturb_input)
    layout_tab2.addLayout(main_window.localSA_param_hbox)

    ###########################################
    # Vbox for GLOBAL sentivity analysis method
    ###########################################
    main_window.globalSA_vbox = QVBoxLayout()
    main_window.globalSA_method_hbox = QHBoxLayout()
    # Global SA method options
    main_window.global_method_label = QLabel("Method:")
    main_window.global_method_label.setVisible(False)
    main_window.globalSA_method_hbox.addWidget(main_window.global_method_label)
    main_window.global_method_combo = QComboBox()
    main_window.global_method_combo.addItems([
        "FAST - Fourier Amplitude Sensitivity Test",
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Method of Morris",
        "Sobol Sensitivity Analysis",
        "Delta Moment-Independent Measure",
        "Derivative-based Global Sensitivity Measure (DGSM)",
        "Fractional Factorial",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ])
    main_window.global_method_combo.setVisible(False)
    main_window.global_method_combo.currentIndexChanged.connect(lambda: main_window.update_global_sampler_options(main_window))
    main_window.globalSA_method_hbox.addWidget(main_window.global_method_combo)
    # Sampler options
    main_window.global_sampler_label = QLabel("Sampler:")
    main_window.global_sampler_label.setVisible(False)
    main_window.globalSA_method_hbox.addWidget(main_window.global_sampler_label)
    main_window.global_sampler_combo = QComboBox()
    main_window.global_sampler_combo.setVisible(False)
    main_window.globalSA_method_hbox.addWidget(main_window.global_sampler_combo)
    main_window.globalSA_vbox.addLayout(main_window.globalSA_method_hbox)
    # Hbox of parameters
    main_window.globalSA_parameters_hbox = QHBoxLayout()
    # Select parameter for global SA
    main_window.global_param_label = QLabel("Select Parameter:")
    main_window.global_param_label.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_param_label)
    main_window.global_param_combo = QComboBox()
    main_window.global_param_combo.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_param_combo)
    # Reference value input
    main_window.global_ref_value_label = QLabel("Ref. Value:")
    main_window.global_ref_value_label.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_ref_value_label)
    main_window.global_ref_value_input = QLineEdit()
    main_window.global_ref_value_input.setValidator(QDoubleValidator())
    main_window.global_ref_value_input.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_ref_value_input)
    # Percentage of range of parameters input
    main_window.global_range_percentage_label = QLabel("Range (%):")
    main_window.global_range_percentage_label.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_range_percentage_label)
    main_window.global_range_percentage = QLineEdit()
    main_window.global_range_percentage.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_range_percentage)
    # bounds of the range - (min, max) float
    main_window.global_bounds_label = QLabel("Bounds (min, max):")
    main_window.global_bounds_label.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_bounds_label)
    main_window.global_bounds = QLineEdit()
    main_window.global_bounds.setReadOnly(True)
    main_window.global_bounds.setVisible(False)
    main_window.globalSA_parameters_hbox.addWidget(main_window.global_bounds)
    main_window.globalSA_vbox.addLayout(main_window.globalSA_parameters_hbox)
    layout_tab2.addLayout(main_window.globalSA_vbox)

    ###########################################
    # Horizontal layout for SA buttons
    ###########################################
    buttonSA_hbox = QHBoxLayout()
    # Sample paramaters button
    main_window.sample_params_button = QPushButton("Sample Parameters")
    main_window.sample_params_button.clicked.connect(lambda: main_window.sample_parameters(main_window))
    main_window.sample_params_button.setEnabled(False)
    buttonSA_hbox.addWidget(main_window.sample_params_button)
    # Plot samples button
    main_window.plot_samples_button = QPushButton("Plot Samples")
    main_window.plot_samples_button.clicked.connect(lambda: main_window.plot_samples(main_window))
    buttonSA_hbox.addWidget(main_window.plot_samples_button)
    main_window.plot_samples_button.setEnabled(False)
    layout_tab2.addLayout(buttonSA_hbox)
    # Figure of samples
    main_window.fig_est_canvas_samples = FigureCanvas(Figure(figsize=(5, 3)))
    main_window.ax_samples = main_window.fig_est_canvas_samples.figure.add_subplot(111)
    main_window.fig_est_canvas_samples.setVisible(False)
    layout_tab2.addWidget(main_window.fig_est_canvas_samples)

    ###########################################
    # Horizontal layout for db file name, QoI selection, and run SA button
    ###########################################
    main_window.analysis_type_label = QLabel("<b>Database, QoI, Simulations, and Sensitivity Analysis</b>")
    main_window.analysis_type_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.analysis_type_label)
    main_window.db_file_name_hbox = QHBoxLayout()
    # DB file name input
    main_window.db_file_name_label = QLabel("DB File Name:")
    main_window.db_file_name_hbox.addWidget(main_window.db_file_name_label)
    main_window.db_file_name_input = QLineEdit()
    main_window.db_file_name_input.setPlaceholderText("Enter DB file name")
    main_window.db_file_name_input.setEnabled(False)
    main_window.db_file_name_hbox.addWidget(main_window.db_file_name_input)
    # Button to open QoI definition window
    main_window.define_qoi_button = QPushButton("Define QoI")
    main_window.define_qoi_button.setEnabled(False)
    main_window.define_qoi_button.clicked.connect(lambda: open_qoi_definition_window(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.define_qoi_button)
    # Run simulations button
    main_window.run_simulations_button = QPushButton("Run Simulations")
    main_window.run_simulations_button.setEnabled(False)
    main_window.run_simulations_button.clicked.connect(lambda: main_window.run_simulations(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.run_simulations_button)
    # Run SA button
    main_window.run_sa_button = QPushButton("Run SA")
    main_window.run_sa_button.setEnabled(False)
    main_window.run_sa_button.clicked.connect(lambda: main_window.run_analysis(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.run_sa_button)
    layout_tab2.addLayout(main_window.db_file_name_hbox)

    ###########################################
    # Current QoI label
    ###########################################
    main_window.current_qoi_label = QLabel("Current QoI(s): None")
    main_window.current_qoi_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.current_qoi_label)

    # Separator line
    layout_tab2.addWidget(QLabel("<hr>"))

    ###########################################
    # Output section - display information
    ###########################################
    main_window.output_label_tab2 = QLabel("<b>Output</b>")
    main_window.output_label_tab2.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.output_label_tab2)
    main_window.output_text_tab2 = QTextEdit()
    main_window.output_text_tab2.setReadOnly(True)
    main_window.output_text_tab2.setMinimumHeight(100)
    layout_tab2.addWidget(main_window.output_text_tab2)

    return layout_tab2

def open_qoi_definition_window(main_window):
    """
    Opens a new window for defining QoIs.
    """
    # Create a new dialog window
    qoi_window = QDialog(main_window)  # Use main_window as the parent
    qoi_window.setWindowTitle("Define QoI")
    qoi_window.setMinimumSize(600, 500)

    # Layout for the QoI window
    layout = QVBoxLayout()

    # Instructions label
    instructions_label = QLabel("Define your QoIs below:")
    layout.addWidget(instructions_label)

    # Predefined QoIs dictionary
    predefined_qoi_funcs = {
        'total_cells': "lambda df: len(df)",
        'live_cells': "lambda df: len(df[df['dead'] == False])",
        'dead_cells': "lambda df: len(df[df['dead'] == True])",
        'max_volume': "lambda df: df['total_volume'].max()",
        'min_volume': "lambda df: df['total_volume'].min()",
        'mean_volume': "lambda df: df['total_volume'].mean()",
        'std_volume': "lambda df: df['total_volume'].std()",
        'total_volume': "lambda df: df['total_volume'].sum()",
        'template_cellType_live': "lambda df: len( df[ (df['dead'] == False) & (df['cell_type'] == <cellType>) ])",
    }

    # Predefined QoIs section
    predefined_qoi_label = QLabel("<b>Select Predefined QoIs</b>")
    predefined_qoi_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(predefined_qoi_label)

    predefined_qoi_combo = QComboBox()
    predefined_qoi_combo.addItems(predefined_qoi_funcs.keys())
    layout.addWidget(predefined_qoi_combo)

    predefined_qoi_lambda_label = QLabel("Lambda Function:")
    layout.addWidget(predefined_qoi_lambda_label)

    predefined_qoi_lambda_display = QTextEdit()
    predefined_qoi_lambda_display.setReadOnly(True)
    predefined_qoi_lambda_display.setText(predefined_qoi_funcs['live_cells'])  # Default display
    layout.addWidget(predefined_qoi_lambda_display)

    # Update lambda display when a predefined QoI is selected
    predefined_qoi_combo.currentIndexChanged.connect(
        lambda: predefined_qoi_lambda_display.setText(predefined_qoi_funcs[predefined_qoi_combo.currentText()])
    )

    # Add predefined QoI button
    add_predefined_qoi_button = QPushButton("Add Selected Predefined QoI")
    layout.addWidget(add_predefined_qoi_button)

    # Separator line
    layout.addWidget(QLabel("<hr>"))

    # Custom QoI section
    custom_qoi_label = QLabel("<b>Define Custom QoIs</b>")
    custom_qoi_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(custom_qoi_label)

    # Input field for custom QoI name
    custom_qoi_name_input = QLineEdit()
    custom_qoi_name_input.setPlaceholderText("Enter QoI name")
    layout.addWidget(custom_qoi_name_input)

    # Input field for custom lambda function
    custom_lambda_input = QLineEdit()
    custom_lambda_input.setPlaceholderText("Enter lambda function (e.g., lambda df: len(df[df['dead'] == False]))")
    layout.addWidget(custom_lambda_input)

    # Add custom QoI button
    add_custom_qoi_button = QPushButton("Add Custom QoI")
    layout.addWidget(add_custom_qoi_button)

    # Separator line
    layout.addWidget(QLabel("<hr>"))

    # Selected QoIs section
    selected_qois_label = QLabel("<b>Selected QoIs</b>")
    selected_qois_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(selected_qois_label)

    selected_qois_list = QListWidget()
    layout.addWidget(selected_qois_list)

    # Initialize the selected QoIs list based on main_window.qoi_funcs
    if main_window.qoi_funcs:
        for qoi_name, qoi_lambda in main_window.qoi_funcs.items():
            selected_qois_list.addItem(f"{qoi_name}: {qoi_lambda}")

    # Add predefined QoI to the dictionary and list
    def add_predefined_qoi():
        qoi_name = predefined_qoi_combo.currentText()
        qoi_lambda = predefined_qoi_funcs[qoi_name]
        if qoi_name not in main_window.qoi_funcs:
            main_window.qoi_funcs[qoi_name] = qoi_lambda
            selected_qois_list.addItem(f"{qoi_name}: {qoi_lambda}")

    add_predefined_qoi_button.clicked.connect(add_predefined_qoi)

    # Add custom QoI to the dictionary and list
    def add_custom_qoi():
        qoi_name = custom_qoi_name_input.text()
        qoi_lambda = custom_lambda_input.text()
        if qoi_name and qoi_lambda and qoi_name not in main_window.qoi_funcs:
            main_window.qoi_funcs[qoi_name] = qoi_lambda
            selected_qois_list.addItem(f"{qoi_name}: {qoi_lambda}")
            custom_qoi_name_input.clear()
            custom_lambda_input.clear()

    add_custom_qoi_button.clicked.connect(add_custom_qoi)

    # Button to remove selected QoI
    remove_qoi_button = QPushButton("Remove Selected QoI")
    layout.addWidget(remove_qoi_button)

    def remove_selected_qoi():
        selected_items = selected_qois_list.selectedItems()
        for item in selected_items:
            qoi_name = item.text().split(":")[0].strip()
            if qoi_name in main_window.qoi_funcs:
                del main_window.qoi_funcs[qoi_name]
            selected_qois_list.takeItem(selected_qois_list.row(item))

    remove_qoi_button.clicked.connect(remove_selected_qoi)

    # Close button
    close_button = QPushButton("Close")
    close_button.clicked.connect(qoi_window.accept)
    layout.addWidget(close_button)
    # Set the layout and show the dialog
    qoi_window.setLayout(layout)
    qoi_window.exec_()

    # Update the main window with the selected QoIs
    main_window.current_qoi_label.setText("Current QoI(s): " + ", ".join(main_window.qoi_funcs.keys()) if main_window.qoi_funcs else "Current QoI(s): None")

def update_output_tab2(main_window, message):
    # Update the output section in Tab 2 with a new message
    main_window.output_text_tab2.append(message)

def load_db_file(main_window, filePath=None):
        # Load the database file and extract parameters for sensitivity analysis
        if filePath is None:
            options = QFileDialog.Options()
            main_window.db_file_path, _ = QFileDialog.getOpenFileName(main_window, "Select .db File", "", "DB Files (*.db);;All Files (*)", options=options)
        else:
            main_window.db_file_path = filePath
        if main_window.db_file_path:
            try:
                # Load the database structure
                main_window.update_output_tab2(main_window, f"Loading database file {main_window.db_file_path} ...")
                main_window.df_metadata, main_window.df_input, main_window.df_output = load_db_structure(main_window.db_file_path)
                # print(main_window.df_output)
                # Load the .ini file
                main_window.load_ini_file(main_window, main_window.df_metadata['Ini_File_Path'].iloc[0], main_window.df_metadata['StructureName'].iloc[0])
                # Define the widget to display db structure
                SA_type = main_window.df_metadata['SA_Type'].iloc[0]
                main_window.analysis_type_dropdown.setCurrentText(SA_type)
                main_window.update_analysis_type(main_window)
                main_window.analysis_type_dropdown.setEnabled(False)
                if SA_type == "Global": # Disable global fields
                    # Update the global method and sampler combo boxes
                    main_window.global_method_combo.setCurrentText(main_window.df_metadata['SA_Method'].iloc[0])
                    main_window.global_method_combo.setEnabled(False)
                    main_window.global_sampler_combo.setCurrentText(main_window.df_metadata['SA_Sampler'].iloc[0])
                    main_window.global_sampler_combo.setEnabled(False)
                    main_window.global_param_combo.setEnabled(False)
                    main_window.global_ref_value_input.setEnabled(False)
                    main_window.global_range_percentage.setEnabled(False)
                    main_window.global_bounds.setEnabled(False)
                    # Populate the global_SA_parameters dictionary with values from the database
                    main_window.global_SA_parameters = {}
                    # Bounds of parameters
                    bounds_str = main_window.df_metadata["Bounds"].iloc[0]
                    bounds_list = [
                        list(map(float, b.strip("[] ").split(',')))  # Remove brackets and split by comma
                        for b in bounds_str.split(",") if b.strip()  # Split by comma and ensure non-empty entries
                    ]
                    # Reference values and perturbations
                    param_ref_str = main_window.df_metadata["Reference_Values"].iloc[0]
                    param_ref_list = [float(r) for r in param_ref_str.split(',')]
                    param_perturb_str = main_window.df_metadata["Perturbations"].iloc[0]
                    param_perturb_list = [float(p) for p in param_perturb_str.split(',')]
                    # Create a dictionary for each parameter
                    for id, param in enumerate(main_window.df_input['ParamName'].unique()):
                        main_window.global_SA_parameters[param] = {"bounds": bounds_list[id], "reference": param_ref_list[id], "range_percentage": param_perturb_list[id]}
                    # Convert df_input to a NumPy array with shape (number of samples, number of parameters)
                    main_window.global_SA_parameters["samples"] = main_window.df_input.pivot(index="SampleID", columns="ParamName", values="ParamValue").to_dict(orient="index")
                elif SA_type == "Local": # Disable local fields
                    main_window.local_param_combo.setEnabled(False)
                    main_window.local_ref_value_input.setEnabled(False)
                    main_window.local_perturb_input.setEnabled(False)
                    # Populate the local_SA_parameters dictionary with values from the database
                    main_window.local_SA_parameters = {}
                    # Reference values and perturbations
                    param_ref_str = main_window.df_metadata["Reference_Values"].iloc[0]
                    param_ref_list = [float(r) for r in param_ref_str.split(',')]
                    param_pertub_str = main_window.df_metadata["Perturbations"].iloc[0]
                    param_pertub_list = [
                        list(map(float, b.strip("[] ").split(',')))  # Remove brackets, spaces, and split by comma
                        for b in param_pertub_str.split(",") if b.strip()  # Split by comma and ensure non-empty entries
                    ]
                    # Convert df_input to a NumPy array with shape (number of samples, number of parameters)
                    main_window.local_SA_parameters["samples"] = main_window.df_input.pivot(index="SampleID", columns="ParamName", values="ParamValue").to_dict(orient="index")
                    
                    # Create a dictionary for each parameter
                    for id, param in enumerate(main_window.df_input['ParamName'].unique()):
                        main_window.local_SA_parameters[param] = {"reference": param_ref_list[id], "perturbations": param_pertub_list[id]}

                # Disable sample_params and Plot samples buttons after successful loading
                main_window.sample_params_button.setEnabled(False)
                main_window.plot_samples_button.setEnabled(True)
                # Check if qois are defined
                if main_window.df_metadata['QoIs'].iloc[0] == "None":
                    main_window.define_qoi_button.setEnabled(True)
                else:
                    main_window.define_qoi_button.setEnabled(False)
                # Set db file to the loaded and disable the field
                main_window.db_file_name_input.setText(main_window.db_file_path)
                main_window.db_file_name_input.setEnabled(False)
                # Disable select the qois and run simulations
                main_window.run_simulations_button.setEnabled(False)

                # Enable the button run SA
                main_window.run_sa_button.setEnabled(True)

                # print a message in the output fields of Tab 2
                message = f"Database file loaded: {main_window.db_file_path}"
                main_window.update_output_tab2(main_window, message)
            except Exception as e:
                error_message = f"Error loading .db file: {e}"
                main_window.update_output_tab2(main_window, error_message)

def update_analysis_type(main_window):
    # Show/hide UI elements based on selected analysis type
    analysis_type = main_window.analysis_type_dropdown.currentText()
    if analysis_type == "Local":
        # Local fields
        main_window.local_param_label.setVisible(True)
        main_window.local_param_combo.setVisible(True)
        main_window.local_ref_value_label.setVisible(True)
        main_window.local_ref_value_input.setVisible(True)
        main_window.local_perturb_label.setVisible(True)
        main_window.local_perturb_input.setVisible(True)
        # Global fields
        main_window.global_method_label.setVisible(False)
        main_window.global_method_combo.setVisible(False)
        main_window.global_sampler_label.setVisible(False)
        main_window.global_sampler_combo.setVisible(False)
        main_window.global_param_label.setVisible(False)
        main_window.global_param_combo.setVisible(False)
        main_window.global_ref_value_label.setVisible(False)
        main_window.global_ref_value_input.setVisible(False)
        main_window.global_range_percentage_label.setVisible(False)
        main_window.global_range_percentage.setVisible(False)
        main_window.global_bounds_label.setVisible(False)
        main_window.global_bounds.setVisible(False)

        # Populate the combo box with parameters from analysis and rules
        main_window.local_param_combo.clear()
        main_window.local_SA_parameters = {}

        # Populate local_SA_parameters with reference values and default perturbations
        for key, value in main_window.analysis_parameters.items():
            ref_value = float(main_window.get_xml_value(main_window, key))  # Get the default XML value - string
            main_window.local_SA_parameters[value[1]] = {"reference": ref_value, "perturbations": [1, 10, 20]}

        for key, value in main_window.analysis_rules_parameters.items():
            ref_value = main_window.get_rule_value(main_window, key)  # Get the default rule value
            main_window.local_SA_parameters[value[1]] = {"reference": ref_value, "perturbations": [1, 10, 20]}

        # Add friendly names to the combo box
        main_window.local_param_combo.addItems(list(main_window.local_SA_parameters.keys()))

        # Connect the combo box to update the input fields
        main_window.local_param_combo.currentIndexChanged.connect(lambda: main_window.update_local_inputs(main_window))

        # Initialize the input fields for the first parameter
        if main_window.local_param_combo.count() > 0:
            main_window.update_local_inputs(main_window)
    elif analysis_type == "Global":
        # Local fields
        main_window.local_param_label.setVisible(False)
        main_window.local_param_combo.setVisible(False)
        main_window.local_ref_value_label.setVisible(False)
        main_window.local_ref_value_input.setVisible(False)
        main_window.local_perturb_label.setVisible(False)
        main_window.local_perturb_input.setVisible(False)
        # Global fields
        main_window.global_method_label.setVisible(True)
        main_window.global_method_combo.setVisible(True)
        main_window.global_sampler_label.setVisible(True)
        main_window.global_sampler_combo.setVisible(True)
        main_window.global_param_label.setVisible(True)
        main_window.global_param_combo.setVisible(True)
        main_window.global_ref_value_label.setVisible(True)
        main_window.global_ref_value_input.setVisible(True)
        main_window.global_range_percentage_label.setVisible(True)
        main_window.global_range_percentage.setVisible(True)
        main_window.global_bounds_label.setVisible(True)
        main_window.global_bounds.setVisible(True)


        # Populate the combo box with parameters from analysis and rules
        main_window.global_param_combo.clear()
        main_window.global_SA_parameters = {}

        # Populate global_SA_parameters with reference values and default range percentage
        for key, value in main_window.analysis_parameters.items():
            ref_value = float(main_window.get_xml_value(main_window, key))  # Get the default XML value - string
            main_window.global_SA_parameters[value[1]] = {"reference": ref_value, "range_percentage": 20.0, "bounds": [float(ref_value) * 0.8, float(ref_value) * 1.2]}

        for key, value in main_window.analysis_rules_parameters.items():
            ref_value = main_window.get_rule_value(main_window, key)  # Get the default rule value
            main_window.global_SA_parameters[value[1]] = {"reference": ref_value, "range_percentage": 20.0, "bounds": [float(ref_value) * 0.8, float(ref_value) * 1.2]}

        # Add friendly names to the combo box
        main_window.global_param_combo.addItems(list(main_window.global_SA_parameters.keys()))

        # Connect the combo box to update the input fields
        main_window.global_param_combo.currentIndexChanged.connect(lambda: main_window.update_global_inputs(main_window))

        # Initialize the input fields for the first parameter
        if main_window.global_param_combo.count() > 0:
            main_window.update_global_inputs(main_window)

        # Connect signals to synchronize global inputs
        main_window.global_ref_value_input.editingFinished.connect(lambda: main_window.update_global_inputs(main_window))
        main_window.global_range_percentage.editingFinished.connect(lambda: main_window.update_global_inputs(main_window))

        # Ensure the sampler options are updated initially
        main_window.global_method_combo.currentIndexChanged.connect(lambda: main_window.update_global_sampler_options(main_window))
        main_window.update_global_sampler_options(main_window)

def update_local_inputs(main_window):
    # Update the reference value and perturbations based on the selected parameter
    selected_param = main_window.local_param_combo.currentText()
    if selected_param in main_window.local_SA_parameters:
        param_data = main_window.local_SA_parameters[selected_param]
        main_window.local_ref_value_input.setText(str(param_data["reference"]))
        main_window.local_perturb_input.setText(",".join(map(str, param_data["perturbations"])))

        # Connect signals to update local_SA_parameters when editing is finished
        main_window.local_ref_value_input.editingFinished.connect(lambda: main_window.update_local_SA_reference(main_window))
        main_window.local_perturb_input.editingFinished.connect(lambda: main_window.update_local_SA_perturbations(main_window))

def update_local_SA_reference(main_window):
    # Update the reference value in local_SA_parameters
    selected_param = main_window.local_param_combo.currentText()
    if selected_param in main_window.local_SA_parameters:
        try:
            new_ref_value = float(main_window.local_ref_value_input.text())
            main_window.local_SA_parameters[selected_param]["reference"] = new_ref_value
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid reference value.")

def update_local_SA_perturbations(main_window):
    # Update the perturbations in local_SA_parameters
    selected_param = main_window.local_param_combo.currentText()
    if selected_param in main_window.local_SA_parameters:
        try:
            new_perturbations = [float(p) for p in main_window.local_perturb_input.text().split(",")]
            main_window.local_SA_parameters[selected_param]["perturbations"] = new_perturbations
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid perturbation values.")

def update_global_inputs(main_window):
    # Update the reference value, range percentage, and bounds based on the selected parameter
    selected_param = main_window.global_param_combo.currentText()
    if selected_param in main_window.global_SA_parameters:
        param_data = main_window.global_SA_parameters[selected_param]
        main_window.global_ref_value_input.setText(str(param_data["reference"]))
        main_window.global_range_percentage.setText(str(param_data["range_percentage"]))

        try:
            range_percentage = param_data["range_percentage"]
            lower_bound = param_data["reference"] * (1 - range_percentage / 100)
            upper_bound = param_data["reference"] * (1 + range_percentage / 100)
            main_window.global_bounds.setText(f"{lower_bound:.3e}, {upper_bound:.3e}")
            # Store bounds in global_SA_parameters
            main_window.global_SA_parameters[selected_param]["bounds"] = [lower_bound, upper_bound]
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid reference value or range percentage.")

        # Update global_SA_parameters when editing is finished
        main_window.global_ref_value_input.editingFinished.connect(lambda: main_window.update_global_SA_reference(main_window))
        main_window.global_range_percentage.editingFinished.connect(lambda: main_window.update_global_SA_range_percentage(main_window))

def update_global_SA_reference(main_window):
    # Update the reference value in global_SA_parameters
    selected_param = main_window.global_param_combo.currentText()
    if selected_param in main_window.global_SA_parameters:
        try:
            new_ref_value = float(main_window.global_ref_value_input.text())
            main_window.global_SA_parameters[selected_param]["reference"] = new_ref_value
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid reference value.")

def update_global_SA_range_percentage(main_window):
    # Update the range percentage in global_SA_parameters
    selected_param = main_window.global_param_combo.currentText()
    if selected_param in main_window.global_SA_parameters:
        try:
            new_range_percentage = float(main_window.global_range_percentage.text())
            main_window.global_SA_parameters[selected_param]["range_percentage"] = new_range_percentage
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid range percentage.")

def sample_parameters(main_window):
    # Sample parameters based on the selected SA
    analysis_type = main_window.analysis_type_dropdown.currentText()
    if analysis_type == "Local":
        try:
            main_window.update_output_tab2(main_window, f"Sampling parameters from One-At-A-Time method...")
            if 'samples' in main_window.local_SA_parameters.keys():
                del main_window.local_SA_parameters["samples"]
            param_names = list(main_window.local_SA_parameters.keys())
            params_ref = {key: main_window.local_SA_parameters[key]["reference"] for key in main_window.local_SA_parameters.keys()}
            # First sample is the reference value
            main_window.local_SA_parameters["samples"] = {0: params_ref}
            main_window.update_output_tab2(main_window, f"\tAdding Reference value: {params_ref}")
            for id, par in enumerate(param_names):
                perturbations = np.array(main_window.local_SA_parameters[par]["perturbations"])
                main_window.update_output_tab2(main_window, f"\tSampling parameter: <{par}> using Perturbations: +/- {perturbations}%")
                perturbations = np.concatenate((-perturbations, perturbations))  # Combine negative and positive perturbations
                for idx, var in enumerate(perturbations):
                    sample_id = id * len(perturbations) + idx + 1  # Unique sample ID for each perturbation
                    if sample_id not in (main_window.local_SA_parameters["samples"].keys()):
                        main_window.local_SA_parameters["samples"][sample_id] = params_ref.copy()  # Start with reference values
                    main_window.local_SA_parameters["samples"][sample_id][par] = params_ref[par] * (1 + var / 100.0)
            main_window.update_output_tab2(main_window, f"Generated {len(main_window.local_SA_parameters['samples'])} samples.")
            # print(main_window.local_SA_parameters["samples"])
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid input for reference value or perturbations.")
    elif analysis_type == "Global":
        try:
            method = main_window.global_method_combo.currentText()
            sampler = main_window.global_sampler_combo.currentText()
            main_window.update_output_tab2(main_window, f"Sampling parameters from method: {method} and sampler: {sampler}.")
            if 'samples' in main_window.global_SA_parameters.keys():
                del main_window.global_SA_parameters["samples"]
            param_names = list(main_window.global_SA_parameters.keys())
            # Define SA problem
            SA_problem = {
                'num_vars': len(main_window.global_SA_parameters),
                'names': param_names,
                'bounds': [main_window.global_SA_parameters[param]["bounds"] for param in main_window.global_SA_parameters]
            }
            # Samplers: "Fast", "Fractional Factorial", "Finite Difference", 
            # "Latin hypercube sampling (LHS)", "Sobol", "Morris"
            if sampler == "Fast":
                #  The number of samples is N*D, where D is the number of parameters
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples:", value=8, min=1)
                M = 4
                if not ok or N <= 4*(M**2):
                    main_window.update_output_tab2(main_window, "Error: The number of samples must be a positive integer greater than 4*(M**2).")
                    return
                else: 
                    global_samples = fast_sampler.sample(SA_problem, N, M=M, seed=42)
            elif sampler == "Fractional Factorial":
                # The number of samples is 2**D, where D is the number of parameters
                global_samples = ff.sample(SA_problem, seed=42)
            elif sampler == "Finite Difference":
                # The number of samples is N*D, where D is the number of parameters
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples:", value=8, min=1)
                if not ok or N < 1:
                    main_window.update_output_tab2(main_window, "Error: The number of samples must be a positive integer.")
                    return
                global_samples = finite_diff.sample(SA_problem, N, seed=42)
            elif sampler == "Latin hypercube sampling (LHS)":
                # The number of samples is N
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples:", value=8, min=1)
                if not ok or N < 1:
                    main_window.update_output_tab2(main_window, "Error: The number of samples must be a positive integer.")
                    return
                global_samples = latin.sample(SA_problem, N, seed=42)
            elif sampler == "Sobol":
                # If second-order indices: N*(2D+2) sample if only first-order indices: N*(D+2) samples, where D is the number of parameters
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples (must be a power of 2):", value=8, min=2)
                if not ok or (N & (N - 1)) != 0:
                    main_window.update_output_tab2(main_window, "Error: The number of samples must be a power of 2.")
                    return
                global_samples = sobol.sample(SA_problem, N, calc_second_order=True, seed=42)
            elif sampler == "Morris":
                # The number of samples is (G/D +1)*N/T, where D is the number of parameters, G is the number of groups, and T is the number of trajectories
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples:", value=8, min=1)
                if not ok or N < 1:
                    main_window.update_output_tab2(main_window, "Error: The number of samples must be a positive integer.")
                    return
                global_samples = morris.sample(SA_problem, N, seed=42)
            else:
                main_window.update_output_tab2(main_window, "Error: Invalid sampler selected.")
                return
            main_window.update_output_tab2(main_window, f"Generated {global_samples.shape[0]} samples.")
            # Convert the samples to a dictionary of dictionaries
            main_window.global_SA_parameters["samples"] = {}
            for i in range(global_samples.shape[0]):
                sample_dict = {}
                for j, param in enumerate(param_names):
                    sample_dict[param] = global_samples[i][j]
                main_window.global_SA_parameters["samples"][i] = sample_dict
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error sampling {analysis_type} SA: {e}")
            return

    # Make the define_qoi, DB file name field and "Run Simulations" button visible
    main_window.db_file_name_input.setEnabled(True)
    main_window.define_qoi_button.setEnabled(True)
    main_window.run_simulations_button.setEnabled(True)

def plot_samples(main_window):
    # Toggle the visibility of the sample plot
    if main_window.fig_est_canvas_samples.isVisible():
        main_window.fig_est_canvas_samples.setVisible(False)
        main_window.plot_samples_button.setText("Plot Samples")
    else:
        main_window.fig_est_canvas_samples.setVisible(True)
        main_window.plot_samples_button.setText("Hide Samples")
        main_window.ax_samples.clear()
        try:
            if main_window.analysis_type_dropdown.currentText() == "Local":
                total_samples = 1
                for key in main_window.local_SA_parameters.keys():
                    if key == "samples": continue
                    # Plot the perturbations for each parameter
                    perturbations = np.array(main_window.local_SA_parameters[key]["perturbations"])
                    perturbations = np.concatenate((-1.0 * perturbations, perturbations), axis=None)
                    total_samples += len(perturbations)
                    main_window.ax_samples.scatter(perturbations, [key] * len(perturbations), s=10, color='k')
                    # Plot the reference value
                    main_window.ax_samples.scatter(0, key, s=10, color='k')
                # Plot vline of reference value
                # main_window.ax_samples.axvline(x=0, color='gray', linestyle='--', label='Reference Value')
                main_window.ax_samples.set_xlabel("Perturbations (%)")
                main_window.ax_samples.set_title(f"One-At-A-Time (OAT) sampling - total of samples: {total_samples}")
                # Adjust plot in canvas
                main_window.fig_est_canvas_samples.figure.tight_layout()  # Call tight_layout on the figure object
                main_window.fig_est_canvas_samples.draw()
            elif main_window.analysis_type_dropdown.currentText() == "Global":
                total_samples = len(main_window.global_SA_parameters["samples"])
                # Plot the samples for each parameter
                for i, key in enumerate(main_window.global_SA_parameters.keys()):
                    if key == "samples": continue
                    global_params = np.array([sample[key] for sample in main_window.global_SA_parameters["samples"].values()])
                    # Plot the samples for each parameter
                    main_window.ax_samples.scatter((global_params-main_window.global_SA_parameters[key]["bounds"][0])/(main_window.global_SA_parameters[key]["bounds"][1] - main_window.global_SA_parameters[key]["bounds"][0]), [key] * total_samples, s=10, color='k')
                # main_window.ax_set_xlim = main_window.ax_samples.set_xlim(0, 1)
                main_window.ax_samples.set_xlabel("Samples")
                main_window.ax_samples.set_title(f"Global sampling - total of samples: {total_samples}")
                # Adjust plot in canvas
                main_window.fig_est_canvas_samples.figure.tight_layout()
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error plotting samples {main_window.analysis_type_dropdown.currentText()} SA: {e}")

def update_global_sampler_options(main_window):
    # FAST - Fourier Amplitude Sensitivity Test: combatible with Fast sampler
    # RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test: combatible with all samplers
    # Method of Morris: combatible with Morris sampler
    # Sobol’ Sensitivity Analysis: combatible with Sobol samplers
    # Delta Moment-Independent Measure: combatible with all samplers
    # Derivative-based Global Sensitivity Measure (DGSM): combatible with Finite Difference sampler
    # Fractional Factorial: combatible with Fractional Factorial sampler
    # PAWN Sensitivity Analysis: combatible with all samplers
    # High-Dimensional Model Representation: combatible with all samplers
    # Regional Sensitivity Analysis: combatible with all samplers
    # Discrepancy Sensitivity Indices: combatible with all samplers
    # Update the global_sampler_combo options based on the selected method
    method = main_window.global_method_combo.currentText()
    main_window.global_sampler_combo.clear()

    if method == "FAST - Fourier Amplitude Sensitivity Test":
        main_window.global_sampler_combo.addItems(["Fast"])
    elif method == "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test":
        main_window.global_sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol", "Morris"
        ])
    elif method == "Method of Morris":
        main_window.global_sampler_combo.addItems(["Morris"])
    elif method == "Sobol Sensitivity Analysis":
        main_window.global_sampler_combo.addItems(["Sobol"])
    elif method == "Delta Moment-Independent Measure":
        main_window.global_sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol", "Morris"
        ])
    elif method == "Derivative-based Global Sensitivity Measure (DGSM)":
        main_window.global_sampler_combo.addItems(["Finite Difference"])
    elif method == "Fractional Factorial":
        main_window.global_sampler_combo.addItems(["Fractional Factorial"])
    elif method in [
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ]:
        main_window.global_sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol", "Morris"
        ])
    else:
        main_window.global_sampler_combo.addItems([])  # No compatible samplers

def run_simulations(main_window):
    # Handle the execution of simulations
    try:
        # Create a dialog window for workers and thread configuration
        dialog = QDialog(main_window)
        dialog.setWindowTitle("Simulation Configuration")
        dialog.setGeometry(100, 100, 400, 200)

        # Create a layout for the dialog
        layout = QVBoxLayout(dialog)

        # Add input for the number of ranks
        workers_layout = QHBoxLayout()
        workers_label = QLabel("Number of workers:")
        workers_input = QLineEdit()
        workers_input.setValidator(QIntValidator(1, 1000))  # Allow only positive integers
        workers_input.setText("1")
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(workers_input)
        layout.addLayout(workers_layout)

        # Display the number of OpenMP threads
        xml_path_omp_threads = ".//parallel/omp_num_threads"
        omp_threads = main_window.xml_tree.find(xml_path_omp_threads).text.strip()
        # Check if the parameter is set in the .ini file
        if xml_path_omp_threads in list(main_window.fixed_parameters.keys()): omp_threads = main_window.fixed_parameters[xml_path_omp_threads]
        total_threads_label = QLabel(f"OpenMP Threads per worker: {omp_threads}")
        layout.addWidget(total_threads_label)

        # Add buttons for confirmation
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect button actions
        def confirm():
            dialog.accept()

        def cancel():
            dialog.reject()

        ok_button.clicked.connect(confirm)
        cancel_button.clicked.connect(cancel)

        # Show the dialog and handle the result
        result = dialog.exec_()
        if result != QDialog.Accepted:
            main_window.update_output_tab2(main_window, "Simulation aborted by user.")
            print("Dialog canceled. Exiting run_simulations early.")  # Debug log
            return  # Ensure the function exits early

        # Retrieve the number of workers after dialog is accepted
        num_workers = workers_input.text()

        # Ensure SA_type is defined only after the dialog is accepted
        SA_type = main_window.analysis_type_dropdown.currentText()
        if not SA_type:
            main_window.update_output_tab2(main_window, "Error: Sensitivity analysis type is not selected.")
            return

        # Initialize other variables only if SA_type is valid
        SA_method, SA_sampler, SA_samples = None, None, None
        if SA_type == "Local":
            try:
                SA_method = "OAT"
                SA_sampler = "OAT"
                SA_samples = main_window.local_SA_parameters.get("samples")
                if SA_samples is None:
                    raise ValueError("No samples generated for local sensitivity analysis.")
            except KeyError:
                main_window.update_output_tab2(main_window, "Error: No samples generated for local sensitivity analysis.")
                return
        elif SA_type == "Global":
            try:
                SA_method = main_window.global_method_combo.currentText()
                SA_sampler = main_window.global_sampler_combo.currentText()
                SA_samples = main_window.global_SA_parameters.get("samples")
                if SA_samples is None:
                    raise ValueError("No samples generated for global sensitivity analysis.")
            except KeyError:
                main_window.update_output_tab2(main_window, "Error: No samples generated for global sensitivity analysis.")
                return
 
        # Ensure all mandatory fields are filled
        db_file_name = main_window.db_file_name_input.text().strip()
        if not db_file_name:
            main_window.update_output_tab2(main_window, "Error: DB file name is required.")
            return
        elif not db_file_name.endswith(".db"):
            main_window.update_output_tab2(main_window, "Error: DB file name must end with '.db'.")
            return
        elif os.path.exists(db_file_name):
            # Ask the user if they want to overwrite the existing file
            reply = QMessageBox.question(main_window, "Overwrite Confirmation", 
                                            f"File '{db_file_name}' already exists. Do you want to overwrite it?", 
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                main_window.update_output_tab2(main_window, "Aborted saving. File was not overwritten.")
                return
        # Check if at least one QoI is selected
        if not main_window.qoi_funcs.keys():
            # Ask the user if they want to run the simulation without QoIs
            reply = QMessageBox.question(main_window, "QoI Warning",
                                            "No QoI(s) defined. Do you want to run the simulation without QoIs? All data will be stored as mcds list.", 
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                main_window.update_output_tab2(main_window, "Aborted saving. No QoI(s) defined.")
                return
            else:
                main_window.update_output_tab2(main_window, "QoI(s) not defined: All data will be stored as mcds list.")
                qoi_str = "None"
        else:
            qoi_str = ', '.join(main_window.qoi_funcs.keys())

        # Determine the samples to use according to the analysis type
        SA_type = main_window.analysis_type_dropdown.currentText()
        if "Local" == SA_type:
            try:
                SA_method = "OAT"
                SA_sampler = "OAT"
                SA_samples = main_window.local_SA_parameters.get("samples")
                SA_param_names = [key for key in main_window.local_SA_parameters.keys() if key != "samples"]
                SA_bounds = None
                SA_ref_values = [main_window.local_SA_parameters[param]["reference"] for param in SA_param_names]
                SA_perturbations = [main_window.local_SA_parameters[param]["perturbations"] for param in SA_param_names]
                if SA_samples is None:
                    raise ValueError("No samples generated for local sensitivity analysis.")
            except KeyError:
                main_window.update_output_tab2(main_window, "Error: No samples generated for local sensitivity analysis.")
                return
        elif "Global" == SA_type:
            try:
                SA_method = main_window.global_method_combo.currentText()
                SA_sampler = main_window.global_sampler_combo.currentText()
                SA_samples = main_window.global_SA_parameters.get("samples")
                SA_param_names = [key for key in main_window.global_SA_parameters.keys() if key != "samples"]
                SA_bounds = [main_window.global_SA_parameters[param]["bounds"] for param in SA_param_names]
                SA_ref_values = [main_window.global_SA_parameters[param]["reference"] for param in SA_param_names]
                SA_perturbations = [main_window.global_SA_parameters[param]["range_percentage"] for param in SA_param_names]
                if SA_samples is None:
                    raise ValueError("No samples generated for global sensitivity analysis.")
            except KeyError:
                main_window.update_output_tab2(main_window, "Error: No samples generated for global sensitivity analysis.")
                return

        # Validate other required inputs
        if not main_window.ini_file_path:
            main_window.update_output_tab2(main_window, "Error: .ini file path is missing.")
            return
        if not main_window.struc_name_input.text().strip():
            main_window.update_output_tab2(main_window, "Error: Structure name is required.")
            return
    except Exception as e:
        main_window.update_output_tab2(main_window, f"Error during simulation execution: {e}")

    # Update the output tab with the current status
    main_window.update_output_tab2(main_window, f"Running simulations with {SA_type} method: {SA_method}, sampler: {SA_sampler}, number of samples: {len(SA_samples)}")
    # Debugging information
    print(f"Running simulations with {SA_type} method: {SA_method}, sampler: {SA_sampler}, number of samples: {len(SA_samples)}")
    print(f".ini file path: {main_window.ini_file_path} - DB file name: {db_file_name} - structure name: {main_window.struc_name_input.text().strip()} - selected QoIs: {qoi_str}")
    print(f"Parameters: {SA_param_names} - Bounds: {SA_bounds} - Reference values: {SA_ref_values} - Perturbations: {SA_perturbations}")

    # Simulate running the model with the selected samples
    try: run_sa_simulations(
        ini_filePath=main_window.ini_file_path,
        strucName=main_window.struc_name_input.text().strip(),
        SA_type=SA_type,
        SA_method=SA_method,
        SA_sampler=SA_sampler,
        param_names=SA_param_names,
        ref_values=SA_ref_values,
        bounds=SA_bounds,
        perturbations=SA_perturbations,
        dic_samples=SA_samples,
        qois_dic=main_window.qoi_funcs,
        db_file=db_file_name,
        use_futures=True,
        num_workers=int(num_workers)  # Pass the number of workers to the function
        )
    except Exception as e:
        main_window.update_output_tab2(main_window, f"Error running simulations: {e}")
        return

    # Simulate saving the samples to the database (replace with actual simulation logic)
    main_window.update_output_tab2(main_window, f"Simulations completed and saved to {db_file_name} with QoIs: {', '.join(main_window.qoi_funcs.keys())}.")
    # Load the database file to display results
    main_window.load_db_file(main_window, db_file_name)


def run_analysis(main_window):
    main_window.update_output_tab2(main_window, "Running sensitivity analysis...")
    # Check if the qoi_funcs dictionary is empty
    if not main_window.qoi_funcs:
        main_window.update_output_tab2(main_window, "Error: No QoI functions defined.")
        return
    # If QoIs are not in db, calculate them and store them in df_output as columns
    if not list(main_window.qoi_funcs.keys())[0] in main_window.df_output.columns:
        main_window.update_output_tab2(main_window, "Calculating QoIs...")
        try:
            df_qois = extract_qoi_from_db(main_window.db_file_path, main_window.qoi_funcs)
            if df_qois.empty: 
                main_window.update_output_tab2(main_window, "Error: No able to extract QoIs from the database.")
                return
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error calculating QoIs: {e}")
            return
    else:
        df_qois = main_window.df_output
    # Run the analysis
    all_qois = list(main_window.qoi_funcs.keys())
    all_times = [col for col in df_qois.columns if col.startswith("time")]
    print(f"all_qois: {all_qois} and all_times: {all_times}")
    # Take the average amoung the replicates and sort the samples
    df_qois = df_qois.groupby(['SampleID']).mean().reset_index()
    df_qois.drop(columns=['ReplicateID'], inplace=True)
    df_qois = df_qois.set_index("SampleID").sort_index()
    # Convert df_input to a NumPy array with shape (number of samples, number of parameters)
    main_window.sa_results = { qoi: {time: None for time in all_times} for qoi in all_qois }
    main_window.qoi_time_values = { time: None for time in all_times }
    if main_window.analysis_type_dropdown.currentText() == "Global":
        param_names = [key for key in main_window.global_SA_parameters.keys() if key != "samples"]
        SA_problem = {
            'num_vars': len(param_names),
            'names': param_names,
            'bounds': [main_window.global_SA_parameters[param]["bounds"] for param in param_names]
        }
        
        for qoi in all_qois:
            for id_time, time in enumerate(list(main_window.qoi_time_values.keys())):
                global_method = main_window.global_method_combo.currentText()
                dic_params = np.array([[dic[param] for param in SA_problem['names']] for dic in main_window.global_SA_parameters["samples"].values()])
                qoi_result = df_qois[f"{qoi}_{id_time}"].to_numpy()
                print(f"qoi_result ({qoi}_{id_time}): {qoi_result}")
                unique_times = df_qois[time].unique()
                if len(unique_times) != 1:
                    raise ValueError(f"Expected a single unique value for time '{time}', but found: {unique_times}")
                main_window.qoi_time_values[time] = unique_times[0]
                main_window.update_output_tab2(main_window, f"Running {global_method} for QoI: {qoi} and time: {main_window.qoi_time_values[time]}") 
                # Run the sensitivity analysis
                if global_method == "FAST - Fourier Amplitude Sensitivity Test":
                    main_window.sa_results[qoi][time] = fast_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test":
                    main_window.sa_results[qoi][time] = rbd_fast_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Method of Morris":
                    main_window.sa_results[qoi][time] = morris_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Sobol Sensitivity Analysis":
                    main_window.sa_results[qoi][time] = sobol_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Delta Moment-Independent Measure":
                    main_window.sa_results[qoi][time] = delta_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Derivative-based Global Sensitivity Measure (DGSM)":
                    main_window.sa_results[qoi][time] = dgsm_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Fractional Factorial":
                    main_window.sa_results[qoi][time] = ff_analyze.analyze(SA_problem, dic_params, qoi_result, second_order=True)
                elif global_method == "PAWN Sensitivity Analysis":
                    main_window.sa_results[qoi][time] = pawn_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "High-Dimensional Model Representation":
                    main_window.sa_results[qoi][time] = hdmr_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Regional Sensitivity Analysis":
                    main_window.sa_results[qoi][time] = rsa_analyze.analyze(SA_problem, dic_params, qoi_result)
                elif global_method == "Discrepancy Sensitivity Indices":
                    main_window.sa_results[qoi][time] = discrepancy_analyze.analyze(SA_problem, dic_params, qoi_result)
                else:
                    main_window.update_output_tab2(main_window, "Error: Invalid global method selected.")
        
        # Plot the results
        main_window.update_output_tab2(main_window, f"Plotting results for {global_method}")
        main_window.plot_sa_results(main_window)
    elif main_window.analysis_type_dropdown.currentText() == "Local":
        param_names = [key for key in main_window.local_SA_parameters.keys() if key != "samples"]
        for qoi in all_qois:
            for id_time, time in enumerate(list(main_window.qoi_time_values.keys())):
                local_method = "OAT"
                qoi_result = df_qois[f"{qoi}_{id_time}"].to_dict()
                print(f"qoi_result ({qoi}_{id_time}): {qoi_result.values()}")
                unique_times = df_qois[time].unique()
                if len(unique_times) != 1:
                    raise ValueError(f"Expected a single unique value for time '{time}', but found: {unique_times}")
                main_window.qoi_time_values[time] = unique_times[0]
                main_window.update_output_tab2(main_window, f"Running {local_method} for QoI: {qoi} and time: {main_window.qoi_time_values[time]}")
                # Run the sensitivity analysis
                if local_method == "OAT":
                    main_window.sa_results[qoi][time] = OAT_analyze(main_window.local_SA_parameters["samples"], qoi_result)
                    # It will return a sensitivity index for each perturbation - sum them
                    for key in main_window.sa_results[qoi][time]: main_window.sa_results[qoi][time][key] = np.sum(main_window.sa_results[qoi][time][key])
                else:
                    main_window.update_output_tab2(main_window, "Error: Invalid local method selected.") 
        
        # Plot the results
        main_window.update_output_tab2(main_window, f"Plotting results for {local_method} SA.")
        main_window.plot_sa_results(main_window)

def plot_sa_results(main_window):
    # Create a new dialog window for the plot
    plot_window = QDialog(main_window)
    plot_window.setWindowTitle("Sensitivity Analysis Results")
    plot_window.setGeometry(100, 100, 800, 600)

    # Create a layout for the dialog
    layout = QVBoxLayout(plot_window)

    # Add a combo box to select the qoi
    plot_sa_qoi_hbox = QHBoxLayout()
    plot_sa_label = QLabel("Select the QoI to plot:")
    plot_sa_label.setAlignment(Qt.AlignCenter)
    plot_sa_qoi_hbox.addWidget(plot_sa_label)
    plot_sa_dropdown_qoi = QComboBox(plot_window)
    plot_sa_dropdown_qoi.setEditable(False)
    plot_sa_dropdown_qoi.addItems(list(main_window.sa_results.keys()))
    plot_sa_qoi_hbox.addWidget(plot_sa_dropdown_qoi)
    layout.addLayout(plot_sa_qoi_hbox)

    # Add a combo box to select the time
    plot_sa_time_hbox = QHBoxLayout()
    plot_sa_time_label = QLabel("Select the time to plot (min):")
    plot_sa_time_label.setAlignment(Qt.AlignCenter)
    plot_sa_time_hbox.addWidget(plot_sa_time_label)
    plot_sa_time_dropdown = QComboBox(plot_window)
    plot_sa_time_dropdown.setEditable(False)
    plot_sa_time_dropdown.addItems([str(value) for value in main_window.qoi_time_values.values()])
    plot_sa_time_hbox.addWidget(plot_sa_time_dropdown)
    layout.addLayout(plot_sa_time_hbox)

    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(5, 3))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)

    # Define the update_plot function before connecting it
    def update_plot():
        # Get the selected qoi
        selected_qoi = plot_sa_dropdown_qoi.currentText()
        selected_time = next((key for key, value in main_window.qoi_time_values.items() if str(value) == plot_sa_time_dropdown.currentText()), None)
        # Clear the previous plot
        print(f"Plotting {selected_qoi} at time {selected_time} - {main_window.sa_results[selected_qoi][selected_time]}")
        figure.clear()
        if main_window.analysis_type_dropdown.currentText() == "Global":
            SA_method = main_window.global_method_combo.currentText()
            ax_1 = figure.add_subplot(1, 1, 1)
            if ('names' in main_window.sa_results[selected_qoi][selected_time].keys()):
                if ('S1' in main_window.sa_results[selected_qoi][selected_time].keys() and 'S1_conf' in main_window.sa_results[selected_qoi][selected_time].keys()):
                    ax_1.barh(main_window.sa_results[selected_qoi][selected_time]['names'], main_window.sa_results[selected_qoi][selected_time]['S1'], xerr=main_window.sa_results[selected_qoi][selected_time]['S1_conf'])
                elif ('ME' in main_window.sa_results[selected_qoi][selected_time].keys()):
                    ax_1.barh(main_window.sa_results[selected_qoi][selected_time]['names'], main_window.sa_results[selected_qoi][selected_time]['ME'])
            else:
                main_window.update_output_tab2(main_window, "Error: Invalid global SA results. Keys: " + str(main_window.sa_results[selected_qoi][selected_time].keys()))
                return
                
            ax_1.set_title(f"Global SA - {SA_method}")
        elif main_window.analysis_type_dropdown.currentText() == "Local":
            SA_method = "OAT method"
            ax_1 = figure.add_subplot(1, 1, 1)
            ax_1.barh(list(main_window.sa_results[selected_qoi][selected_time].keys()), list(main_window.sa_results[selected_qoi][selected_time].values()))
            ax_1.set_xlabel("Sensitivity Index")
            ax_1.set_title(f"Local SA - {SA_method}")
        # Adjust layout and draw the canvas
        figure.tight_layout()
        canvas.draw()

    # Connect the combo box to update the plot
    plot_sa_dropdown_qoi.currentIndexChanged.connect(update_plot)
    plot_sa_time_dropdown.currentIndexChanged.connect(update_plot)

    # Set the default selected qoi and update the plot
    plot_sa_dropdown_qoi.setCurrentIndex(0)
    update_plot()

    # Show the dialog
    plot_window.exec_()
