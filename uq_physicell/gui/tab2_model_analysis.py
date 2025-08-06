import os, sys
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QComboBox, QLineEdit, QTextEdit, QDialog, QFileDialog, QInputDialog, QListWidget, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
import pandas as pd
import traceback

# My local modules
from uq_physicell.model_analysis.samplers import run_global_sampler, run_local_sampler
from uq_physicell.model_analysis.sensitivity_analysis import run_global_sa, run_local_sa, samplers_to_method
from uq_physicell.model_analysis.main import ModelAnalysisContext, run_simulations
from uq_physicell.model_analysis.utils import calculate_qoi_statistics
from uq_physicell.model_analysis.database import load_structure

def create_tab2(main_window):
    # Add the following methods to the main_window instance
    main_window.update_output_tab2 = update_output_tab2
    main_window.load_ma_database = load_ma_database
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
    main_window.run_simulations_function = run_simulations_function
    main_window.run_analysis = run_analysis
    main_window.plot_sa_results = plot_sa_results
    main_window.plot_qois = plot_qois

    layout_tab2 = QVBoxLayout()

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
    main_window.analysis_type_hbox.addStretch()
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
    main_window.globalSA_method_hbox.addStretch()
    main_window.globalSA_method_hbox.addSpacing(10)
    # Sampler options
    main_window.global_sampler_label = QLabel("Sampler:")
    main_window.global_sampler_label.setVisible(False)
    main_window.globalSA_method_hbox.addWidget(main_window.global_sampler_label)
    main_window.global_sampler_combo = QComboBox()
    main_window.global_sampler_combo.setVisible(False)
    main_window.globalSA_method_hbox.addWidget(main_window.global_sampler_combo)
    main_window.globalSA_vbox.addLayout(main_window.globalSA_method_hbox)
    main_window.globalSA_method_hbox.addStretch()
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
    main_window.sample_params_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.sample_params_button.clicked.connect(lambda: main_window.sample_parameters(main_window))
    main_window.sample_params_button.setEnabled(False)
    buttonSA_hbox.addWidget(main_window.sample_params_button)
    # Plot samples button
    main_window.plot_samples_button = QPushButton("Plot Samples")
    main_window.plot_samples_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_samples_button.clicked.connect(lambda: main_window.plot_samples(main_window))
    buttonSA_hbox.addWidget(main_window.plot_samples_button)
    main_window.plot_samples_button.setEnabled(False)
    layout_tab2.addLayout(buttonSA_hbox)

    ###########################################
    # Horizontal layout for db file name, QoI selection, plot_qoi, and run SA button
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
    main_window.define_qoi_button = QPushButton("Define QoI(s)")
    main_window.define_qoi_button.setEnabled(False)
    main_window.define_qoi_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.define_qoi_button.clicked.connect(lambda: open_qoi_definition_window(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.define_qoi_button)
    # Run simulations button
    main_window.run_simulations_button = QPushButton("Run Simulations")
    main_window.run_simulations_button.setEnabled(False)
    main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.run_simulations_button.clicked.connect(lambda: main_window.run_simulations_function(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.run_simulations_button)
    # Plot QoI button
    main_window.plot_qois_button = QPushButton("Plot QoI(s)")
    main_window.plot_qois_button.setEnabled(False)
    main_window.plot_qois_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_qois_button.clicked.connect(lambda: main_window.plot_qois(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.plot_qois_button)
    # Run SA button
    main_window.run_sa_button = QPushButton("Run SA")
    main_window.run_sa_button.setEnabled(False)
    main_window.run_sa_button.setStyleSheet("background-color: lightgreen; color: black")
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

    # Predefined QoIs dictionary based on the database structure
    predefined_qoi_funcs = {}
    custom_qoi_option = True
    if not main_window.df_output.empty:
        if isinstance(main_window.df_output['Data'].iloc[0], pd.DataFrame):
            predefined_qoi_funcs = {qoi_name: None for qoi_name in main_window.df_output['Data'].iloc[0].columns if qoi_name not in ['time'] }
            custom_qoi_option = False
    if not predefined_qoi_funcs:
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

    # Reset the qois
    main_window.df_qois = pd.DataFrame()
    
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
    predefined_qoi_lambda_display.setText(list(predefined_qoi_funcs.keys())[0])  # Default display first qoi
    layout.addWidget(predefined_qoi_lambda_display)

    # Update lambda display when a predefined QoI is selected
    predefined_qoi_combo.currentIndexChanged.connect(
        lambda: predefined_qoi_lambda_display.setText(predefined_qoi_funcs[predefined_qoi_combo.currentText()])
    )

    # Add predefined QoI button
    add_predefined_qoi_button = QPushButton("Add Selected Predefined QoI")
    add_predefined_qoi_button.setStyleSheet("background-color: lightgreen; color: black")
    layout.addWidget(add_predefined_qoi_button)

    # Separator line
    layout.addWidget(QLabel("<hr>"))

    # Custom QoI section
    custom_qoi_label = QLabel("<b>Define Custom QoIs</b>")
    custom_qoi_label.setAlignment(Qt.AlignCenter)
    custom_qoi_label.setEnabled(custom_qoi_option)
    layout.addWidget(custom_qoi_label)

    # Input field for custom QoI name
    custom_qoi_name_input = QLineEdit()
    custom_qoi_name_input.setPlaceholderText("Enter QoI name")
    custom_qoi_name_input.setEnabled(custom_qoi_option)
    layout.addWidget(custom_qoi_name_input)

    # Input field for custom lambda function
    custom_lambda_input = QLineEdit()
    custom_lambda_input.setPlaceholderText("Enter lambda function (e.g., lambda df: len(df[df['dead'] == False]))")
    custom_lambda_input.setEnabled(custom_qoi_option)
    layout.addWidget(custom_lambda_input)

    # Add custom QoI button
    add_custom_qoi_button = QPushButton("Add Custom QoI")
    add_custom_qoi_button.setEnabled(custom_qoi_option)
    add_custom_qoi_button.setStyleSheet("background-color: lightgreen; color: black")
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
    remove_qoi_button.setStyleSheet("background-color: yellow; color: black")
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
    close_button.setStyleSheet("background-color: lightgreen; color: black")
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

def load_ma_database(main_window):
    try:
        # Load the database structure
        main_window.update_output_tab2(main_window, f"Loading database file {main_window.ma_file_path} ...")
        df_metadata, df_parameter_space, df_qois, dic_input, main_window.df_output = load_structure(main_window.ma_file_path)

        # Load the .ini file
        main_window.load_ini_file(main_window, df_metadata['Ini_File_Path'].iloc[0], df_metadata['StructureName'].iloc[0])
        print(df_metadata)
        # Define the widget to display db structure
        Sampler = df_metadata['Sampler'].iloc[0]
        Param_explorer_type = "Local" if Sampler == "OAT" else "Global"
        # Switch the exploration type to the one defined in the database
        main_window.analysis_type_dropdown.setCurrentText(Param_explorer_type)
        main_window.update_analysis_type(main_window)
        main_window.analysis_type_dropdown.setEnabled(False)
        if Param_explorer_type == "Global": # Global fields
            main_window.global_method_combo.clear()
            if Sampler in samplers_to_method:
                main_window.global_method_combo.addItems(samplers_to_method[Sampler])
            else: # LHS sampler map to methods that are not constrained
                main_window.global_method_combo.addItems(samplers_to_method["Latin hypercube sampling (LHS)"])
            main_window.global_method_combo.setCurrentText(main_window.global_method_combo.itemText(0))
            # main_window.global_method_combo.setEnabled(False)
            main_window.global_sampler_combo.setCurrentText(Sampler)
            main_window.global_param_combo.setEnabled(True)
            main_window.global_ref_value_input.setEnabled(False)
            main_window.global_sampler_combo.setEnabled(False)
            main_window.global_range_percentage.setEnabled(False)
            main_window.global_bounds.setEnabled(False)
            # Populate the global_SA_parameters dictionary with values from the database
            main_window.global_SA_parameters = {}
            main_window.global_SA_parameters["samples"] = dic_input
            for id, param in enumerate(df_parameter_space['ParamName']):
                main_window.global_SA_parameters[param] = {"lower_bounds": df_parameter_space['Lower_Bound'].iloc[id],
                                                            "upper_bounds": df_parameter_space['Upper_Bound'].iloc[id],
                                                            "ref_value": df_parameter_space['ReferenceValue'].iloc[id], 
                                                            "perturbation": float(df_parameter_space['Perturbation'].iloc[id])}
            # Update the global parameters
            main_window.update_global_inputs(main_window)
        else: # Activate local fields
            main_window.local_param_combo.setEnabled(True)
            main_window.local_ref_value_input.setEnabled(True)
            main_window.local_perturb_input.setEnabled(True)
            # Populate the local_SA_parameters dictionary with values from the database
            main_window.local_SA_parameters = {}
            main_window.local_SA_parameters["samples"] = dic_input
            for id, param in enumerate(df_parameter_space['ParamName']):
                if type(df_parameter_space['Perturbation'].iloc[id]) == list:
                    df_parameter_space['Perturbation'].iloc[id] = [float(x) for x in df_parameter_space['Perturbation'].iloc[id]]
                main_window.local_SA_parameters[param] = {"ref_value": df_parameter_space['ReferenceValue'].iloc[id], 
                                                            "perturbation": df_parameter_space['Perturbation'].iloc[id]}
            # Update the local parameters
            main_window.update_local_inputs(main_window)
    
        # Check if qois are defined
        if (df_qois['QOI_Name'].iloc[0] == None):
            main_window.define_qoi_button.setEnabled(True)
        else:
            main_window.define_qoi_button.setEnabled(False)
        
        # Enable the plot QoI button
        main_window.plot_qois_button.setEnabled(True)
        # Enable the button run SA
        main_window.run_sa_button.setEnabled(True)

        # Disable sample_params button after successful loading
        main_window.sample_params_button.setEnabled(False)
        # Disable run simulations button after successful loading
        main_window.run_simulations_button.setEnabled(False)

        # Set db file to the loaded and disable the field
        main_window.db_file_name_input.setText(main_window.ma_file_path)
        main_window.db_file_name_input.setEnabled(False)

        # Enable the button to plot samples
        main_window.plot_samples_button.setEnabled(True)

        # Reset the qois
        main_window.qoi_funcs = {}
        main_window.df_qois = pd.DataFrame()
        
        # print a message in the output fields of Tab 2
        message = f"Database file loaded: {main_window.ma_file_path}"
        main_window.update_output_tab2(main_window, message)
    except Exception as e:
        error_message = f"Error loading .db file: {e} (Type: {type(e).__name__})"
        # Add traceback for more technical details
        error_message += f"\nTraceback: {traceback.format_exc()}"
        print(error_message)
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
            main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [1, 10, 20]}

        for key, value in main_window.analysis_rules_parameters.items():
            ref_value = main_window.get_rule_value(main_window, key)  # Get the default rule value
            main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [1, 10, 20]}

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
            # print(f"Update Analysis type - {key}: {ref_value}")
            main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 20.0, "lower_bounds": float(ref_value) * 0.8, "upper_bounds": float(ref_value) * 1.2}

        for key, value in main_window.analysis_rules_parameters.items():
            ref_value = main_window.get_rule_value(main_window, key)  # Get the default rule value
            # print(f"Update Analysis type - {key}: {ref_value}")
            main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 20.0, "lower_bounds": float(ref_value) * 0.8, "upper_bounds": float(ref_value) * 1.2}

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
        main_window.local_ref_value_input.setText(str(param_data["ref_value"]))
        main_window.local_perturb_input.setText(",".join(map(str, param_data["perturbation"])))

        # Connect signals to update local_SA_parameters when editing is finished
        main_window.local_ref_value_input.editingFinished.connect(lambda: main_window.update_local_SA_reference(main_window))
        main_window.local_perturb_input.editingFinished.connect(lambda: main_window.update_local_SA_perturbations(main_window))

def update_local_SA_reference(main_window):
    # Update the reference value in local_SA_parameters
    selected_param = main_window.local_param_combo.currentText()
    if selected_param in main_window.local_SA_parameters:
        try:
            new_ref_value = float(main_window.local_ref_value_input.text())
            main_window.local_SA_parameters[selected_param]["ref_value"] = new_ref_value
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid reference value.")

def update_local_SA_perturbations(main_window):
    # Update the perturbations in local_SA_parameters
    selected_param = main_window.local_param_combo.currentText()
    if selected_param in main_window.local_SA_parameters:
        try:
            new_perturbations = [float(p) for p in main_window.local_perturb_input.text().split(",")]
            main_window.local_SA_parameters[selected_param]["perturbation"] = new_perturbations
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid perturbation values.")

def update_global_inputs(main_window):
    # Update the reference value, range percentage, and bounds based on the selected parameter
    selected_param = main_window.global_param_combo.currentText()
    if selected_param in main_window.global_SA_parameters:
        param_data = main_window.global_SA_parameters[selected_param]
        main_window.global_ref_value_input.setText(str(param_data["ref_value"]))
        main_window.global_range_percentage.setText(str(param_data["perturbation"]))

        try:
            range_percentage = param_data["perturbation"]
            lower_bound = param_data["ref_value"] * (1 - range_percentage / 100)
            upper_bound = param_data["ref_value"] * (1 + range_percentage / 100)
            main_window.global_bounds.setText(f"{lower_bound:.3e}, {upper_bound:.3e}")
            # Store bounds in global_SA_parameters
            main_window.global_SA_parameters[selected_param]["lower_bounds"] = lower_bound
            main_window.global_SA_parameters[selected_param]["upper_bounds"] = upper_bound
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
            main_window.global_SA_parameters[selected_param]["ref_value"] = new_ref_value
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
        main_window.update_output_tab2(main_window, f"Sampling parameters from One-At-A-Time approach...")
        # Check if samples already exist, if so, delete them
        if 'samples' in main_window.local_SA_parameters.keys():
            del main_window.local_SA_parameters["samples"]
        # Run the local sampler
        try:
            main_window.local_SA_parameters["samples"] = run_local_sampler(main_window.local_SA_parameters)
        except ValueError as e:
            main_window.update_output_tab2(main_window, f"Error in local sampler: {e}")
            return
    elif analysis_type == "Global":
        sampler = main_window.global_sampler_combo.currentText()
        main_window.update_output_tab2(main_window, f"Sampling parameters from sampler: {sampler}...")
        # Check if samples already exist, if so, delete them
        if 'samples' in main_window.global_SA_parameters.keys():
            del main_window.global_SA_parameters["samples"]
        # Run the global sampler based on the selected sampler
        try:
            # Request number of samples from the user to run the global sampler
            if sampler not in ['Fractional Factorial', 'Finite Difference']:
                N, ok = QInputDialog.getInt(main_window, "Number of Samples", "Enter the desired number of samples:", value=8, min=1)
            else:
                N = None  # For Fractional Factorial and Finite Difference, N is not required
            main_window.global_SA_parameters["samples"] = run_global_sampler(main_window.global_SA_parameters, sampler, N=N)
        except ValueError as e:
            main_window.update_output_tab2(main_window, f"Error in global sampler: {e}")
            return

    # Make the define_qoi, DB file name field and "Run Simulations" button visible
    main_window.db_file_name_input.setEnabled(True)
    main_window.define_qoi_button.setEnabled(True)
    main_window.run_simulations_button.setEnabled(True)

def plot_samples(main_window):
    # Create a dialog window for plotting samples
    plot_samples_window = QDialog(main_window)
    plot_samples_window.setWindowTitle("Parameter Samples")
    plot_samples_window.setGeometry(100, 100, 800, 600)
    layout = QVBoxLayout(plot_samples_window)

    # Add info label
    info_label = QLabel("Parameter Sampling Visualization")
    info_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(info_label)

    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(6, 4))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)

    def update_plot():
        figure.clear()
        ax = figure.add_subplot(111)
        try:
            if main_window.analysis_type_dropdown.currentText() == "Local":
                perturbations_df = pd.DataFrame(main_window.local_SA_parameters["samples"]).T
                for col in perturbations_df.columns:
                    perturbations_df[col] = 100.0 * (
                        perturbations_df[col] - main_window.local_SA_parameters[col]["ref_value"]
                    ) / main_window.local_SA_parameters[col]["ref_value"]
                perturbations_df = perturbations_df.reset_index().melt(
                    id_vars="index", var_name="Parameter", value_name="Perturbation"
                )
                perturbations_df = perturbations_df.rename(columns={"index": "SampleID"})
                perturbations_df['Frequency'] = perturbations_df.groupby(
                    ['Perturbation', 'Parameter']
                )['SampleID'].transform('count')
                scatter_plot = sns.scatterplot(
                    data=perturbations_df,
                    x="Perturbation",
                    y="Parameter",
                    hue="SampleID",
                    size="Frequency",
                    palette="viridis",
                    ax=ax,
                    alpha=0.7,
                )
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                scatter_plot.set_title(
                    f"One-At-A-Time (OAT) sampling - Total Samples: {len(main_window.local_SA_parameters['samples'])}",
                    fontsize=8,
                )
                scatter_plot.set_xlabel("Perturbations (%)", fontsize=8)
                scatter_plot.set_ylabel("")
                handles, labels = scatter_plot.get_legend_handles_labels()
                new_handles = [h for h, l in zip(handles, labels) if l not in ["SampleID", "Frequency"]]
                new_labels = [l for l in labels if l not in ["SampleID", "Frequency"]]
                scatter_plot.legend(
                    handles=new_handles,
                    labels=new_labels,
                    title="Sample Index",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    title_fontsize=8,
                    fontsize=8,
                    ncol=2
                )
            elif main_window.analysis_type_dropdown.currentText() == "Global":
                normalized_df = pd.DataFrame(main_window.global_SA_parameters["samples"]).T
                for col in normalized_df.columns:
                    normalized_df[col] = (
                        normalized_df[col] - main_window.global_SA_parameters[col]["lower_bounds"]
                    ) / (
                        main_window.global_SA_parameters[col]["upper_bounds"] - main_window.global_SA_parameters[col]["lower_bounds"]
                    )
                normalized_df = normalized_df.reset_index().melt(
                    id_vars="index", var_name="Parameter", value_name="Normalized Value"
                )
                scatter_plot = sns.scatterplot(
                    data=normalized_df,
                    x="Normalized Value",
                    y="Parameter",
                    hue="index",
                    palette="viridis",
                    s=50,
                    ax=ax,
                    alpha=0.7,
                )
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                scatter_plot.set_title(
                    f"Global Sampling - Total Samples: {len(main_window.global_SA_parameters['samples'])}",
                    fontsize=8,
                )
                scatter_plot.set_xlabel("")
                scatter_plot.set_ylabel("")
                scatter_plot.legend(
                    title="Sample Index",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    title_fontsize=8,
                    fontsize=8,
                )
            figure.set_constrained_layout(True)
            canvas.draw()
        except Exception as e:
            QMessageBox.warning(plot_samples_window, "Plot Error", f"Error plotting samples: {e}")

    update_plot()

    # Add a close button
    close_button = QPushButton("Close")
    close_button.setStyleSheet("background-color: lightgreen; color: black")
    close_button.clicked.connect(plot_samples_window.accept)
    layout.addWidget(close_button)

    plot_samples_window.exec_()

def update_global_sampler_options(main_window):
    # FAST - Fourier Amplitude Sensitivity Test: combatible with Fast sampler
    # RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test: combatible with all samplers
    # Sobol’ Sensitivity Analysis: combatible with Sobol samplers
    # Delta Moment-Independent Measure: combatible with all samplers
    # Derivative-based Global Sensitivity Measure (DGSM): combatible with Finite Difference sampler
    # Fractional Factorial: combatible with Fractional Factorial sampler
    # PAWN Sensitivity Analysis: combatible with all samplers
    # High-Dimensional Model Representation: combatible with all samplers
    # Regional Sensitivity Analysis: combatible with all samplers
    # Discrepancy Sensitivity Indices: combatible with all samplers
    # Update the global_sampler_combo options based on the selected method
    # Avoid unnecessary updates if the combo box is disabled - special case when sampler loaded from .db file will update the method list
    if not main_window.global_sampler_combo.isEnabled(): return  
    method = main_window.global_method_combo.currentText()
    main_window.global_sampler_combo.clear()

    if method == "FAST - Fourier Amplitude Sensitivity Test":
        main_window.global_sampler_combo.addItems(["Fast"])
    elif method == "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test":
        main_window.global_sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol"
        ])
    elif method == "Sobol Sensitivity Analysis":
        main_window.global_sampler_combo.addItems(["Sobol"])
    elif method == "Delta Moment-Independent Measure":
        main_window.global_sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol"
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
            "Latin hypercube sampling (LHS)", "Sobol"
        ])
    else:
        main_window.global_sampler_combo.addItems([])  # No compatible samplers  

def run_simulations_function(main_window):
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
    except Exception as e:
        main_window.update_output_tab2(main_window, f"Error setting up simulations: {e}")
        print(f"Error setting up simulations: {e}")
        return

    try:
        # Update the output tab with the current status
        main_window.update_output_tab2(main_window, f"Running simulations with sampler: {SA_sampler} and number of samples: {len(SA_samples)}")
        model_config = {"ini_path": main_window.ini_file_path, "struc_name": main_window.struc_name_input.text().strip()}
        # Determine the samples to use according to the analysis type
        SA_type = main_window.analysis_type_dropdown.currentText()
        # Simulatethe model with the selected samples
        if "Local" == SA_type:
            sampler = "OAT"
            # Model Analysis context
            context = ModelAnalysisContext(db_file_name, model_config, sampler, main_window.local_SA_parameters, qoi_str, num_workers=int(num_workers))
            run_simulations(context)
        elif "Global" == SA_type:
            sampler = main_window.global_sampler_combo.currentText()
            # Model Analysis context
            context = ModelAnalysisContext(db_file_name, model_config, sampler, main_window.global_SA_parameters, qoi_str, num_workers=int(num_workers))
            run_simulations(context)
    except Exception as e:
        main_window.update_output_tab2(main_window, f"Error: Running simulations with {sampler} failed: {e}")
        print(f"Error: Running simulations with {sampler} failed: {e}")
        return

    # Simulate saving the samples to the database (replace with actual simulation logic)
    main_window.update_output_tab2(main_window, f"Simulations completed and saved to {db_file_name} with QoIs: {', '.join(main_window.qoi_funcs.keys())}.")
    # Load the database file to display results
    main_window.load_db_file(main_window, db_file_name)
    

def plot_qois(main_window):
    main_window.update_output_tab2(main_window, "Plotting QoIs...")
    # Create a dialog window for plotting QoIs
    plot_qoi_window = QDialog(main_window)
    plot_qoi_window.setWindowTitle("Plot QoIs")
    plot_qoi_window.setGeometry(100, 100, 800, 600)
    # Create a layout for the dialog
    layout = QVBoxLayout(plot_qoi_window)
    # Add input for the QoI to plot
    plot_qoi_hbox = QHBoxLayout()
    plot_qoi_label = QLabel("Select QoI to plot:")
    plot_qoi_combo = QComboBox()
    plot_qoi_combo.addItems(list(main_window.qoi_funcs.keys()))
    plot_qoi_hbox.addWidget(plot_qoi_label)
    plot_qoi_hbox.addWidget(plot_qoi_combo)
    layout.addLayout(plot_qoi_hbox)
    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(5, 3))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)
    # Calculate the QoIs if not already done
    if main_window.df_qois.empty:
        try: main_window.df_qois = calculate_qoi_statistics(main_window.df_output, main_window.qoi_funcs, db_file_path = main_window.db_file_name_input.text().strip())
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error calculating QoIs: {e}")
            return

    def update_plot_qoi():
        # Clear the previous plot
        figure.clear()
        ax = figure.add_subplot(111)
        selected_qoi = plot_qoi_combo.currentText()
        # Plot the selected QoI
        if selected_qoi in main_window.qoi_funcs.keys():
            qoi_columns = sorted([col for col in main_window.df_qois.columns if col.startswith(selected_qoi)])
            time_columns = sorted([col for col in main_window.df_qois.columns if col.startswith("time_")])
            # Prepare the data for seaborn
            plot_data = pd.DataFrame({
                "Time": main_window.df_qois[time_columns].values.flatten(),
                selected_qoi: main_window.df_qois[qoi_columns].values.flatten(),
                "SampleID": main_window.df_qois.index.repeat(len(qoi_columns))
            })
            # Plot using seaborn
            sns.lineplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(selected_qoi)
            ax.legend(title="Sample Index")
            canvas.draw()
        else:
            main_window.update_output_tab2(main_window, f"Error: {selected_qoi} not found in the output data.")
        # Adjust layout and draw the canvas
        # figure.tight_layout()
        figure.set_constrained_layout(True)
        canvas.draw()

    # Connect the combo box to update the plot
    plot_qoi_combo.currentIndexChanged.connect(update_plot_qoi)
    # Set the default selected qoi and update the plot
    plot_qoi_combo.setCurrentIndex(0)
    update_plot_qoi()
    # Show the dialog
    plot_qoi_window.exec_()

def run_analysis(main_window):
    main_window.update_output_tab2(main_window, "Running sensitivity analysis...")
     # Calculate the QoIs if not already done
    if main_window.df_qois.empty:
        try: main_window.df_qois = calculate_qoi_statistics(main_window.df_output, main_window.qoi_funcs, db_file_path = main_window.db_file_name_input.text().strip())
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error calculating QoIs: {e}")
            return
    # Prepare the QoIs for analysis
    all_qois_names = list(main_window.qoi_funcs.keys())
    all_times_label = [col for col in main_window.df_qois.columns if col.startswith("time")]
    print(f"all_qois: {all_qois_names} and all_times: {all_times_label}")
    if main_window.analysis_type_dropdown.currentText() == "Global":
        global_method = main_window.global_method_combo.currentText()
        try:
            main_window.sa_results, main_window.qoi_time_values = run_global_sa(main_window.global_SA_parameters, global_method, all_times_label, all_qois_names, main_window.df_qois)
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error running global sensitivity analysis: {e}")
            return

        # Plot the results
        main_window.update_output_tab2(main_window, f"Plotting results for {global_method}")
        main_window.plot_sa_results(main_window)
    elif main_window.analysis_type_dropdown.currentText() == "Local":
        try:
            main_window.sa_results, main_window.qoi_time_values = run_local_sa(main_window.local_SA_parameters, all_times_label, all_qois_names, main_window.df_qois)
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error running local sensitivity analysis: {e}")
            return
       
        # Plot the results
        main_window.update_output_tab2(main_window, f"Plotting results for OAT SA.")
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
    plot_sa_SI_hbox = QHBoxLayout()
    plot_sa_SI_label = QLabel("Select sensitivity measurement to plot:")
    plot_sa_SI_label.setAlignment(Qt.AlignCenter)
    plot_sa_SI_hbox.addWidget(plot_sa_SI_label)
    plot_sa_SI_dropdown = QComboBox(plot_window)
    plot_sa_SI_dropdown.setEditable(False)
    if main_window.analysis_type_dropdown.currentText() == "Global":
        qoi_label = list(main_window.sa_results.keys())[-1]
        time_label = list(main_window.sa_results[qoi_label].keys())[-1]
        sensitivity_measurements = list(main_window.sa_results[qoi_label][time_label].keys())
        # Remove 'names' and 'target' from the list
        if 'names' in sensitivity_measurements: sensitivity_measurements.remove('names') # All methods
        if 'target' in sensitivity_measurements: sensitivity_measurements.remove('target') # for Regional SA method
    else:
        sensitivity_measurements = ["SI"]
        plot_sa_SI_dropdown.setEnabled(False)
    plot_sa_SI_dropdown.addItems(sensitivity_measurements)
    plot_sa_SI_hbox.addWidget(plot_sa_SI_dropdown)
    layout.addLayout(plot_sa_SI_hbox)
    # print(f"SI: {main_window.sa_results[ list(main_window.sa_results.keys())[0] ][ list(main_window.qoi_time_values)[0] ].values()}")
    # print(f"QoI: {list(main_window.sa_results.keys())}")
    # print(f"Times: {list(main_window.qoi_time_values.values())}")
    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(5, 3))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)
    # Define the update_plot function before connecting it
    def update_plot():
         # Clear the previous plot
        figure.clear()
        ax = figure.add_subplot(111)
        # Get the selected qoi
        selected_qoi = plot_sa_dropdown_qoi.currentText()
        selected_sm = plot_sa_SI_dropdown.currentText()
        # Clear the previous plot
        print(f"Plotting {selected_sm} of SA from {selected_qoi}.")
        try:
            if main_window.analysis_type_dropdown.currentText() == "Global":
                SA_method = main_window.global_method_combo.currentText()
                # This is necessary because Sobol method does not return the names of the parameters
                param_names = [key for key in main_window.global_SA_parameters.keys() if key != "samples"]
                plot_data = pd.DataFrame([
                    {
                        "Time": main_window.qoi_time_values[time_label],
                        "Sensitivity Index": main_window.sa_results[selected_qoi][time_label][selected_sm][param_id],
                        "Parameter": param
                    }
                    for time_label in main_window.sa_results[selected_qoi].keys()
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
                sns.lineplot(data=plot_data, x="Time", y="Sensitivity Index", hue="Parameter", ax=ax, palette=custom_palette, hue_order=parameter_order)                
                ax.set_xlabel("Time (min)")
                ax.set_ylabel(f"Sensitivity Measure ({selected_sm})")
                ax.set_title(f"Global SA - {SA_method}", fontsize=8)
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)
            elif main_window.analysis_type_dropdown.currentText() == "Local":
                SA_method = "OAT method"
                # Prepare the data for seaborn
                plot_data = pd.DataFrame([
                    {
                        "Time": main_window.qoi_time_values[time_label],
                        "Sensitivity Index": main_window.sa_results[selected_qoi][time_label][param],
                        "Parameter": param
                    }
                    for time_label in main_window.sa_results[selected_qoi].keys()
                    for param in main_window.sa_results[selected_qoi][time_label].keys()
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
                ax.set_title(f"Local SA - {SA_method}")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)
            # Adjust layout and draw the canvas
            # figure.tight_layout()
            figure.set_constrained_layout(True)
            canvas.draw()
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error plotting SA results: {e}")
            return

    # Connect the combo box to update the plot
    plot_sa_dropdown_qoi.currentIndexChanged.connect(update_plot)
    plot_sa_SI_dropdown.currentIndexChanged.connect(update_plot)

    # Set the default selected qoi and update the plot
    plot_sa_dropdown_qoi.setCurrentIndex(0)
    update_plot()

    # Show the dialog
    plot_window.exec_()
