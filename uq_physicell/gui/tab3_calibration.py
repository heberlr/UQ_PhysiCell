import pandas as pd
import seaborn as sns
import logging
import sys

# All the specific classes we need
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QDialog, QFileDialog, QListWidget, QSpacerItem, QSizePolicy, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from threading import Thread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# My local modules
from ..gui.tab2_model_analysis import QtTextEditLogHandler
from ..database.bo_db import load_structure
from ..bo.bo_context import CalibrationContext, run_bayesian_optimization
from ..bo.plots import (
    get_observed_qoi, plot_parameter_space, plot_qoi_param, plot_parameter_vs_fitness
)
from ..bo.utils import analyze_pareto_results


def on_run_calibration_clicked(main_window):
    """Toggle handler for run/cancel calibration button. Shows options dialog, starts
    a background worker thread for calibration, or requests cancellation if running.
    """
    # If a calibration thread is running, request cancellation
    if getattr(main_window, 'calibration_thread', None) and main_window.calibration_thread.is_alive():
        main_window.calibration_cancelled = True
        # If a CalibrationContext exists for the running job, set its cancel flag as well
        if hasattr(main_window, 'calib_context') and main_window.calib_context is not None:
            try:
                main_window.calib_context.cancel_requested = True
            except Exception:
                pass
        if hasattr(main_window, 'post_message'):
            main_window.post_message('tab3', "Calibration cancellation requested.")
        else:
            main_window.output_text_tab3.append("Calibration cancellation requested.")
        return

    # Otherwise display options dialog and start calibration
    class BOOptionsDialog(QDialog):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Set Bayesian Optimization Options")
            self.setMinimumSize(350, 200)
            layout = QVBoxLayout()
            # Background info
            xml_path_omp_threads = ".//parallel/omp_num_threads"
            try:
                num_replicates = main_window.num_replicates_input.text().strip()
            except Exception:
                num_replicates = "1"
            try:
                omp_threads = main_window.xml_tree.find(xml_path_omp_threads).text.strip()
            except Exception:
                omp_threads = "1"
            total_threads_label = QLabel(f"OpenMP Threads per Model: {omp_threads}")
            layout.addWidget(total_threads_label)
            number_of_replicates_label = QLabel(f"Number of Replicates: {num_replicates}")
            layout.addWidget(number_of_replicates_label)
            # Separator line
            layout.addWidget(QLabel("<hr>"))

            # num_initial_samples
            hbox1 = QHBoxLayout()
            hbox1.addWidget(QLabel("Initial samples:"))
            self.initial_samples_input = QLineEdit("5")
            hbox1.addWidget(self.initial_samples_input)
            layout.addLayout(hbox1)
            # num_iterations
            hbox2 = QHBoxLayout()
            hbox2.addWidget(QLabel("Iterations:"))
            self.iterations_input = QLineEdit("10")
            hbox2.addWidget(self.iterations_input)
            layout.addLayout(hbox2)
            # max_workers
            hbox3 = QHBoxLayout()
            hbox3.addWidget(QLabel("Max workers:"))
            self.max_workers_input = QLineEdit("4")
            hbox3.addWidget(self.max_workers_input)
            layout.addLayout(hbox3)
            # use_exponential_fitness
            hbox4 = QHBoxLayout()
            hbox4.addWidget(QLabel("Use exponential fitness:"))
            self.exp_fitness_combo = QComboBox()
            self.exp_fitness_combo.addItems(["True", "False"])
            hbox4.addWidget(self.exp_fitness_combo)
            layout.addLayout(hbox4)
            # OK/Cancel
            btn_hbox = QHBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            btn_hbox.addWidget(ok_btn)
            btn_hbox.addWidget(cancel_btn)
            layout.addLayout(btn_hbox)
            self.setLayout(layout)
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)

        def get_options(self):
            return {
                "num_initial_samples": int(self.initial_samples_input.text()),
                "num_iterations": int(self.iterations_input.text()),
                "max_workers": int(self.max_workers_input.text()),
                "use_exponential_fitness": self.exp_fitness_combo.currentText() == "True"
            }

    dialog = BOOptionsDialog(main_window)
    if dialog.exec_() != QDialog.Accepted:
        main_window.output_text_tab3.append("Calibration cancelled.")
        return

    bo_options = dialog.get_options()
    main_window.output_text_tab3.append(f"Starting calibration with options: {bo_options}")

    # Start the background worker
    main_window.calibration_cancelled = False
    worker = Thread(target=run_calibration_worker, args=(main_window, bo_options), daemon=True)
    main_window.calibration_thread = worker
    worker.start()

    # update UI to show cancel state
    try:
        main_window.run_calibration_button.setText("Cancel Calibration")
        main_window.run_calibration_button.setStyleSheet("background-color: salmon; color: black")
    except Exception:
        pass

    # Poll for completion and restore UI
    if getattr(main_window, '_calib_completion_timer', None) is None:
        timer = QTimer(main_window)
        timer.setInterval(500)

        def _poll():
            if not getattr(main_window, 'calibration_thread', None) or not main_window.calibration_thread.is_alive():
                timer.stop()
                main_window._calib_completion_timer = None
                try:
                    main_window.run_calibration_button.setText("Run Calibration")
                    main_window.run_calibration_button.setStyleSheet("background-color: lightgreen; color: black")
                    # enable plot results if a BO file path exists
                    if getattr(main_window, 'bo_file_path', None):
                        main_window.plot_calibration_button.setEnabled(True)
                except Exception:
                    pass

        timer.timeout.connect(_poll)
        main_window._calib_completion_timer = timer
        timer.start()


def run_calibration_worker(main_window, bo_options):
    """Background worker that performs the Bayesian Optimization.
    UI updates that must run on the main thread are scheduled via QTimer.singleShot.
    """
    try:
        # Set up logging handlers that write to the GUI
        gui_handler = QtTextEditLogHandler(main_window.output_text_tab3)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        main_window.logger_tab3 = logging.getLogger(__name__ + '.calib')
        main_window.logger_tab3.setLevel(logging.INFO)
        main_window.logger_tab3.handlers = [gui_handler, console_handler]

        # Build search space and fixed params
        search_space = {}
        fixed_params = {}
        for param_name in main_window.df_param_space['Parameter']:
            try:
                fixed_val = main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, "Fixed"].values[0]
            except Exception:
                fixed_val = None
            if pd.notna(fixed_val):
                fixed_params[param_name] = fixed_val
            else:
                search_space[param_name] = {
                    "type": main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, "Type"].values[0],
                    "lower_bound": main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, "Lower Bound"].values[0],
                    "upper_bound": main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, "Upper Bound"].values[0]
                }

        main_window.calib_context = CalibrationContext(
            db_path=main_window.calibration_file_input.text(),
            obsData=main_window.obs_file_name_input.text(),
            obsData_columns={**dict(zip(main_window.df_qois['QoI_Name'], main_window.df_qois['ObsData_Column'])), 'time': 'Time'},
            model_config={"ini_path": main_window.ini_file_path, "struc_name": main_window.struc_name_input.text().strip(),
                          "numReplicates": int(main_window.num_replicates_input.text().strip())},
            qoi_functions=main_window.qoi_funcs,
            distance_functions={qoi_name: {"function": main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, "QoI_distanceFunction"].values[0],
                                           "weight": main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, "QoI_distanceWeight"].values[0]} for qoi_name in main_window.df_qois['QoI_Name']},
            search_space=search_space,
            bo_options=bo_options,
            logger=main_window.logger_tab3
        )
        if fixed_params:
            main_window.calib_context.fixed_params = fixed_params

        # Quick cancellation check before starting
        if getattr(main_window, 'calibration_cancelled', False):
            main_window.logger_tab3.info('Calibration cancelled before start')
            return

        # Run the heavy Bayesian Optimization routine (may be long-running)
        try:
            run_bayesian_optimization(main_window.calib_context)
        except Exception as e:
            main_window.logger_tab3.error(f"Error occurred during calibration: {e}")

        # Schedule GUI updates on the main thread after completion
        def _on_done():
            try:
                main_window.bo_file_path = main_window.calibration_file_input.text()
                main_window.load_bo_database(main_window)
            except Exception as e:
                # Best effort to notify the user
                try:
                    main_window.output_text_tab3.append(f"Error loading Bayesian Optimization results: {e}")
                except Exception:
                    pass
            try:
                main_window.run_calibration_button.setText("Run Calibration")
                main_window.run_calibration_button.setStyleSheet("background-color: lightgreen; color: black")
                main_window.plot_calibration_button.setEnabled(True)
            except Exception:
                pass

        QTimer.singleShot(0, _on_done)

    finally:
        # attempt to remove handlers to avoid duplicate logs on repeated runs
        try:
            main_window.logger_tab3.handlers = []
        except Exception:
            pass


def create_tab3(main_window):
    # Add methods to the main_window instance
    main_window.load_ini_calibration = load_ini_calibration
    main_window.load_obs_data = load_obs_data
    main_window.load_bo_database = load_bo_database
    main_window.define_parameter_space = define_parameter_space
    main_window.define_qois = define_qois
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
    main_window.load_ini_button_tab3.clicked.connect(lambda: main_window.load_ini_calibration(main_window))
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
    hbox_inout_tab3.addWidget(QLabel("Parameter(s):"))
    main_window.define_param_button_tab3 = QPushButton("Define Input Space")
    main_window.define_param_button_tab3.setStyleSheet("background-color: lightgreen; color: black")
    # main_window.define_param_button_tab3.setEnabled(False)
    main_window.define_param_button_tab3.clicked.connect(lambda: main_window.define_parameter_space(main_window))
    hbox_inout_tab3.addWidget(main_window.define_param_button_tab3)
    # Add a horizontal spacer (scape)
    hbox_inout_tab3.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))
    # Define QoIs
    hbox_inout_tab3.addWidget(QLabel("QoI(s) and Obs. Data:"))
    main_window.define_qoi_button_tab3 = QPushButton("Define Output Space")
    # main_window.define_qoi_button_tab3.setEnabled(False)
    main_window.define_qoi_button_tab3.setStyleSheet("background-color: lightgreen; color: black")
    main_window.define_qoi_button_tab3.clicked.connect(lambda: main_window.define_qois(main_window))
    hbox_inout_tab3.addWidget(main_window.define_qoi_button_tab3)
    # Add a horizontal spacer (scape)
    hbox_inout_tab3.addStretch() # push everything to the left
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

    # File name for calibration results
    hbox_calib_tab3.addWidget(QLabel("Calibration Results File:"))
    # LineEdit for file name
    main_window.calibration_file_input = QLineEdit()
    main_window.calibration_file_input.setPlaceholderText("Enter file name to store calibration results")
    main_window.calibration_file_input.setEnabled(True)
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
    main_window.run_calibration_button.setEnabled(True)
    # initialize calibration threading state
    main_window.calibration_thread = None
    main_window.calibration_cancelled = False
    main_window._calib_completion_timer = None
    # Connect to a run/cancel handler (toggles between starting and requesting cancellation)
    main_window.run_calibration_button.clicked.connect(lambda: on_run_calibration_clicked(main_window))
    calibration_buttons_hbox.addWidget(main_window.run_calibration_button)

    main_window.plot_calibration_button = QPushButton("Plot Results")
    main_window.plot_calibration_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_calibration_button.setEnabled(False)
    main_window.plot_calibration_button.clicked.connect(lambda: main_window.plot_calibration_results(main_window))
    calibration_buttons_hbox.addWidget(main_window.plot_calibration_button)

    layout_tab3.addLayout(calibration_buttons_hbox)

    ###########################################
    # Display section
    ###########################################
    main_window.output_label_tab3 = QLabel("<b>Display</b>")
    main_window.output_label_tab3.setAlignment(Qt.AlignCenter)
    layout_tab3.addWidget(main_window.output_label_tab3)

    main_window.output_text_tab3 = QTextEdit()
    main_window.output_text_tab3.setReadOnly(True)
    main_window.output_text_tab3.setMinimumHeight(100)
    layout_tab3.addWidget(main_window.output_text_tab3)
    
    # Register with main window's message system if available
    if hasattr(main_window, 'add_output_widget'):
        main_window.add_output_widget('tab3', main_window.output_text_tab3)
        main_window.post_message('tab3', "Welcome to Calibration! Load a model and observational data, set parameters, and Quantity(s) of Interest (QoIs) to begin.")
    
    return layout_tab3


def load_obs_data(main_window):
    # Load observational data from a .csv file
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getOpenFileName(main_window, "Select Observational Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    if file_path:
        try:
            main_window.obs_data = pd.read_csv(file_path)
            main_window.obs_file_name_input.setText(file_path)
            # Thread-safe update via main window
            if hasattr(main_window, 'post_message'):
                main_window.post_message('tab3', f"Loaded observational data from {file_path}")
            else:
                main_window.output_text_tab3.append(f"Loaded observational data from {file_path}")
        except Exception as e:
            if hasattr(main_window, 'post_message'):
                main_window.post_message('tab3', f"Error loading observational data: {e}")
            else:
                main_window.output_text_tab3.append(f"Error loading observational data: {e}")

def load_ini_calibration(main_window):
    try: 
        main_window.load_ini_file(main_window)
    except Exception as e:
        main_window.output_text_tab3.append(e)
    # Switch the selected tab to Tab 3
    main_window.tabs.setCurrentIndex(2)
    # Update the ini_file_name_input
    main_window.ini_file_name_input.setText(main_window.ini_file_path)
    # Clean up the parameter space
    main_window.df_param_space = pd.DataFrame()

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

def define_parameter_space(main_window):
    if not main_window.analysis_parameters and not main_window.analysis_rules_parameters:
        QMessageBox.warning(main_window, "Missing model structure", "Please load the model structure (.ini file) with the appropriate parameters before defining the input space for calibration.")
        return
    # Function to overwrite parameter in the DataFrame
    def overwrite_parameter():
        param_name = param_combo.currentText()
        param_type = param_type_combo.currentText()
        lower_bound = param_lower_bound_input.text()
        upper_bound = param_upper_bound_input.text()
        fixed_value = param_fixed_input.text()

        # Overwrite parameter
        main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Type'] = param_type
        if lower_bound != "":
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Lower Bound'] = float(lower_bound)
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Upper Bound'] = float(upper_bound)
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Fixed'] = None
        elif fixed_value != "":
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Fixed'] = float(fixed_value)
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Lower Bound'] = None
            main_window.df_param_space.loc[main_window.df_param_space['Parameter'] == param_name, 'Upper Bound'] = None
        else:
            QMessageBox.warning(main_window, "Input Error", "Please provide either lower and upper bounds or a fixed value for the parameter.")
            return

        # Update the table display
        update_param_table()
    
    # Function to update the table with current parameter space data
    def update_param_table():
        if main_window.df_param_space is not None and not main_window.df_param_space.empty:
            param_table.setRowCount(len(main_window.df_param_space))
            for i, row in main_window.df_param_space.iterrows():
                param_table.setItem(i, 0, QTableWidgetItem(str(row['Parameter'])))
                param_table.setItem(i, 1, QTableWidgetItem(str(row['Type'])))
                param_table.setItem(i, 2, QTableWidgetItem(str(row['Lower Bound'])))
                param_table.setItem(i, 3, QTableWidgetItem(str(row['Upper Bound'])))
                param_table.setItem(i, 4, QTableWidgetItem(str(row['Fixed'])))
        else:
            param_table.setRowCount(0)

    # Define the parameter space for the calibration
    param_space_window = QDialog(main_window)
    param_space_window.setWindowTitle("Define Parameter Space")
    param_space_window.setMinimumSize(600, 400)
    layout = QVBoxLayout()

    # Parameters ComboBox
    parameter_name_hbox = QHBoxLayout()
    param_label = QLabel("Parameter:")
    parameter_name_hbox.addWidget(param_label)
    param_combo = QComboBox()
    parameter_name_hbox.addWidget(param_combo)
    layout.addLayout(parameter_name_hbox)

    # Parameter Type
    param_type_hbox = QHBoxLayout()
    param_type_label = QLabel("Type:")
    param_type_hbox.addWidget(param_type_label)
    param_type_combo = QComboBox()
    param_type_combo.addItems(["real", "integer"])
    param_type_hbox.addWidget(param_type_combo)
    layout.addLayout(param_type_hbox)

    # Lower_Bound
    param_lower_bound_hbox = QHBoxLayout()
    param_lower_bound_label = QLabel("Lower Bound:")
    param_lower_bound_hbox.addWidget(param_lower_bound_label)
    param_lower_bound_input = QLineEdit()
    param_lower_bound_hbox.addWidget(param_lower_bound_input)
    layout.addLayout(param_lower_bound_hbox)

    # Upper_Bound
    param_upper_bound_hbox = QHBoxLayout()
    param_upper_bound_label = QLabel("Upper Bound:")
    param_upper_bound_hbox.addWidget(param_upper_bound_label)
    param_upper_bound_input = QLineEdit()
    param_upper_bound_hbox.addWidget(param_upper_bound_input)
    layout.addLayout(param_upper_bound_hbox)

    # Fixed Parameter
    param_fixed_hbox = QHBoxLayout()
    param_fixed_label = QLabel("Fixed:")
    param_fixed_hbox.addWidget(param_fixed_label)
    param_fixed_input = QLineEdit()
    param_fixed_hbox.addWidget(param_fixed_input)
    layout.addLayout(param_fixed_hbox)

    # Button Overwrite
    add_param_button = QPushButton("Overwrite Parameter")
    layout.addWidget(add_param_button)
    add_param_button.clicked.connect(overwrite_parameter)

    # Table view
    param_table = QTableWidget()
    param_table.setColumnCount(5)
    param_table.setHorizontalHeaderLabels(["Parameter", "Type", "Lower Bound", "Upper Bound", "Fixed"])
    param_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table non-editable
    layout.addWidget(param_table)

    # Initialize df_param_space if it doesn't exist
    if not hasattr(main_window, 'df_param_space'):
        main_window.df_param_space = pd.DataFrame(columns=['Parameter', 'Type', 'Lower Bound', 'Upper Bound', 'Fixed'])
        # Explicitly set dtypes to prevent future warnings
        main_window.df_param_space = main_window.df_param_space.astype({
            'Parameter': 'object',
            'Type': 'object', 
            'Lower Bound': 'float64',
            'Upper Bound': 'float64',
            'Fixed': 'float64'
        })
    
    # Initialize upper and lower bound inputs
    if main_window.df_param_space.empty:
        # From XML
        for key, value in main_window.analysis_parameters.items():
            ref_value = float(main_window.get_parameter_value_xml(main_window, key))  # Get the default XML value - string
            lower_bound = float(ref_value) * 0.8
            upper_bound = float(ref_value) * 1.2
            new_row = pd.DataFrame([{'Parameter': value[1], 'Type': 'real', 'Lower Bound': lower_bound, 'Upper Bound': upper_bound, 'Fixed': None}])
            main_window.df_param_space = pd.concat([main_window.df_param_space, new_row], ignore_index=True)
        # From rules
        for key, value in main_window.analysis_rules_parameters.items():
            ref_value = float(main_window.get_rule_value(main_window, key))  # Get the default rule value - string
            lower_bound = float(ref_value) * 0.8
            upper_bound = float(ref_value) * 1.2
            new_row = pd.DataFrame([{'Parameter': value[1], 'Type': 'real', 'Lower Bound': lower_bound, 'Upper Bound': upper_bound, 'Fixed': None}])
            main_window.df_param_space = pd.concat([main_window.df_param_space, new_row], ignore_index=True)
    # Populate ComboBox
    param_combo.addItems(main_window.df_param_space['Parameter'].values)

    # Initialize the table with existing data
    update_param_table()

    # Close button
    close_button = QPushButton("Close")
    close_button.setStyleSheet("background-color: lightgreen; color: black")
    close_button.clicked.connect(param_space_window.accept)
    layout.addWidget(close_button)
    
    param_space_window.setLayout(layout)
    param_space_window.exec_()

def define_qois(main_window):
    if not hasattr(main_window, "obs_data"):
        QMessageBox.warning(main_window, "Missing observationa data", "Load the observational data before of define QoI(s).")
        return
    
    # Function to overwrite qoi in the DataFrame
    def add_overwrite_qoi():
        try:
            qoi_name = qoi_combo.currentText()
            columnObsData_name = obs_combo.currentText()
            distance = distance_combo.currentText()
            distance_weight = distance_weight_input.text()
            # Create the row or Overwrite qoi
            if main_window.df_qois.empty or qoi_name not in main_window.df_qois['QoI_Name'].values:
                new_row = pd.DataFrame([{
                    'QoI_Name': qoi_name,
                    'QoI_Function': main_window.qoi_funcs[qoi_name],
                    'ObsData_Column': columnObsData_name,
                    'QoI_distanceFunction': distance,
                    'QoI_distanceWeight': float(distance_weight)
                }])
                main_window.df_qois = pd.concat([main_window.df_qois, new_row], ignore_index=True)
            else:
                # Overwrite qoi
                main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, 'QoI_Function'] = main_window.qoi_funcs[qoi_name]
                main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, 'ObsData_Column'] = columnObsData_name
                main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, 'QoI_distanceFunction'] = distance
                main_window.df_qois.loc[main_window.df_qois['QoI_Name'] == qoi_name, 'QoI_distanceWeight'] = float(distance_weight)

            # Update the table display
            update_qoi_table()
        except Exception as e:
            logging.error(f"Error defining QoI: {e}")

    # Function to remove qoi in the Dataframe
    def remove_qoi():
        qoi_name = qoi_combo.currentText()
        # Remove
        main_window.df_qois.drop(main_window.df_qois[main_window.df_qois['QoI_Name'] == qoi_name].index, inplace=True)

        # Update the table display
        update_qoi_table()

    # Function to update the table with current qois data
    def update_qoi_table():
        if main_window.df_qois is not None and not main_window.df_qois.empty:
            qoi_table.setRowCount(len(main_window.df_qois))
            # Add items to the qoi combox
            qoi_combo.clear()
            qoi_combo.addItems(main_window.df_qois['QoI_Name'].values)
            for i, row in main_window.df_qois.iterrows():
                qoi_table.setItem(i, 0, QTableWidgetItem(str(row['QoI_Name'])))
                qoi_table.setItem(i, 1, QTableWidgetItem(str(row['QoI_Function'])))
                qoi_table.setItem(i, 2, QTableWidgetItem(str(row['ObsData_Column'])))
                qoi_table.setItem(i, 3, QTableWidgetItem(str(row['QoI_distanceFunction'])))
                qoi_table.setItem(i, 4, QTableWidgetItem(str(row['QoI_distanceWeight'])))
        else:
            qoi_table.setRowCount(0)
    
    # Function to open and update the QoI definition window
    def open_and_update_qoi_definition():
        main_window.open_qoi_definition_window(main_window)
        # Update qoi_combo after the window is closed
        qoi_combo.clear()
        if main_window.qoi_funcs:
            qoi_combo.addItems(main_window.qoi_funcs.keys())

    # Define QoIs and map them with observational data
    qoi_window = QDialog(main_window)
    qoi_window.setWindowTitle("Define QoIs and Map with Observational Data")
    qoi_window.setMinimumSize(600, 500)

    layout = QVBoxLayout()

    # Define QoI Button from model analysis tab (tab2)
    define_qoi_button = QPushButton("Define QoI(s)")
    define_qoi_button.setStyleSheet("background-color: lightgreen; color: black")
    layout.addWidget(define_qoi_button)
    define_qoi_button.clicked.connect(open_and_update_qoi_definition)
    
    # Separator line
    layout.addWidget(QLabel("<hr>"))

    # QoI and observational data mapping
    qoi_obs_hbox = QHBoxLayout()
    qoi_label = QLabel("Select QoI:")
    qoi_obs_hbox.addWidget(qoi_label)
    qoi_combo = QComboBox()
    qoi_combo.addItems(main_window.qoi_funcs.keys())
    qoi_obs_hbox.addWidget(qoi_combo)
    layout.addLayout(qoi_obs_hbox)

    # Map qoi to observational data
    obs_data_hbox = QHBoxLayout()
    obs_label = QLabel("Select Observational Data Column:")
    obs_data_hbox.addWidget(obs_label)
    obs_combo = QComboBox()
    obs_combo.addItems(main_window.obs_data.columns)
    obs_data_hbox.addWidget(obs_combo)
    layout.addLayout(obs_data_hbox)

    # Distance
    distance_hbox = QHBoxLayout()
    distance_label = QLabel("Distance:")
    distance_hbox.addWidget(distance_label)
    distance_combo = QComboBox()
    distance_combo.addItems(["SumSquaredDifferences", "Manhattan", "Chebyshev"])
    distance_hbox.addWidget(distance_combo)
    distance_weight_label = QLabel("Weight:")
    distance_hbox.addWidget(distance_weight_label)
    distance_weight_input = QLineEdit("1.0")
    distance_hbox.addWidget(distance_weight_input)
    layout.addLayout(distance_hbox)
    
    # Add qoi button
    buttons_hbox = QHBoxLayout()
    add_button = QPushButton("Add/Overwrite QoI")
    add_button.setStyleSheet("background-color: lightgreen; color: black")
    add_button.clicked.connect(add_overwrite_qoi)
    buttons_hbox.addWidget(add_button)
    # Remove
    remove_button = QPushButton("Remove QOI")
    remove_button.setStyleSheet("background-color: lightgreen; color: black")
    remove_button.clicked.connect(remove_qoi)
    buttons_hbox.addWidget(remove_button)
    layout.addLayout(buttons_hbox)

    # Table to view QoI(s)
    qoi_table = QTableWidget()
    qoi_table.setColumnCount(5)
    qoi_table.setHorizontalHeaderLabels(["QoI Name", "Function", "Column ObsData", "QoI_distanceFunction", "Distance Weight"])
    qoi_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table non-editable
    layout.addWidget(qoi_table)

    # Initialize df_qois if it doesn't exist
    if not hasattr(main_window, 'df_qois'):
        main_window.df_qois = pd.DataFrame(columns=['QoI_Name', 'QoI_Function', 'ObsData_Column', 'QoI_distanceFunction', 'QoI_distanceWeight'])
        # Explicitly set dtypes to prevent future warnings
        main_window.df_qois = main_window.df_qois.astype({
            'QoI_Name': 'object',
            'QoI_Function': 'object',
            'ObsData_Column': 'object',
            'QoI_distanceFunction': 'object',
            'QoI_distanceWeight': 'float64'
        })
    
    # Update the table with existing data
    update_qoi_table()

    # Close button
    close_button = QPushButton("Close")
    close_button.setStyleSheet("background-color: lightgreen; color: black")
    close_button.clicked.connect(qoi_window.accept)
    layout.addWidget(close_button)

    qoi_window.setLayout(layout)
    qoi_window.exec_()


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