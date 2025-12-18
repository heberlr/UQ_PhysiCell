import os, sys
import logging
import seaborn as sns
import numpy as np
import pandas as pd
import traceback

# All the specific classes we need
from threading import Thread
from PyQt5.QtWidgets import (QVBoxLayout, QLabel, QHBoxLayout, QPushButton, 
                            QComboBox, QLineEdit, QTextEdit, QDialog, 
                            QInputDialog, QListWidget, QMessageBox, QCheckBox, QApplication)
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# My local modules
from ..model_analysis.samplers import run_global_sampler, run_local_sampler
from ..model_analysis.sensitivity_analysis import run_global_sa, run_local_sa, samplers_to_method
from ..model_analysis.ma_context import ModelAnalysisContext, run_simulations
from ..model_analysis.utils import calculate_qoi_statistics
from ..database.ma_db import load_structure

class QtTextEditLogHandler(logging.Handler):
    """Thread-safe logging handler that uses signal/slot mechanism."""

    def __init__(self, text_edit_widget):
        logging.Handler.__init__(self)
        self.text_edit = text_edit_widget
        self.main_window = None
        self.tab_id = None
        
        # Try to find the main window that contains this text edit
        parent = text_edit_widget.parent()
        while parent:
            if hasattr(parent, 'post_message') and hasattr(parent, 'message_queue'):
                self.main_window = parent
                self.tab_id = 'tab2'  # Default tab ID
                
                # Register the widget with the main window
                if hasattr(parent, 'add_output_widget'):
                    parent.add_output_widget(self.tab_id, text_edit_widget)
                break
            parent = parent.parent()

    def emit(self, record):
        """Emit a log record in a thread-safe way."""
        try:
            msg = self.format(record)
            
            # Use the main window's message queue if available
            if self.main_window and hasattr(self.main_window, 'post_message'):
                self.main_window.post_message(self.tab_id, msg)
            else:
                # Fall back to stderr
                print(msg, file=sys.stderr)
                
        except Exception as e:
            # If anything goes wrong, write to stderr
            print(f"Logging error: {e}", file=sys.stderr)

    def close(self):
        """Override close method to handle proper cleanup."""
        self.text_edit = None
        super().close()
    
    # Set flushOnClose to False to prevent access during shutdown
    flushOnClose = False

class NonZeroDoubleValidator(QDoubleValidator):
    """Custom validator that accepts double values but excludes zero"""
    def validate(self, input_str, pos):
        state, input_str, pos = super().validate(input_str, pos)
        if state == QDoubleValidator.Acceptable:
            try:
                value = float(input_str)
                if value == 0.0:
                    return (QDoubleValidator.Invalid, input_str, pos)
            except ValueError:
                pass
        return (state, input_str, pos)

def on_run_simulations_clicked(main_window):
    # First ensure the UI is in a clean state if a previous thread exists but is dead
    if hasattr(main_window, 'simulation_thread') and main_window.simulation_thread is not None:
        if not main_window.simulation_thread.is_alive():
            main_window.simulation_thread = None
            main_window.run_simulations_button.setText("Run Simulations")
            main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
    
    if main_window.simulation_thread is None:
        # --- Dialog and input collection (main thread only) ---
        dialog = QDialog(main_window)
        dialog.setWindowTitle("Simulation Configuration")
        dialog.setGeometry(100, 100, 400, 200)
        layout = QVBoxLayout(dialog)
        workers_layout = QHBoxLayout()
        workers_label = QLabel("Number of workers:")
        workers_input = QLineEdit()
        workers_input.setValidator(QIntValidator(1, 1000))
        workers_input.setText("1")
        workers_layout.addWidget(workers_label)
        workers_layout.addWidget(workers_input)
        layout.addLayout(workers_layout)
        xml_path_omp_threads = ".//parallel/omp_num_threads"
        num_replicates = main_window.num_replicates_input.text().strip()
        omp_threads = main_window.xml_tree.find(xml_path_omp_threads).text.strip()
        if xml_path_omp_threads in list(main_window.fixed_parameters.keys()):
            omp_threads = main_window.fixed_parameters[xml_path_omp_threads]
        total_threads_label = QLabel(f"OpenMP Threads per Model: {omp_threads}")
        layout.addWidget(total_threads_label)
        number_of_replicates_label = QLabel(f"Number of Replicates: {num_replicates}")
        layout.addWidget(number_of_replicates_label)
        number_of_samples = len(main_window.local_SA_parameters.get("samples")) if main_window.sampling_type_dropdown.currentText() == "Local" else len(main_window.global_SA_parameters.get("samples"))
        number_of_samples_label = QLabel(f"Number of Samples: {number_of_samples}")
        layout.addWidget(number_of_samples_label)
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        def confirm():
            dialog.accept()
        def cancel():
            dialog.reject()
        ok_button.clicked.connect(confirm)
        cancel_button.clicked.connect(cancel)
        result = dialog.exec_()
        if result != QDialog.Accepted:
            main_window.update_output_tab2(main_window, "Simulation aborted by user.")
            return
        num_workers = workers_input.text()
        # --- Collect all needed parameters for simulation ---
        sampling_type = main_window.sampling_type_dropdown.currentText()
        SA_method = main_window.SA_method_combo.currentText() if sampling_type else None
        SA_sampler = main_window.sampler_combo.currentText() if sampling_type else None
        SA_samples = main_window.local_SA_parameters.get("samples") if sampling_type == "Local" else main_window.global_SA_parameters.get("samples")
        db_file_name = main_window.db_file_name_input.text().strip()
        qoi_str = ', '.join(main_window.qoi_funcs.keys()) if main_window.qoi_funcs else None
        model_config = {"ini_path": main_window.ini_file_path, "struc_name": main_window.struc_name_input.text().strip()}
        # --- Validate inputs (all in main thread) ---
        if not sampling_type:
            main_window.update_output_tab2(main_window, "Error: Sensitivity analysis type is not selected.")
            return
        if not db_file_name:
            main_window.update_output_tab2(main_window, "Error: DB file name is required.")
            return
        if not db_file_name.endswith(".db"):
            main_window.update_output_tab2(main_window, "Error: DB file name must end with '.db'.")
            return
        if os.path.exists(db_file_name):
            reply = QMessageBox.question(main_window, "Overwrite Confirmation", f"File '{db_file_name}' already exists. Do you want to overwrite it?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                main_window.update_output_tab2(main_window, "Aborted saving. File was not overwritten.")
                return
        if not main_window.qoi_funcs.keys():
            reply = QMessageBox.question(main_window, "QoI Warning", "No QoI(s) defined. Do you want to run the simulation without QoIs? All data will be stored as mcds list.", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                main_window.update_output_tab2(main_window, "Aborted saving. No QoI(s) defined.")
                return
            else:
                main_window.update_output_tab2(main_window, "QoI(s) not defined: All data will be stored as mcds list.")
                qoi_str = None
        # --- Start simulation in background thread ---
        main_window.simulation_cancelled = False
        main_window.should_load_db = False  # Initialize flag for database loading
        # Configure logging only if logger not already set up
        if not hasattr(main_window, 'logger_tab2') or not main_window.logger_tab2:
            # Create logger in main thread
            main_window.logger_tab2 = logging.getLogger(__name__)
            main_window.logger_tab2.setLevel(logging.INFO)
            
            # Remove any existing handlers to avoid duplicates
            for handler in main_window.logger_tab2.handlers[:]:
                main_window.logger_tab2.removeHandler(handler)
            
            # Create and add handlers in main thread
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            main_window.logger_tab2.addHandler(console_handler)
            
            # Create a custom handler that uses the main window's message queue
            class QueueLogHandler(logging.Handler):
                def __init__(self, main_window, tab_id='tab2'):
                    super().__init__()
                    self.main_window = main_window
                    self.tab_id = tab_id
                    
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        if hasattr(self.main_window, 'post_message'):
                            self.main_window.post_message(self.tab_id, msg)
                        else:
                            print(msg, file=sys.stderr)
                    except Exception as e:
                        print(f"Logging error: {e}", file=sys.stderr)
            
            # Add our custom queue handler
            gui_handler = QueueLogHandler(main_window)
            gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            main_window.logger_tab2.addHandler(gui_handler)
                
            # Prevent propagation to root logger to avoid duplicate messages
            main_window.logger_tab2.propagate = False
            
        # Now define the thread function that will use the already-configured logger
        def run_simulation_thread():
            try:
                main_window.update_output_tab2(main_window, f"Running simulations with sampler: {SA_sampler} and number of samples: {len(SA_samples)}")
                # Prepare context
                if sampling_type == "Local":
                    # Create the context and store it in the main window for cancellation access
                    main_window.simulation_context = ModelAnalysisContext(
                        db_file_name, model_config, SA_sampler, 
                        main_window.local_SA_parameters, qoi_str, 
                        num_workers=int(num_workers), logger=main_window.logger_tab2
                    )
                    main_window.simulation_context.dic_samples = SA_samples
                    # Set up the cancelled method to check the GUI cancellation flag
                    main_window.simulation_context.cancelled = lambda: getattr(main_window, 'simulation_cancelled', False)
                    run_simulations(main_window.simulation_context)
                elif sampling_type == "Global":
                    # Create the context and store it in the main window for cancellation access
                    main_window.simulation_context = ModelAnalysisContext(
                        db_file_name, model_config, SA_sampler, 
                        main_window.global_SA_parameters, qoi_str, 
                        num_workers=int(num_workers), logger=main_window.logger_tab2
                    )
                    main_window.simulation_context.dic_samples = SA_samples
                    # Set up the cancelled method to check the GUI cancellation flag
                    main_window.simulation_context.cancelled = lambda: getattr(main_window, 'simulation_cancelled', False)
                    run_simulations(main_window.simulation_context)
                    # Clean up simulation context
                if hasattr(main_window, 'simulation_context') and main_window.simulation_context is not None:
                    if hasattr(main_window.simulation_context, 'model') and main_window.simulation_context.model is not None:
                        # Make sure all processes are terminated
                        main_window.simulation_context.model.terminate_all_simulations()
                    # Remove the reference to allow proper garbage collection
                    main_window.simulation_context = None
                
                # Update UI in the main thread via Qt's signal mechanism
                if getattr(main_window, 'simulation_cancelled', False):
                    main_window.update_output_tab2(main_window, "Simulation cancelled and all processes terminated successfully.")
                else:
                    main_window.update_output_tab2(main_window, f"Simulations completed and saved to {db_file_name} with QoIs: {', '.join(main_window.qoi_funcs.keys())}.")
                    main_window.ma_file_path = db_file_name

                    # Signal that we need to load the database when returning to the main thread
                    # We'll use a flag to indicate this in the completion timer
                    main_window.should_load_db = True
                    main_window.db_to_load = db_file_name
                
                # Always reset the UI state when done, regardless of success or cancellation
                def reset_ui_after_completion():
                    try:
                        main_window.run_simulations_button.setText("Run Simulations")
                        main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
                        main_window.run_simulations_button.setEnabled(True)
                        main_window.simulation_thread = None
                        # Force an immediate processing of the event queue to ensure UI updates
                        QApplication.processEvents()
                    except Exception as reset_error:
                        print(f"Error resetting UI after completion: {reset_error}")
                
            except Exception as e:
                # Log the full error with traceback
                error_with_traceback = f"Error: Running simulations failed: {e}\n{traceback.format_exc()}"
                main_window.update_output_tab2(main_window, error_with_traceback)
                main_window.simulation_cancelled = True
                
                # Clean up simulation context on error
                if hasattr(main_window, 'simulation_context') and main_window.simulation_context is not None:
                    try:
                        if hasattr(main_window.simulation_context, 'model') and main_window.simulation_context.model is not None:
                            # Make sure all processes are terminated
                            main_window.simulation_context.model.terminate_all_simulations()
                    except Exception as cleanup_error:
                        main_window.update_output_tab2(main_window, f"Warning: Error during cleanup: {cleanup_error}")
                    # Remove the reference to allow proper garbage collection
                    main_window.simulation_context = None
                
            # Reset the UI state when done, regardless of success, cancellation, or error
            # Use QTimer to ensure UI updates happen on the main thread
            QTimer.singleShot(0, reset_ui_after_completion)
        # Set UI state before starting thread
        main_window.run_simulations_button.setText("Cancel")
        main_window.run_simulations_button.setStyleSheet("background-color: orange; color: black")
        
        # Create a completion timer that will check for thread completion regardless of success/failure
        completion_timer = QTimer(main_window)
        
        def check_completion():
            if main_window.simulation_thread is None or not main_window.simulation_thread.is_alive():
                # Thread has completed, reset the button
                main_window.run_simulations_button.setText("Run Simulations")
                main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
                main_window.run_simulations_button.setEnabled(True)
                main_window.simulation_thread = None
                
                # Check if we need to load a database (set by the thread when finishing successfully)
                if hasattr(main_window, 'should_load_db') and main_window.should_load_db:
                    main_window.should_load_db = False  # Reset the flag
                    # Now that we're safely in the main thread, load the database
                    try:
                        main_window.load_ma_database(main_window)
                    except Exception as e:
                        error_msg = f"Error loading database: {str(e)}\n{traceback.format_exc()}"
                        main_window.update_output_tab2(main_window, error_msg)
                
                # Stop the timer and store it
                completion_timer.stop()
                if hasattr(main_window, '_completion_timers'):
                    main_window._completion_timers = []
                main_window._completion_timers.append(completion_timer)
        
        completion_timer.timeout.connect(check_completion)
        completion_timer.start(500)  # Check every 500ms
        
        # Create and start the thread after UI updates
        main_window.simulation_thread = Thread(target=run_simulation_thread)
        main_window.simulation_thread.daemon = True  # Make thread daemon so it doesn't block app exit
        main_window.simulation_thread.start()
    else:
        # Set the cancellation flag
        main_window.simulation_cancelled = True
        main_window.update_output_tab2(main_window, "Cancellation requested. Terminating all simulations...")
        
        # If we have a ModelAnalysisContext object stored, use its request_cancellation method
        if hasattr(main_window, 'simulation_context') and main_window.simulation_context is not None:
            main_window.simulation_context.request_cancellation()
            # Update UI with process termination status
            if hasattr(main_window.simulation_context, 'model') and main_window.simulation_context.model is not None:
                process_count = len(getattr(main_window.simulation_context.model, 'active_processes', {}))
                if process_count > 0:
                    main_window.update_output_tab2(main_window, f"Terminating {process_count} active processes...")
        
        # Don't join the thread here - that would block the GUI
        # Instead, disable the button temporarily to prevent multiple cancellations
        main_window.run_simulations_button.setEnabled(False)
        main_window.run_simulations_button.setText("Cancelling...")
        main_window.run_simulations_button.setStyleSheet("background-color: orange; color: black")
        
        # Use QTimer to periodically check if the thread has completed
        def check_thread_completion():
            # Check if thread is still running
            if main_window.simulation_thread is None or not main_window.simulation_thread.is_alive():
                # Thread has completed, reset the button
                main_window.run_simulations_button.setText("Run Simulations")
                main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
                main_window.run_simulations_button.setEnabled(True)
                main_window.simulation_thread = None
                cancel_timer.stop()
                main_window.update_output_tab2(main_window, "Simulation cancellation complete.")
                
                # Force immediate UI update
                QApplication.processEvents()
            else:
                # Thread is still running, check for any remaining processes
                if hasattr(main_window, 'simulation_context') and main_window.simulation_context is not None:
                    if hasattr(main_window.simulation_context, 'model') and main_window.simulation_context.model is not None:
                        process_count = len(getattr(main_window.simulation_context.model, 'active_processes', {}))
                        if process_count > 0:
                            main_window.run_simulations_button.setText(f"Cancelling ({process_count})...")
        
        cancel_timer = QTimer(main_window)
        cancel_timer.timeout.connect(check_thread_completion)
        cancel_timer.start(200)  # Check more frequently (every 200 ms) for better responsiveness
        
        # Store the timer reference on the main window to prevent garbage collection
        if not hasattr(main_window, '_cancel_timers'):
            main_window._cancel_timers = []
        main_window._cancel_timers.append(cancel_timer)

def create_tab2(main_window):
    # Add the following methods to the main_window instance
    main_window.update_output_tab2 = update_output_tab2
    main_window.load_ma_database = load_ma_database
    main_window.update_sampling_type = update_sampling_type
    main_window.update_local_inputs = update_local_inputs
    main_window.update_local_SA_reference = update_local_SA_reference
    main_window.update_local_SA_perturbations = update_local_SA_perturbations
    main_window.update_global_inputs = update_global_inputs
    main_window.update_global_SA_reference = update_global_SA_reference
    main_window.update_global_SA_range_percentage = update_global_SA_range_percentage
    main_window.sample_parameters = sample_parameters
    main_window.plot_samples = plot_samples
    main_window.update_sampler_options = update_sampler_options
    main_window.run_simulations_function = run_simulations_function
    main_window.open_qoi_definition_window = open_qoi_definition_window
    main_window.run_analysis = run_analysis
    main_window.plot_sa_results = plot_sa_results
    main_window.plot_qois = plot_qois

    layout_tab2 = QVBoxLayout()

    ###########################################
    # Dropdown for sampling type
    ###########################################
    main_window.sampling_title_label = QLabel("<b>Parameter Sampling</b>")
    main_window.sampling_title_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.sampling_title_label)
    main_window.sampling_type_hbox = QHBoxLayout()
    main_window.sampling_type_label = QLabel("Parameter Sampling Type:")
    main_window.sampling_type_dropdown = QComboBox()
    main_window.sampling_type_dropdown.addItems(["Local", "Global"])
    main_window.sampling_type_dropdown.currentIndexChanged.connect(lambda: main_window.update_sampling_type(main_window))
    main_window.sampling_type_hbox.addWidget(main_window.sampling_type_label)
    main_window.sampling_type_hbox.addWidget(main_window.sampling_type_dropdown)
    main_window.sampling_type_hbox.addSpacing(20)  # Space between the two blocks
    # Sampler options
    main_window.sampler_label = QLabel("Sampler:")
    main_window.sampler_combo = QComboBox()
    main_window.sampling_type_hbox.addWidget(main_window.sampler_label)
    main_window.sampling_type_hbox.addWidget(main_window.sampler_combo)
    main_window.sampling_type_hbox.addStretch()  # Push everything to the left
    layout_tab2.addLayout(main_window.sampling_type_hbox)

    ###########################################
    # Hbox for LOCAL sampling method
    ###########################################
    main_window.local_param_hbox = QHBoxLayout()
    # Select parameter for local SA
    main_window.local_param_label = QLabel("Select Parameter:")
    main_window.local_param_label.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_param_label)
    main_window.local_param_combo = QComboBox()
    main_window.local_param_combo.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_param_combo)
    # Reference value input
    main_window.local_ref_value_label = QLabel("Reference Value:")
    main_window.local_ref_value_label.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_ref_value_label)
    main_window.local_ref_value_input = QLineEdit()
    main_window.local_ref_value_input.setValidator(NonZeroDoubleValidator())
    main_window.local_ref_value_input.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_ref_value_input)
    # Percentage perturbations input
    main_window.local_perturb_label = QLabel("Percentage Perturbation(s) (+/-):")
    main_window.local_perturb_label.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_perturb_label)
    main_window.local_perturb_input = QLineEdit()
    main_window.local_perturb_input.setVisible(False)
    main_window.local_param_hbox.addWidget(main_window.local_perturb_input)
    layout_tab2.addLayout(main_window.local_param_hbox)

    ###########################################
    # Hbox for GLOBAL sampling method
    ###########################################
    main_window.global_parameters_hbox = QHBoxLayout()
    # Select parameter for global SA
    main_window.global_param_label = QLabel("Select Parameter:")
    main_window.global_param_label.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_param_label)
    main_window.global_param_combo = QComboBox()
    main_window.global_param_combo.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_param_combo)
    # Reference value input
    main_window.global_ref_value_label = QLabel("Ref. Value:")
    main_window.global_ref_value_label.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_ref_value_label)
    main_window.global_ref_value_input = QLineEdit()
    main_window.global_ref_value_input.setValidator(NonZeroDoubleValidator())
    main_window.global_ref_value_input.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_ref_value_input)
    # Percentage of range of parameters input
    main_window.global_range_percentage_label = QLabel("Range (%):")
    main_window.global_range_percentage_label.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_range_percentage_label)
    main_window.global_range_percentage = QLineEdit()
    main_window.global_range_percentage.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_range_percentage)
    # bounds of the range - (min, max) float
    main_window.global_bounds_label = QLabel("Bounds (min, max):")
    main_window.global_bounds_label.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_bounds_label)
    main_window.global_bounds = QLineEdit()
    main_window.global_bounds.setReadOnly(True)
    main_window.global_bounds.setVisible(False)
    main_window.global_parameters_hbox.addWidget(main_window.global_bounds)
    layout_tab2.addLayout(main_window.global_parameters_hbox)

    ###########################################
    # Horizontal layout for Sampling and Plot
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
    # Horizontal layout for Run simulations: Define Qoi(s), Run Simulations, Plot QoI(s)
    ###########################################
    main_window.analysis_type_label = QLabel("<b>Run Simulations and DefineQoI(s)</b>")
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
    main_window.define_qoi_button.clicked.connect(lambda: main_window.open_qoi_definition_window(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.define_qoi_button)
    # Run simulations button
    main_window.run_simulations_button = QPushButton("Run Simulations")
    main_window.run_simulations_button.setEnabled(False)
    main_window.run_simulations_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.simulation_thread = None
    main_window.simulation_cancelled = False
    main_window._cancel_timers = []
    main_window._completion_timers = []
    main_window.run_simulations_button.clicked.connect(lambda: on_run_simulations_clicked(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.run_simulations_button)
    # Plot QoI button
    main_window.plot_qois_button = QPushButton("Plot QoI(s)")
    main_window.plot_qois_button.setEnabled(False)
    main_window.plot_qois_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_qois_button.clicked.connect(lambda: main_window.plot_qois(main_window))
    main_window.db_file_name_hbox.addWidget(main_window.plot_qois_button)
    layout_tab2.addLayout(main_window.db_file_name_hbox)
    # Current QoI label
    main_window.current_qoi_label = QLabel("Current QoI(s): None")
    main_window.current_qoi_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.current_qoi_label)

    # Separator line
    layout_tab2.addWidget(QLabel("<hr>"))

    ###########################################
    # Horizontal layout for Select SA method and Run SA button
    ###########################################
    main_window.SA_title_label = QLabel("<b>Sensitivity Analysis</b>")
    main_window.SA_title_label.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.SA_title_label)
    main_window.SA_name_hbox = QHBoxLayout()
    # SA method combo
    main_window.SA_method_label = QLabel("Method:")
    main_window.SA_name_hbox.addWidget(main_window.SA_method_label)
    main_window.SA_method_combo = QComboBox()
    main_window.SA_method_combo.currentIndexChanged.connect(lambda: main_window.update_sampler_options(main_window))
    main_window.SA_name_hbox.addWidget(main_window.SA_method_combo)
    main_window.sampling_type_hbox.addSpacing(20)  # Space between the two blocks
    # Run SA button
    main_window.run_sa_button = QPushButton("Run SA")
    main_window.run_sa_button.setEnabled(False)
    main_window.run_sa_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.run_sa_button.clicked.connect(lambda: main_window.run_analysis(main_window))
    main_window.SA_name_hbox.addWidget(main_window.run_sa_button)
    # Plot SA button
    main_window.plot_sa_button = QPushButton("Plot SA")
    main_window.plot_sa_button.setEnabled(False)
    main_window.plot_sa_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.plot_sa_button.clicked.connect(lambda: main_window.plot_sa_results(main_window))
    main_window.SA_name_hbox.addWidget(main_window.plot_sa_button)
    layout_tab2.addLayout(main_window.SA_name_hbox)
    main_window.SA_name_hbox.addStretch() # push everything to the left
    # Initialize the sampler options
    main_window.update_sampling_type(main_window)

    # Separator line
    layout_tab2.addWidget(QLabel("<hr>"))

    ###########################################
    # Display section - display information
    ###########################################
    main_window.output_label_tab2 = QLabel("<b>Display</b>")
    main_window.output_label_tab2.setAlignment(Qt.AlignCenter)
    layout_tab2.addWidget(main_window.output_label_tab2)
    
    main_window.output_text_tab2 = QTextEdit()
    main_window.output_text_tab2.setReadOnly(True)
    main_window.output_text_tab2.setMinimumHeight(100)
    layout_tab2.addWidget(main_window.output_text_tab2)
    
    # Register the output widget with the main window's message system
    if hasattr(main_window, 'add_output_widget'):
        main_window.add_output_widget('tab2', main_window.output_text_tab2)
        main_window.post_message('tab2', "Welcome to Model Analysis! Load a model and set parameters to begin.")

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
    if not main_window.df_output.empty and 'Data' in main_window.df_output.columns:
        if isinstance(main_window.df_output['Data'].iloc[0], pd.DataFrame):
            predefined_qoi_funcs = {qoi_name: None for qoi_name in main_window.df_output['Data'].iloc[0].columns if qoi_name not in ['time'] }
            custom_qoi_option = False
    if not predefined_qoi_funcs:
        predefined_qoi_funcs = {
            'total_cells': "lambda df: len(df)",
            'live_cells': "lambda df: len(df[df['dead'] == False])",
            'dead_cells': "lambda df: len(df[df['dead'] == True])",
            'mean_radial_distance': "lambda df: df[['position_x', 'position_y', 'position_z']].apply(lambda row: ((row['position_x']**2 + row['position_y']**2 + row['position_z']**2)**0.5), axis=1).mean()",
            'max_volume': "lambda df: df['total_volume'].max()",
            'min_volume': "lambda df: df['total_volume'].min()",
            'mean_volume': "lambda df: df['total_volume'].mean()",
            'std_volume': "lambda df: df['total_volume'].std()",
            'total_volume': "lambda df: df['total_volume'].sum()",
            'run_time': "lambda mcds: mcds.get_runtime()", # no argument needed
            'template_cellType_live': "lambda df: len( df[ (df['dead'] == False) & (df['cell_type'] == <cellType>) ])\n # Replace <cellType> with the desired cell type name",
            'template_meanSubstrate': "lambda df_subs: df_subs[ <substrateName>].mean() \n # Replace <substrateName> with the desired substrate name",
            'template_stdSubstrate': "lambda df_subs: df_subs[ <substrateName>].std() \n # Replace <substrateName> with the desired substrate name",
            'template_cellType_meanRadialDistance': "lambda df: df[ df['cell_type'] == <cellType> ][['position_x', 'position_y', 'position_z']].apply(lambda row: ((row['position_x']**2 + row['position_y']**2 + row['position_z']**2)**0.5), axis=1).mean() \n # Replace <cellType> with the desired cell type name",
            # 'Persistent homology - Vectorisation (muspan - topological data analysis)': "lambda df: compute_persistent_homology(df)",
        }

    # Reset the qois
    main_window.df_summary_qois = pd.DataFrame()
    
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
    """Thread-safe method to update the output QTextEdit in tab2."""
    try:
        # Always use the message queue for thread safety, regardless of which thread we're in
        if hasattr(main_window, 'post_message') and hasattr(main_window, 'message_queue'):
            main_window.post_message('tab2', message)
            return
        
        # Only use direct update if we're in the main thread AND the queue system isn't available
        if hasattr(main_window, 'output_text_tab2') and QApplication.instance().thread() == QThread.currentThread():
            main_window.output_text_tab2.append(message)
            return
        
        # Last resort - print to stdout
        print(f"[Tab2] {message}", file=sys.stdout)
        sys.stdout.flush()
        
    except Exception as e:
        # Print any errors to stderr
        print(f"Error in update_output_tab2: {e}", file=sys.stderr)
        print(f"Original message: {message}", file=sys.stderr)
    except Exception:
        # Absolute last resort - just append the message directly
        # May have thread safety issues but prevents complete failure
        try:
            main_window.output_text_tab2.append(message)
        except:
            pass

def load_ma_database(main_window):
    try:
        # Make sure we're on the main thread
        if QThread.currentThread() != QApplication.instance().thread():
            main_window.update_output_tab2(main_window, "WARNING: load_ma_database called from non-main thread. Scheduling on main thread...")
            QTimer.singleShot(0, lambda: main_window.load_ma_database(main_window))
            return
            
        # Verify that ma_file_path is set
        if not hasattr(main_window, 'ma_file_path') or main_window.ma_file_path is None:
            main_window.update_output_tab2(main_window, "ERROR: ma_file_path is not set")
            return
            
        # Verify that file exists
        if not os.path.exists(main_window.ma_file_path):
            main_window.update_output_tab2(main_window, f"ERROR: Database file {main_window.ma_file_path} does not exist")
            return
        
        # Load the database structure
        main_window.update_output_tab2(main_window, f"Loading database file {main_window.ma_file_path} ...")
        df_metadata, df_parameter_space, df_qois, dic_input, main_window.df_output = load_structure(main_window.ma_file_path, load_result=False)
        
        # Verify that data was loaded correctly
        if df_metadata.empty:
            main_window.update_output_tab2(main_window, f"ERROR: Failed to load metadata from {main_window.ma_file_path}")
            return
            
        # Load the .ini file
        main_window.update_output_tab2(main_window, f"Loading .ini file from {df_metadata['Ini_File_Path'].iloc[0]}...")
        main_window.load_ini_file(main_window, df_metadata['Ini_File_Path'].iloc[0], df_metadata['StructureName'].iloc[0])
        print(df_metadata)

        # Define the widget to display db structure
        Sampler = df_metadata['Sampler'].iloc[0]
        Param_explorer_type = "Local" if Sampler == "OAT" else "Global"
        # Switch the exploration type to the one defined in the database
        main_window.sampling_type_dropdown.setCurrentText(Param_explorer_type)
        main_window.update_sampling_type(main_window)
        if Param_explorer_type == "Global": # Global fields
            main_window.SA_method_combo.clear()
            if Sampler in samplers_to_method:
                main_window.SA_method_combo.addItems(samplers_to_method[Sampler])
            else: # LHS sampler map to methods that are not constrained
                main_window.SA_method_combo.addItems(samplers_to_method["Latin hypercube sampling (LHS)"])
            main_window.SA_method_combo.setCurrentText(main_window.SA_method_combo.itemText(0))
            main_window.sampler_combo.setCurrentText(Sampler)
            main_window.global_param_combo.setEnabled(True)
            main_window.global_ref_value_input.setEnabled(False)
            main_window.sampler_combo.setEnabled(False)
            main_window.global_range_percentage.setEnabled(False)
            main_window.global_bounds.setEnabled(False)
            # Populate the global_SA_parameters dictionary with values from the database
            main_window.global_SA_parameters = {}
            main_window.global_SA_parameters["samples"] = dic_input
            for id, param in enumerate(df_parameter_space['ParamName']):
                main_window.global_SA_parameters[param] = {"lower_bound": df_parameter_space['lower_bound'].iloc[id],
                                                            "upper_bound": df_parameter_space['upper_bound'].iloc[id],
                                                            "ref_value": df_parameter_space['ref_value'].iloc[id]}
                perturbation = df_parameter_space['perturbation'].iloc[id]
                try:
                    main_window.global_SA_parameters[param]["perturbation"] = float(perturbation)
                except Exception as e:
                    print(f"Warning: Could not convert perturbation ({perturbation}) to float for parameter {param}.")
                    # Calculate perturbation as percentage based on bounds and ref_value
                    main_window.global_SA_parameters[param]["perturbation"] = 100.0 * (df_parameter_space['upper_bound'].iloc[id]/df_parameter_space['ref_value'].iloc[id] - 1.0)
            # Update the global parameters
            main_window.update_global_inputs(main_window)
        else: 
            # Clean the method
            main_window.SA_method_combo.clear()
            main_window.SA_method_combo.addItems(samplers_to_method[Sampler])
            main_window.SA_method_combo.setCurrentText(main_window.SA_method_combo.itemText(0))
            main_window.sampler_combo.clear()
            main_window.sampler_combo.addItems([Sampler])
            main_window.sampler_combo.setCurrentText(Sampler)
            print("Setting local sampler to:", main_window.sampler_combo.itemText(0))
            # Deactivate local fields
            main_window.sampler_combo.setEnabled(False)
            main_window.local_param_combo.setEnabled(True)
            main_window.local_ref_value_input.setEnabled(False)
            main_window.local_perturb_input.setEnabled(False)
            # Populate the local_SA_parameters dictionary with values from the database
            main_window.local_SA_parameters = {}
            main_window.local_SA_parameters["samples"] = dic_input
            for id, param in enumerate(df_parameter_space['ParamName']):
                if type(df_parameter_space['perturbation'].iloc[id]) == list:
                    df_parameter_space['perturbation'].iloc[id] = [float(x) for x in df_parameter_space['perturbation'].iloc[id]]
                main_window.local_SA_parameters[param] = {"ref_value": df_parameter_space['ref_value'].iloc[id], 
                                                            "perturbation": df_parameter_space['perturbation'].iloc[id]}
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
        main_window.plot_sa_button.setEnabled(True)
        # Disable sampling type button after successful loading
        main_window.sampling_type_dropdown.setEnabled(False)
        # Disable sample_params button after successful loading
        main_window.sample_params_button.setEnabled(False)
        # Disable run simulations button after successful loading
        main_window.run_simulations_button.setEnabled(False)

        # Set db file to the loaded and disable the field
        main_window.db_file_name_input.setText(main_window.ma_file_path)
        main_window.db_file_name_input.setEnabled(False)

        # Enable the plot samples button only if samples are present
        if main_window.sampling_type_dropdown.currentText() == "Local" and "samples" in main_window.local_SA_parameters:
            main_window.plot_samples_button.setEnabled(True)
        elif main_window.sampling_type_dropdown.currentText() == "Global" and "samples" in main_window.global_SA_parameters:
            main_window.plot_samples_button.setEnabled(True)
        else:
            main_window.plot_samples_button.setEnabled(False)
            main_window.update_output_tab2(main_window, "No samples found in database. You may need to run parameter sampling.")

        # Load QoIs from database if available
        if not df_qois.empty and 'QOI_Name' in df_qois.columns and 'QOI_Function' in df_qois.columns:
            main_window.qoi_funcs = {row['QOI_Name']: row['QOI_Function'] for _, row in df_qois.iterrows() if row['QOI_Name'] is not None}
            if main_window.qoi_funcs:
                main_window.current_qoi_label.setText("Current QoI(s): " + ", ".join(main_window.qoi_funcs.keys()))
            else:
                main_window.qoi_funcs = {}
                main_window.current_qoi_label.setText("Current QoI(s): None")
        else:
            main_window.qoi_funcs = {}
            main_window.current_qoi_label.setText("Current QoI(s): None")
            
        # Reset summary dataframes
        main_window.df_summary_qois = pd.DataFrame()

        # print a message in the output fields of Tab 2
        message = f"Database file loaded: {main_window.ma_file_path}"
        main_window.update_output_tab2(main_window, message)

        # Active the run simulations button
        main_window.run_simulations_button.setEnabled(True)
        
    except ValueError:
        # Re-raise ValueError (from .ini file loading) to propagate it up
        raise
    except Exception as e:
        # Handle other exceptions (TypeError, AttributeError, etc.)
        error_message = f"Error loading .db file: {e} (Type: {type(e).__name__})"
        # Add traceback for more technical details
        error_message += f"\nTraceback: {traceback.format_exc()}"
        print(error_message)
        main_window.update_output_tab2(main_window, error_message)
        raise ValueError(error_message)

def update_sampling_type(main_window):
    # Show/hide UI elements based on selected sampling type
    if main_window.sampling_type_dropdown.currentText() == "Local":
        # Update the sensitivity analysis method combo box
        main_window.SA_method_combo.clear()
        main_window.SA_method_combo.addItems(["OAT - One-at-a-Time"])
        main_window.SA_method_combo.setCurrentIndex(0)
        # Update the sampler
        main_window.update_sampler_options(main_window)
        main_window.sampler_combo.setCurrentIndex(0)
        # Local fields
        main_window.local_param_label.setVisible(True)
        main_window.local_param_combo.setVisible(True)
        main_window.local_ref_value_label.setVisible(True)
        main_window.local_ref_value_input.setVisible(True)
        main_window.local_perturb_label.setVisible(True)
        main_window.local_perturb_input.setVisible(True)
        # Global fields
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
            # Get the default XML value - string
            string_value = main_window.get_parameter_value_xml(main_window, key)
            try: # Try to convert to float
                ref_value = float(string_value) 
                main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [1, 10, 20], "type": "float"}
            except ValueError: # If conversion fails, assign boolean values
                ref_value = 0.0 if string_value.lower() == 'false' else 1.0
                main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [100], "type": "bool"}

        for key, value in main_window.analysis_rules_parameters.items():
            # Get the default rule value - string
            string_value = main_window.get_rule_value(main_window, key)
            try: # Try to convert to float
                ref_value = float(string_value)
                main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [1, 10, 20], "type": "float"}
            except ValueError: # # If conversion fails, assign boolean values
                ref_value = 0.0 if string_value.lower() == 'false' else 1.0
                main_window.local_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": [100], "type": "bool"}

        # Add friendly names to the combo box
        main_window.local_param_combo.addItems(list(main_window.local_SA_parameters.keys()))

        # Connect the combo box to update the input fields
        main_window.local_param_combo.currentIndexChanged.connect(lambda: main_window.update_local_inputs(main_window))

        # Initialize the input fields for the first parameter
        if main_window.local_param_combo.count() > 0:
            main_window.update_local_inputs(main_window)
    elif main_window.sampling_type_dropdown.currentText() == "Global":
        # Update the sensitivity analysis method combo box
        main_window.SA_method_combo.clear()
        main_window.SA_method_combo.addItems(["Sobol Sensitivity Analysis",
            "FAST - Fourier Amplitude Sensitivity Test",
            "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
            "Delta Moment-Independent Measure",
            "Derivative-based Global Sensitivity Measure (DGSM)",
            "Fractional Factorial",
            "PAWN Sensitivity Analysis",
            "High-Dimensional Model Representation",
            "Regional Sensitivity Analysis",
            "Discrepancy Sensitivity Indices"
        ])
        # Update the sampler
        main_window.update_sampler_options(main_window)
        main_window.sampler_combo.setCurrentIndex(0)
        # Local fields
        main_window.local_param_label.setVisible(False)
        main_window.local_param_combo.setVisible(False)
        main_window.local_ref_value_label.setVisible(False)
        main_window.local_ref_value_input.setVisible(False)
        main_window.local_perturb_label.setVisible(False)
        main_window.local_perturb_input.setVisible(False)
        # Global fields
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
            # Get the default XML value - string
            string_value = main_window.get_parameter_value_xml(main_window, key)
            try: # Try to convert to float
                ref_value = float(string_value)
                # print(f"Update Analysis type - {key}: {ref_value}")
                main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 20.0, "lower_bound": ref_value * 0.8, "upper_bound": ref_value * 1.2, "type": "float"}
            except ValueError: # If conversion fails, assign boolean values
                ref_value = 0.0 if string_value.lower() == 'false' else 1.0
                main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 100.0, "lower_bound": 0.0, "upper_bound": 1.0, "type": "bool"}

        for key, value in main_window.analysis_rules_parameters.items():
            # Get the default rule value
            string_value = main_window.get_rule_value(main_window, key)
            try: # Try to convert to float
                ref_value = float(string_value)
                # print(f"Update Analysis type - {key}: {ref_value}")
                main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 20.0, "lower_bound": ref_value * 0.8, "upper_bound": ref_value * 1.2, "type": "float"}
            except ValueError: # If conversion fails, assign boolean values
                ref_value = 0.0 if string_value.lower() == 'false' else 1.0
                main_window.global_SA_parameters[value[1]] = {"ref_value": ref_value, "perturbation": 100.0, "lower_bound": 0.0, "upper_bound": 1.0, "type": "bool"}

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
            main_window.global_SA_parameters[selected_param]["lower_bound"] = lower_bound
            main_window.global_SA_parameters[selected_param]["upper_bound"] = upper_bound
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
            main_window.global_SA_parameters[selected_param]["perturbation"] = new_range_percentage
        except ValueError:
            main_window.update_output_tab2(main_window, "Error: Invalid range percentage.")

def check_bounds(dic_parameters):
    # Check that all lower bounds are less than upper bounds
    for param, values in dic_parameters.items():
        if param == "samples":
            continue
        if values["lower_bound"] >= values["upper_bound"]:
            raise ValueError(f"Lower bound {values['lower_bound']} is not less than upper bound {values['upper_bound']} for parameter {param}.")
    return True

def check_ref_values(dic_parameters):
    # Check that all reference values are within bounds
    for param, values in dic_parameters.items():
        if param == "samples":
            continue
        if values["ref_value"] == 0 and values["type"] == "float": # Prevent zero reference values for float parameters
            raise ValueError(f"Reference value for parameter {param} cannot be zero.")
    return True

def sample_parameters(main_window):
    # Sample parameters based on the selected SA
    if main_window.sampling_type_dropdown.currentText() == "Local":
        # Check the validator for the reference value and perturbation inputs
        if check_ref_values(main_window.local_SA_parameters) == False:
            QMessageBox.warning(main_window, "Invalid Input", "Reference value must be a non-zero number for float parameters.")
            return
        main_window.update_output_tab2(main_window, f"Sampling parameters using One-At-A-Time approach...")
        # Check if samples already exist, if so, delete them
        if 'samples' in main_window.local_SA_parameters.keys():
            del main_window.local_SA_parameters["samples"]
        # Run the local sampler
        try:
            main_window.local_SA_parameters["samples"] = run_local_sampler(main_window.local_SA_parameters)
        except ValueError as e:
            main_window.update_output_tab2(main_window, f"Error in local sampler: {e}")
            return
    elif main_window.sampling_type_dropdown.currentText() == "Global":
        # Check the validator for the parameter bounds inputs
        if check_bounds(main_window.global_SA_parameters) == False:
            QMessageBox.warning(main_window, "Invalid Input", "Check that all lower bounds are less than upper bounds.")
            return
        sampler = main_window.sampler_combo.currentText()
        main_window.update_output_tab2(main_window, f"Sampling parameters using {sampler}...")
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
    main_window.update_output_tab2(main_window, f"Sampling parameters completed.")
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
    
    # Check if samples data exists
    has_samples = False
    if main_window.sampling_type_dropdown.currentText() == "Local":
        has_samples = hasattr(main_window, 'local_SA_parameters') and main_window.local_SA_parameters and "samples" in main_window.local_SA_parameters
    elif main_window.sampling_type_dropdown.currentText() == "Global":
        has_samples = hasattr(main_window, 'global_SA_parameters') and main_window.global_SA_parameters and "samples" in main_window.global_SA_parameters
    
    if not has_samples:
        # Show error message and close the dialog
        QMessageBox.warning(plot_samples_window, "No Samples", "No parameter samples found. Please run parameter sampling first.")
        plot_samples_window.reject()
        return
    
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
        if main_window.sampling_type_dropdown.currentText() == "Local":
            # print(main_window.local_SA_parameters)
            perturbations_df = pd.DataFrame(main_window.local_SA_parameters["samples"]).T
            for col in perturbations_df.columns:
                if 'type' in main_window.local_SA_parameters[col] and main_window.local_SA_parameters[col]["type"] == "bool":
                    # For boolean parameters, map 0/1 to 0%/+100%
                    perturbations_df[col] = perturbations_df[col] * 100.0
                else:
                    perturbations_df[col] = 100.0 * (
                        perturbations_df[col] - main_window.local_SA_parameters[col]["ref_value"]
                    ) / main_window.local_SA_parameters[col]["ref_value"]
            perturbations_df = perturbations_df.reset_index().melt(
                id_vars="index", var_name="Parameter", value_name="perturbation"
            )
            perturbations_df = perturbations_df.rename(columns={"index": "SampleID"})
            perturbations_df['Frequency'] = perturbations_df.groupby(
                ['perturbation', 'Parameter']
            )['SampleID'].transform('count')
            scatter_plot = sns.scatterplot(
                data=perturbations_df,
                x="perturbation",
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
        elif main_window.sampling_type_dropdown.currentText() == "Global":
            # print(main_window.global_SA_parameters)
            normalized_df = pd.DataFrame(main_window.global_SA_parameters["samples"]).T
            for col in normalized_df.columns:
                normalized_df[col] = (
                    normalized_df[col] - main_window.global_SA_parameters[col]["lower_bound"]
                ) / (
                    main_window.global_SA_parameters[col]["upper_bound"] - main_window.global_SA_parameters[col]["lower_bound"]
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
            # Only add legend if there are labeled artists
            handles, labels = scatter_plot.get_legend_handles_labels()
            if handles and labels:
                scatter_plot.legend(
                    title="Sample Index",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    title_fontsize=8,
                    fontsize=8,
                )
        figure.set_constrained_layout(True)
        canvas.draw()

    try:
        update_plot()
         # Add a close button
        close_button = QPushButton("Close")
        close_button.setStyleSheet("background-color: lightgreen; color: black")
        close_button.clicked.connect(plot_samples_window.accept)
        layout.addWidget(close_button)

        plot_samples_window.exec_()
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting samples: {e}")
        main_window.update_output_tab2(main_window, f"Error plotting samples: {e}")
        QMessageBox.warning(plot_samples_window, "Plot Error", f"Error plotting samples: check the parameters ranges. ")

def update_sampler_options(main_window):
    # OAT - One-At-A-Time: combatible with OAT sampler
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
    # Update the sampler_combo options based on the selected method
    # Avoid unnecessary updates if the combo box is disabled - special case when sampler loaded from .db file will update the method list

    sampling_type = main_window.sampling_type_dropdown.currentText()
    method = main_window.SA_method_combo.currentText()
    # Clean the samplers list
    main_window.sampler_combo.clear()

    if method == "OAT - One-at-a-Time" and sampling_type == "Local": # Local sampling
        main_window.sampler_combo.addItems(["OAT"])
    elif method == "FAST - Fourier Amplitude Sensitivity Test" and sampling_type == "Global":
        main_window.sampler_combo.addItems(["Fast"])
    elif method == "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test" and sampling_type == "Global":
        main_window.sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol"
        ])
    elif method == "Sobol Sensitivity Analysis" and sampling_type == "Global":
        main_window.sampler_combo.addItems(["Sobol"])
    elif method == "Delta Moment-Independent Measure" and sampling_type == "Global":
        main_window.sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol"
        ])
    elif method == "Derivative-based Global Sensitivity Measure (DGSM)" and sampling_type == "Global":
        main_window.sampler_combo.addItems(["Finite Difference"])
    elif method == "Fractional Factorial" and sampling_type == "Global":
        main_window.sampler_combo.addItems(["Fractional Factorial"])
    elif method in [
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ] and sampling_type == "Global":
        main_window.sampler_combo.addItems([
            "Fast", "Fractional Factorial", "Finite Difference", 
            "Latin hypercube sampling (LHS)", "Sobol"
        ])
    else: # No compatible samplers
        main_window.sampler_combo.addItems([])  

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
        num_replicates = main_window.num_replicates_input.text().strip()
        omp_threads = main_window.xml_tree.find(xml_path_omp_threads).text.strip()
        # Check if the parameter is set in the .ini file
        if xml_path_omp_threads in list(main_window.fixed_parameters.keys()): omp_threads = main_window.fixed_parameters[xml_path_omp_threads]
        total_threads_label = QLabel(f"OpenMP Threads per Model: {omp_threads}")
        layout.addWidget(total_threads_label)
        number_of_replicates_label = QLabel(f"Number of Replicates: {num_replicates}")
        layout.addWidget(number_of_replicates_label)
        number_of_samples = len(main_window.local_SA_parameters.get("samples")) if main_window.sampling_type_dropdown.currentText() == "Local" else len(main_window.global_SA_parameters.get("samples"))
        number_of_samples_label = QLabel(f"Number of Samples: {number_of_samples}")
        layout.addWidget(number_of_samples_label)

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

        # Ensure sampling_type is defined only after the dialog is accepted
        if not main_window.sampling_type_dropdown.currentText():
            main_window.update_output_tab2(main_window, "Error: Sensitivity analysis type is not selected.")
            return

        # Initialize other variables only if sampling_type is valid
        SA_method, SA_sampler, SA_samples = None, None, None
        if main_window.sampling_type_dropdown.currentText() == "Local":
            try:
                SA_method = main_window.SA_method_combo.currentText()
                SA_sampler = main_window.sampler_combo.currentText()
                SA_samples = main_window.local_SA_parameters.get("samples")
                if SA_samples is None:
                    raise ValueError("No samples generated for local sensitivity analysis.")
            except KeyError:
                main_window.update_output_tab2(main_window, "Error: No samples generated for local sensitivity analysis.")
                return
        elif main_window.sampling_type_dropdown.currentText() == "Global":
            try:
                SA_method = main_window.SA_method_combo.currentText()
                SA_sampler = main_window.sampler_combo.currentText()
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
                qoi_str = None
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
        sampling_type = main_window.sampling_type_dropdown.currentText()
        # Setup logging
        # Create custom handler that writes to the GUI output text area
        gui_handler = QtTextEditLogHandler(main_window.output_text_tab2)
        gui_handler.setLevel(logging.INFO)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Create console handler for stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Configure logging only if logger not already set up
        if not hasattr(main_window, 'logger_tab2') or not main_window.logger_tab2:
            # Don't use basicConfig as it affects the root logger and can cause duplicated output
            main_window.logger_tab2 = logging.getLogger(__name__)
            main_window.logger_tab2.setLevel(logging.INFO)
            # Remove any existing handlers to avoid duplicates
            for handler in main_window.logger_tab2.handlers[:]:
                main_window.logger_tab2.removeHandler(handler)
            # Add our handlers
            main_window.logger_tab2.addHandler(gui_handler)
            main_window.logger_tab2.addHandler(console_handler)
            # Prevent propagation to root logger to avoid duplicate messages
            main_window.logger_tab2.propagate = False
        # Simulate the model with the selected samples
        if main_window.sampling_type_dropdown.currentText() == "Local":
            sampler = main_window.sampler_combo.currentText()
            # Model Analysis context
            context = ModelAnalysisContext(db_file_name, model_config, sampler, main_window.local_SA_parameters, qoi_str, num_workers=int(num_workers), logger=main_window.logger_tab2)
            context.dic_samples = SA_samples
            context.cancelled = lambda: getattr(main_window, 'simulation_cancelled', False)
            run_simulations(context)
        elif main_window.sampling_type_dropdown.currentText() == "Global":
            sampler = main_window.sampler_combo.currentText()
            # Model Analysis context
            context = ModelAnalysisContext(db_file_name, model_config, sampler, main_window.global_SA_parameters, qoi_str, num_workers=int(num_workers), logger=main_window.logger_tab2)
            context.dic_samples = SA_samples
            context.cancelled = lambda: getattr(main_window, 'simulation_cancelled', False)
            run_simulations(context)
        # After run, re-enable button
        main_window.run_simulations_button.setEnabled(True)
    except Exception as e:
        main_window.update_output_tab2(main_window, f"Error: Running simulations with {sampler} failed: {e}")
        print(f"Error: Running simulations with {sampler} failed: {e}")
        return

    # Simulate saving the samples to the database (replace with actual simulation logic)
    if getattr(main_window, 'simulation_cancelled', False):
        main_window.update_output_tab2(main_window, "Simulation cancelled.")
    else:
        main_window.update_output_tab2(main_window, f"Simulations completed and saved to {db_file_name} with QoIs: {', '.join(main_window.qoi_funcs.keys() if main_window.qoi_funcs else None)}.")
        # Load the database file to display results
        main_window.ma_file_path = db_file_name
        main_window.load_ma_database(main_window)
    

def plot_qois(main_window):
    main_window.update_output_tab2(main_window, "Plotting QoIs...")
    # Check if we have QoIs defined
    if not main_window.qoi_funcs:
        main_window.update_output_tab2(main_window, "Error: No QoI functions defined. Please define QoI functions first.")
        QMessageBox.warning(main_window, "No QoIs Defined", "Please define QoI functions before plotting.")
        return
        
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
    plot_qoi_mcse_checkbox = QCheckBox("Show Relative Monte Carlo Standard Error (MCSE)")
    plot_qoi_hbox.addWidget(plot_qoi_mcse_checkbox)
    layout.addLayout(plot_qoi_hbox)
    # Create a new figure and canvas for the plot
    figure = Figure(figsize=(5, 3))
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)
    # Calculate the QoIs if not already done
    if main_window.df_summary_qois.empty:
        try: main_window.df_summary_qois, main_window.df_relative_mcse = calculate_qoi_statistics(main_window.df_output, main_window.qoi_funcs, db_file_path = main_window.db_file_name_input.text().strip())
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
            if plot_qoi_mcse_checkbox.isChecked(): df_plot = main_window.df_relative_mcse
            else: df_plot = main_window.df_summary_qois
            print(f"Columns: {df_plot.columns}")
            qoi_columns = sorted([col for col in df_plot.columns if col.startswith(selected_qoi)])
            time_columns = sorted([col for col in df_plot.columns if col.startswith("time_")])
            # Prepare the data for seaborn
            plot_data = pd.DataFrame({
                "Time": df_plot[time_columns].values.flatten(),
                selected_qoi: df_plot[qoi_columns].values.flatten(),
                "SampleID": df_plot.index.repeat(len(qoi_columns))
            })
            # Plot using seaborn
            sns.lineplot(data=plot_data, x="Time", y=selected_qoi, hue="SampleID", ax=ax)
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(selected_qoi)
            # Only add legend if there are labeled artists
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(title="Sample Index")
            canvas.draw()
        else:
            main_window.update_output_tab2(main_window, f"Error: {selected_qoi} not found in the output data.")
        # Adjust layout and draw the canvas
        # figure.tight_layout()
        figure.set_constrained_layout(True)
        canvas.draw()
    
    try:
        # Connect the combo box to update the plot
        plot_qoi_combo.currentIndexChanged.connect(update_plot_qoi)
        plot_qoi_mcse_checkbox.stateChanged.connect(update_plot_qoi)
        # Set the default selected qoi and update the plot
        plot_qoi_combo.setCurrentIndex(0)
        update_plot_qoi()
        # Show the dialog
        plot_qoi_window.exec_()
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting QoIs: {e}")
        main_window.update_output_tab2(main_window, f"Error plotting QoIs: {e}")
        QMessageBox.warning(plot_qoi_window, "Plot Error", f"Error plotting QoIs: {e}")

def run_analysis(main_window):
    main_window.update_output_tab2(main_window, "Running sensitivity analysis...")
     # Calculate the QoIs if not already done
    if main_window.df_summary_qois.empty:
        try: main_window.df_summary_qois, main_window.df_relative_mcse = calculate_qoi_statistics(main_window.df_output, main_window.qoi_funcs, db_file_path = main_window.db_file_name_input.text().strip())
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error calculating QoIs: {e}")
            return
    # Prepare the QoIs for analysis
    all_qois_names = list(main_window.qoi_funcs.keys())
    all_times_label = [col for col in main_window.df_summary_qois.columns if col.startswith("time")]
    print(f"all_qois: {all_qois_names} and all_times: {all_times_label}")
    if main_window.sampling_type_dropdown.currentText() == "Global":
        global_method = main_window.SA_method_combo.currentText()
        try:
            main_window.sa_results, main_window.qoi_time_values = run_global_sa(main_window.global_SA_parameters, global_method, all_times_label, all_qois_names, main_window.df_summary_qois)
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error running global sensitivity analysis: {e}")
            return
    elif main_window.sampling_type_dropdown.currentText() == "Local":
        try:
            main_window.sa_results, main_window.qoi_time_values = run_local_sa(main_window.local_SA_parameters, all_times_label, all_qois_names, main_window.df_summary_qois)
        except Exception as e:
            main_window.update_output_tab2(main_window, f"Error running local sensitivity analysis: {e}")
            return
       
    main_window.update_output_tab2(main_window, "Sensitivity analysis completed.")

def plot_sa_results(main_window):
    if hasattr(main_window, 'sa_results') == False or not main_window.sa_results or hasattr(main_window, 'qoi_time_values') == False or not main_window.qoi_time_values:
        main_window.update_output_tab2(main_window, "Error: No sensitivity analysis results found. Please run the analysis first.")
        QMessageBox.warning(main_window, "No SA Results", "Please run sensitivity analysis before plotting results.")
        return
    # Plot the results
    main_window.update_output_tab2(main_window, f"Plotting results for {main_window.SA_method_combo.currentText()}")
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
    if main_window.sampling_type_dropdown.currentText() == "Global":
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
            if main_window.sampling_type_dropdown.currentText() == "Global":
                SA_method = main_window.SA_method_combo.currentText()
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
                # Only add legend if there are labeled artists
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title_fontsize=8, fontsize=8)
            elif main_window.sampling_type_dropdown.currentText() == "Local":
                SA_method = main_window.SA_method_combo.currentText()
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
                # Only add legend if there are labeled artists
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
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
