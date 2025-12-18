import configparser
import xml.etree.ElementTree as ET
import os
import pandas as pd
import traceback

from PyQt5.QtWidgets import QFileDialog, QInputDialog

from ..database.ma_db import get_database_type

def load_xml_file(main_window, filePath=None):
     # Clear parameters, .ini File Preview, and output
    main_window.analysis_parameters.clear()
    main_window.fixed_parameters.clear()
    main_window.analysis_rules_parameters.clear()
    main_window.fixed_rules_parameters.clear()
    # main_window.preview_table.clear()
    main_window.output_text.clear()
    # Load the XML file and parse it
    if filePath is None:
        # Open file dialog to select XML file
        options = QFileDialog.Options()
        main_window.xml_file_path, _ = QFileDialog.getOpenFileName(main_window, "Select PhysiCell XML File", "", "XML Files (*.xml);;All Files (*)", options=options)
    else:
        main_window.xml_file_path = filePath

    if main_window.xml_file_path:
        try:
            # Parse XML file
            main_window.xml_tree = ET.parse(main_window.xml_file_path).getroot()
            # Define the path to the rules CSV file
            folder_path = main_window.xml_file_path.rsplit("/", 1)[0]  # Get the folder path of the XML file
            filename = main_window.xml_tree.find(".//cell_rules/rulesets/ruleset/filename").text.strip()
            main_window.rule_path = os.path.join(folder_path, filename)  # Initialize rule path
            # Build parent-child mapping
            main_window.parent_map = {child: parent for parent in main_window.xml_tree.iter() for child in parent}
            # Clear existing combo boxes
            main_window.clear_combo_boxes(main_window)
            # Create the first combo box for root-level children
            main_window.create_combo_box(main_window, main_window.xml_tree, "Root")
            main_window.update_output_tab1(main_window, f"Loaded XML file: {main_window.xml_file_path}")
            # Activate the buttons
            main_window.new_value_input.setEnabled(True)
            main_window.remove_param_button.setEnabled(True)
            main_window.load_rules_button.setEnabled(True)
        except Exception as e:
            main_window.clear_combo_boxes(main_window)
            main_window.update_output_tab1(main_window, f"Error loading XML: {e}")

def load_csv_file(main_window):
    # Toggle between loading and unloading the rules CSV
    if main_window.load_rules_button.text() == "Unload Rules CSV":
        # Unload the rules widgets
        main_window.clear_rule_section(main_window)  # Clear the rule section
        main_window.load_rules_button.setText("Load Rules CSV")
        main_window.update_output_tab1(main_window, "Unloaded rules CSV.")
    else:
        # Load the rules CSV
        try:
            if os.path.exists(main_window.rule_path):
                # Predefine column names
                column_names = ["cell", "signal", "direction", "behavior", "saturation", "half-max", "hill-power", "apply-dead"]
                main_window.csv_data = pd.read_csv(main_window.rule_path, names=column_names, header=None)
                # Add 'inactive' column as False by default - All rules in the CSV are active
                main_window.csv_data['inactive'] = 'false'
                # print(main_window.csv_data)
                main_window.update_output_tab1(main_window, f"Loaded CSV file: {main_window.rule_path}")
                main_window.create_rule_section(main_window)  # Create the rules section
                main_window.load_rules_button.setText("Unload Rules CSV")
            else:
                main_window.update_output_tab1(main_window, f"Error: CSV file not found at {main_window.rule_path}")
        except Exception as e:
            main_window.update_output_tab1(main_window, f"Error loading CSV file: {e}")

def update_rules_file(main_window):
    # Update the rules file path based on the selected folder and filename
    try:
        # Check if folder_path is already set in fixed_parameters, else use the same folder of XML
        if ".//cell_rules/rulesets/ruleset/folder" in main_window.fixed_parameters.keys():
            folder_path = main_window.fixed_parameters[".//cell_rules/rulesets/ruleset/folder"]
        else:
            folder_path = folder_path = main_window.xml_file_path.rsplit("/", 1)[0]  # Get the folder path of the XML file
        # Check if filename is already set in fixed_parameters, else use the XML
        if ".//cell_rules/rulesets/ruleset/filename" in main_window.fixed_parameters.keys():
            filename = main_window.fixed_parameters[".//cell_rules/rulesets/ruleset/filename"]
        else:
            filename = main_window.get_parameter_value_xml(main_window, ".//cell_rules/rulesets/ruleset/filename")
        main_window.rule_path = os.path.join(folder_path, filename)
        main_window.update_output_tab1(main_window, f"Updated rules file path: {main_window.rule_path}")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error updating rules file path: {e}")

def load_ini_file(main_window, filePath=None, strucName=None):
    # Load .ini file and extract parameters for model exploration
    if filePath is None:
        options = QFileDialog.Options()
        main_window.ini_file_path, _ = QFileDialog.getOpenFileName(main_window, "Select .ini File", "", "INI Files (*.ini);;All Files (*)", options=options)
        # Switch the selected tab to Tab 2
        main_window.tabs.setCurrentIndex(1)
    else:
        main_window.ini_file_path = filePath
    # Convert to relative path
    main_window.ini_file_path = os.path.relpath(main_window.ini_file_path, os.getcwd())
    if main_window.ini_file_path:
        try:
            # Read the .ini file
            config = configparser.ConfigParser()
            config.read(main_window.ini_file_path)

            # Get all sections from the .ini file
            sections = config.sections()
            if not sections:
                main_window.update_output_tab1(main_window, "Error: No sections found in the .ini file.")
                raise ValueError("No sections found in the .ini file.")
            # Determine which section to load
            if strucName is not None:
                # print(f"Provided structure name: {strucName}")
                # print(f"Available sections: {sections}")
                if strucName in sections:
                    struc_name = strucName
                else:
                    raise ValueError(f"Structure name '{strucName}' not found in the .ini file.")
            else: # If no structure name provided, prompt the user to select one
                # If only one section, load it directly
                if len(sections) == 1:
                    struc_name = sections[0]
                else: # Create a dialog to display radio buttons for section selection
                    dialog = QInputDialog(main_window)
                    dialog.setWindowTitle("Select structure name")
                    dialog.setLabelText("Select the structure section to load:")
                    dialog.setComboBoxItems(sections)
                    dialog.setOption(QInputDialog.UseListViewForComboBoxItems)

                    if dialog.exec_() == QInputDialog.Accepted:
                        struc_name = dialog.textValue()
                        if struc_name not in sections:
                            raise ValueError(f"Structure name '{struc_name}' not found in the .ini file.")
                    else:
                        raise ValueError("No section selected.")

            # Extract parameters from the selected section
            main_window.struc_name_input.setText(struc_name)
            main_window.executable_path_input.setText(config[struc_name]['executable'])
            main_window.num_replicates_input.setText(config[struc_name]['numReplicates'])
            main_window.load_xml_file(main_window, config[struc_name]['configFile_ref'])

            # Add the loaded parameters to dictionaries
            for key, value in eval(config[struc_name]['parameters']).items():
                if isinstance(value, list):
                    main_window.analysis_parameters[key] = value
                else:
                    main_window.fixed_parameters[key] = value

            # Check if rules file exists in the fixed parameters from .ini file
            if ".//cell_rules/rulesets/ruleset/filename" in main_window.fixed_parameters.keys():
                update_rules_file(main_window)

            # Extract the rule file path if it exists
            if 'rulesFile_ref' in config[struc_name].keys():
                main_window.rule_path = config[struc_name]['rulesFile_ref']
                main_window.load_csv_file(main_window) # load the rules CSV file
                main_window.load_csv_file(main_window) # the rules section in the tab
                for key, value in eval(config[struc_name]['parameters_rules']).items():
                    if isinstance(value, list):
                        main_window.analysis_rules_parameters[key] = value
                    else:
                        main_window.fixed_rules_parameters[key] = value

            # Update the preview of the .ini file
            main_window.update_preview_table(main_window)

            # Clean the parameter samples
            if hasattr(main_window, 'local_SA_parameters'):
                main_window.local_SA_parameters.clear()
            if hasattr(main_window, 'global_SA_parameters'):
                main_window.global_SA_parameters.clear()

            # Write a message in the output fields of Tab 1 and Tab 2
            message = f".ini file loaded: {main_window.ini_file_path} - structure name: {struc_name}"
            main_window.update_output_tab1(main_window, message)  # Tab 1 output
            main_window.update_output_tab2(main_window, message)  # Tab 2 output

            # Re-enable buttons after successful loading
            main_window.sample_params_button.setEnabled(True)
            main_window.plot_samples_button.setEnabled(True)
            # Re-enable widgets after run_SA
            main_window.global_ref_value_input.setEnabled(True)
            main_window.global_range_percentage.setEnabled(True)
            main_window.global_bounds.setEnabled(True)
            main_window.local_param_combo.setEnabled(True)
            main_window.local_ref_value_input.setEnabled(True)
            main_window.local_perturb_input.setEnabled(True)
            # Disable the run SA button
            main_window.run_sa_button.setEnabled(False)

            # Update the analysis type dropdown
            main_window.update_sampling_type(main_window)

        except Exception as e:
            # Ensure buttons remain disabled if loading fails
            main_window.sample_params_button.setEnabled(False)
            main_window.plot_samples_button.setEnabled(False)
            
            # Enhanced error message with more context
            error_message = f"Error loading .ini file: {e} (Type: {type(e).__name__})"
            
            # Add file and section context if available
            if hasattr(main_window, 'ini_file_path'):
                error_message += f"\nFile: {main_window.ini_file_path}"
            if 'struc_name' in locals():
                error_message += f"\nSection: {struc_name}"
            
            # Add traceback for more technical details
            error_message += f"\nTraceback: {traceback.format_exc()}"
            
            print(error_message)
            main_window.update_output_tab1(main_window, error_message)  # Tab 1 output
            main_window.update_output_tab2(main_window, error_message)  # Tab 2 output
            raise ValueError(error_message)

def load_db_file(main_window, filePath=None):
        # Load the database file and extract parameters for sensitivity analysis
        if filePath is None:
            options = QFileDialog.Options()
            db_file_path, _ = QFileDialog.getOpenFileName(main_window, "Select .db File", "", "DB Files (*.db);;All Files (*)", options=options)
        else:
            db_file_path = filePath
        if db_file_path:
            try:
                db_type = get_database_type(db_file_path)  # Check if the file is a valid database
                # Convert to relative path
                db_file_path = os.path.relpath(db_file_path, os.getcwd())
                print(f"Loading Model Analysis database from {db_file_path} - type: {db_type}")
                # Model Analysis database
                if db_type == 'MA':
                    main_window.ma_file_path = db_file_path
                    main_window.load_ma_database(main_window)
                    # Switch the selected tab to Tab 2 - Model Analysis
                    main_window.tabs.setCurrentIndex(1)
                # Bayesian Optimization database
                elif db_type == 'BO':
                    main_window.bo_file_path = db_file_path
                    main_window.load_bo_database(main_window)
                    # Switch the selected tab to Tab 3 - Bayesian Optimization
                    main_window.tabs.setCurrentIndex(2)
                else:
                    error_message = "The selected file is not a valid MA or BO database."
                    main_window.update_output_tab2(main_window, error_message)
                    return
            except ValueError as e:
                # Handle the case where the file is not a valid .db file
                error_message = f"Error loading .db file: {e} (Type: {type(e).__name__})"

                # Add file context if available
                if hasattr(main_window, 'ma_file_path'):
                    error_message += f"\nFile: {main_window.ma_file_path}"
                if hasattr(main_window, 'bo_file_path'):
                    error_message += f"\nFile: {main_window.bo_file_path}"
                # Add traceback for more technical details
                error_message += f"\nTraceback: {traceback.format_exc()}"
                print(error_message)
                main_window.update_output_tab2(main_window, error_message)
                return