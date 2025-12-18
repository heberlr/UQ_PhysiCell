from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QComboBox, QFileDialog, QInputDialog, QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import os
import pandas as pd
import configparser

from .load_files import load_xml_file, load_csv_file, update_rules_file, load_ini_file

def create_tab1(main_window):
    # Add the following methods to the main_window instance
    main_window.load_xml_file = load_xml_file
    main_window.create_combo_box = create_combo_box
    main_window.clear_combo_boxes = clear_combo_boxes
    main_window.handle_combo_selection = handle_combo_selection
    main_window.update_selected_param_label = update_selected_param_label
    main_window.load_csv_file = load_csv_file
    main_window.clear_layout = clear_layout
    main_window.clear_rule_section = clear_rule_section
    main_window.create_rule_section = create_rule_section
    main_window.handle_combo_selection_for_rules = handle_combo_selection_for_rules
    main_window.set_rule_parameter = set_rule_parameter
    main_window.add_rule_to_analysis = add_rule_to_analysis
    main_window.remove_rule_parameter = remove_rule_parameter
    main_window.get_parameter_path_xml = get_parameter_path_xml
    main_window.get_parameter_value_xml = get_parameter_value_xml
    main_window.get_rule_value = get_rule_value
    main_window.set_parameter_value = set_parameter_value
    main_window.add_parameter_to_analysis = add_parameter_to_analysis
    main_window.remove_parameter = remove_parameter
    main_window.update_rules_file = update_rules_file
    # main_window.update_ini_preview = update_ini_preview
    main_window.update_preview_table = update_preview_table
    main_window.update_output_tab1 = update_output_tab1
    main_window.save_ini_file = save_ini_file
    main_window.load_ini_file = load_ini_file

    # Create a scroll area
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)  # Allow resizing

    # Create a container widget for the layout
    container_widget = QWidget()
    layout_tab1 = QVBoxLayout(container_widget)

    # Add the container widget to the scroll area
    scroll_area.setWidget(container_widget)

    # Button to load XML
    load_xml_hbox = QHBoxLayout()
    load_xml_label = QLabel("<b>PhysiCell configuration file</b>: ")
    load_xml_hbox.addWidget(load_xml_label)
    load_xml_button = QPushButton("Load PhysiCell .xml File")
    load_xml_button.setStyleSheet("background-color: lightgreen; color: black")
    load_xml_button.clicked.connect(lambda: main_window.load_xml_file(main_window))
    load_xml_hbox.addWidget(load_xml_button)
    # Add stretch to push everything to the left
    load_xml_hbox.addStretch()
    layout_tab1.addLayout(load_xml_hbox)

    # Separator line
    layout_tab1.addWidget(QLabel("<hr>"))

    # Add a line and title
    main_window.parameter_selection_label = QLabel("<b>Parameter Selection</b>")
    main_window.parameter_selection_label.setAlignment(Qt.AlignCenter)
    layout_tab1.addWidget(main_window.parameter_selection_label)

    # Horizontal layout for combo box
    main_window.combo_scroll = QScrollArea()
    main_window.combo_scroll.setWidgetResizable(True)
    main_window.combo_scroll_widget = QWidget()
    main_window.combo_hbox = QHBoxLayout(main_window.combo_scroll_widget)
    main_window.combo_scroll.setWidget(main_window.combo_scroll_widget)
    layout_tab1.addWidget(main_window.combo_scroll)
    main_window.combo_hbox.setAlignment(Qt.AlignLeft)
    main_window.combo_label = QLabel("Select Parameter:")
    main_window.combo_hbox.addWidget(main_window.combo_label)
    main_window.combo_box = QComboBox()
    main_window.combo_box.addItem("Select root...")
    main_window.combo_box.setEnabled(False)
    main_window.combo_hbox.addWidget(main_window.combo_box)
    main_window.combo_hbox.addStretch()
    
    # Create a group box for parameter details
    main_window.param_details_groupbox = QGroupBox("Parameter Details")
    param_details_layout = QVBoxLayout()
    # Label to display selected parameter path
    main_window.selected_param_label = QLabel("XML path: None")
    param_details_layout.addWidget(main_window.selected_param_label)
    # Label to display selected parameter value
    main_window.selected_value_label = QLabel("Current value: None")
    param_details_layout.addWidget(main_window.selected_value_label)
    # Set the layout for the group box
    main_window.param_details_groupbox.setLayout(param_details_layout)
    # Add the group box to the main layout
    layout_tab1.addWidget(main_window.param_details_groupbox)

    # Horizontal layout for value input and buttons
    main_window.value_hbox = QHBoxLayout()
    main_window.value_label = QLabel("New Value:")
    main_window.value_hbox.addWidget(main_window.value_label)
    main_window.new_value_input = QLineEdit()
    main_window.new_value_input.setPlaceholderText("Enter new value")
    main_window.new_value_input.setEnabled(False)
    main_window.value_hbox.addWidget(main_window.new_value_input)
    # Button to set parameter value
    main_window.set_param_button = QPushButton("Set Parameter")
    main_window.set_param_button.setStyleSheet("background-color: lightgreen; color: darkgray;")
    main_window.set_param_button.setEnabled(False)
    main_window.set_param_button.clicked.connect(lambda: main_window.set_parameter_value(main_window))
    main_window.value_hbox.addWidget(main_window.set_param_button)
    # Button to add parameter to analysis
    main_window.add_analysis_button = QPushButton("Add to Analysis")
    main_window.add_analysis_button.setEnabled(False)
    main_window.add_analysis_button.setStyleSheet("background-color: lightgreen; color: darkgray;")
    main_window.add_analysis_button.clicked.connect(lambda: main_window.add_parameter_to_analysis(main_window))
    main_window.value_hbox.addWidget(main_window.add_analysis_button)
    # Add text into new_value_input enables the set parameter button and disables it if empty (oposite for add_analysis_button)
    main_window.new_value_input.textChanged.connect(
        lambda: [
            main_window.set_param_button.setEnabled(main_window.new_value_input.text().strip() != ""),
            main_window.set_param_button.setStyleSheet(
                "background-color: lightgreen; color: black;" if main_window.new_value_input.text().strip() 
                else "background-color: lightgreen; color: darkgray;"
            ),
            main_window.add_analysis_button.setEnabled(not main_window.set_param_button.isEnabled()),
            main_window.add_analysis_button.setStyleSheet(
                "background-color: lightgreen; color: black;" if not main_window.new_value_input.text().strip() 
                else "background-color: lightgreen; color: darkgray;"
            )
        ]
    )
    main_window.remove_param_button = QPushButton("Remove Parameter")
    main_window.remove_param_button.setEnabled(False)
    main_window.remove_param_button.setStyleSheet("background-color: yellow; color: black")
    main_window.remove_param_button.clicked.connect(lambda: main_window.remove_parameter(main_window))
    main_window.value_hbox.addWidget(main_window.remove_param_button)

    # Add space between buttons
    main_window.value_hbox.addStretch()
    main_window.load_rules_button = QPushButton("Load Rules CSV")
    main_window.load_rules_button.setEnabled(False)
    main_window.load_rules_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.load_rules_button.clicked.connect(lambda: main_window.load_csv_file(main_window))
    main_window.value_hbox.addWidget(main_window.load_rules_button)
    layout_tab1.addLayout(main_window.value_hbox)

    # Add the rule section here
    main_window.rule_section_vbox = QVBoxLayout()
    main_window.rule_section_vbox.addStretch()
    layout_tab1.addLayout(main_window.rule_section_vbox)

    # Separator line
    layout_tab1.addWidget(QLabel("<hr>"))

    # Preview table
    main_window.ini_preview_label = QLabel("<b> Preview Table</b>")
    main_window.ini_preview_label.setAlignment(Qt.AlignCenter)
    layout_tab1.addWidget(main_window.ini_preview_label)
    main_window.preview_table = QTableWidget()
    main_window.preview_table.setColumnCount(4)
    main_window.preview_table.setHorizontalHeaderLabels(["Parameter Path", "Value", "Name", "Place"])
    main_window.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table non-editable
    header = main_window.preview_table.horizontalHeader()
    for col in range(main_window.preview_table.columnCount()):
        header.setSectionResizeMode(col, QHeaderView.Stretch)
    # Set gray background for header
    header = main_window.preview_table.horizontalHeader()
    header.setStyleSheet("QHeaderView::section { background-color: lightgray; color: black; font-weight: bold; }")
    layout_tab1.addWidget(main_window.preview_table)

    # Separator line
    layout_tab1.addWidget(QLabel("<hr>"))

    # Display section
    main_window.output_label = QLabel("<b>Display</b>")
    main_window.output_label.setAlignment(Qt.AlignCenter)
    layout_tab1.addWidget(main_window.output_label)

    main_window.output_text = QTextEdit()
    main_window.output_text.setReadOnly(True)
    main_window.output_text.setMinimumHeight(100)
    layout_tab1.addWidget(main_window.output_text)

    # Horizontal layout for mandatory fields and save button
    main_window.save_hbox = QHBoxLayout()

    # Structure Name input
    main_window.struc_name_label = QLabel("Structure Name:")
    main_window.save_hbox.addWidget(main_window.struc_name_label)
    main_window.struc_name_input = QLineEdit()
    main_window.struc_name_input.setPlaceholderText("Enter structure name")
    main_window.save_hbox.addWidget(main_window.struc_name_input)

    # Executable Path input
    main_window.executable_path_label = QLabel("Executable Path:")
    main_window.save_hbox.addWidget(main_window.executable_path_label)
    main_window.executable_path_input = QLineEdit()
    main_window.executable_path_input.setEnabled(False)
    main_window.executable_path_input.setPlaceholderText("Enter executable path")
    main_window.executable_path_browse_button = QPushButton("Select")
    main_window.executable_path_browse_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.executable_path_browse_button.clicked.connect(lambda: main_window.executable_path_input.setText( os.path.relpath(QFileDialog.getOpenFileName(main_window, "Select Executable", "", "Executable Files (*)")[0], os.getcwd()) ))
    main_window.executable_path_input.setPlaceholderText("Enter executable path")
    main_window.save_hbox.addWidget(main_window.executable_path_input)
    main_window.save_hbox.addWidget(main_window.executable_path_browse_button)

    # Number of Replicates input
    main_window.num_replicates_label = QLabel("Num Replicates:")
    main_window.save_hbox.addWidget(main_window.num_replicates_label)
    main_window.num_replicates_input = QLineEdit()
    main_window.num_replicates_input.setPlaceholderText("Enter number of replicates")
    main_window.num_replicates_input.setValidator(QIntValidator(1, 1000))
    main_window.save_hbox.addWidget(main_window.num_replicates_input)

    # Save .ini File button
    main_window.save_ini_button = QPushButton("Save .ini File")
    main_window.save_ini_button.setStyleSheet("background-color: yellow; color: black")
    main_window.save_ini_button.clicked.connect(lambda: main_window.save_ini_file(main_window))
    main_window.save_hbox.addWidget(main_window.save_ini_button)

    layout_tab1.addLayout(main_window.save_hbox)

    return scroll_area

def create_combo_box(main_window, parent_node, label):
    # Create a new combo box for the given parent node
    combo_box = QComboBox()
    combo_box.addItem(f"Select {label}...")
    for child in parent_node:
        # Use the 'name' attribute if it exists, otherwise use the tag
        display_name = child.get("name", child.tag)
        # Append the 'index' attribute if it exists
        if "index" in child.attrib:
            display_name = f"{display_name}[{int(child.get('index')) + 1}]"
        combo_box.addItem(display_name)
    combo_box.currentIndexChanged.connect(lambda: main_window.handle_combo_selection(main_window, combo_box, parent_node))

    # Add "->" label only if there is already a combo box in the layout
    if main_window.combo_hbox.count() > 1:
        arrow_label = QLabel("\u2794")  # Unicode for a proper arrow (➔)
        arrow_label.setAlignment(Qt.AlignCenter)
        main_window.combo_hbox.addWidget(arrow_label)

    # Add the combo box to the layout
    main_window.combo_hbox.addWidget(combo_box)

def clear_combo_boxes(main_window, starting_index=1):
    # Clear combo boxes and "->" labels starting from the given index
    while main_window.combo_hbox.count() > starting_index:
        widget = main_window.combo_hbox.takeAt(starting_index).widget()
        if widget:
            widget.deleteLater()

def handle_combo_selection(main_window, combo_box, parent_node):
    # Handle selection in the combo box
    selected_display_name = combo_box.currentText()
    if selected_display_name.startswith("Select") or not selected_display_name:
        return

    # Find the selected child node
    for child in parent_node:
        # Match by 'name' attribute if it exists, otherwise by tag
        display_name = child.get("name", child.tag)
        if "index" in child.attrib:
            display_name = f"{display_name}[{int(child.get('index')) + 1}]"
        if display_name == selected_display_name:
            # Clear combo boxes below the current one
            main_window.clear_combo_boxes(main_window, starting_index=main_window.combo_hbox.indexOf(combo_box) + 1)

            # Check if the selected node has children
            if len(child) > 0:
                # Create a new combo box for the child node
                main_window.create_combo_box(main_window, child, selected_display_name)
            else:
                # Leaf node reached, display path and value
                main_window.current_leaf_node = child
                path = main_window.get_parameter_path_xml(main_window, child)
                value = child.text.strip() if child.text else "None"

                # Update the parameter and value labels
                main_window.update_selected_param_label(main_window, path, value)
            break


def update_selected_param_label(main_window, path, value):
    # Determine the displayed value
    new_value = ''
    if path in main_window.fixed_parameters:
        new_value += f"\u2794 {main_window.fixed_parameters[path]}"
    elif path in main_window.analysis_parameters:
        new_value += "\u2794 <analysis>"

    # Update the labels
    main_window.selected_param_label.setText(f"XML path: {path}")
    main_window.selected_value_label.setText(f"Current value: {value} {new_value}")

def clear_rule_section(main_window):
    # Clear all widgets in the rule_section_vbox
    while main_window.rule_section_vbox.count():
        item = main_window.rule_section_vbox.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            main_window.clear_layout(main_window, item.layout())

def create_rule_section(main_window):
    # Create the rules section below handle_combo_selection
    if not hasattr(main_window, 'csv_data') or main_window.csv_data.empty:
        main_window.update_output_tab1(main_window, "Error: No CSV data loaded.")
        return

    # Ensure the rule_section_vbox is cleared before adding new widgets
    main_window.clear_rule_section(main_window)

    # Separator line
    main_window.rule_section_vbox.addWidget(QLabel("<hr>"))

    # Add a title for the rules section
    rules_title_label = QLabel("<b>Rules Parameters</b>")
    rules_title_label.setAlignment(Qt.AlignCenter)
    main_window.rule_section_vbox.addWidget(rules_title_label)

    # Layout for rule combo boxes
    rule_hbox = QHBoxLayout()

    # Create combo boxes for cell, signal, direction, behavior, and parameter
    main_window.cell_combo = QComboBox()
    main_window.signal_combo = QComboBox()
    main_window.direction_combo = QComboBox()
    main_window.behavior_combo = QComboBox()
    main_window.parameter_combo = QComboBox()

    # Populate the first four combo boxes with unique values from the respective columns
    main_window.cell_combo.addItems(main_window.csv_data.iloc[:, 0].unique())
    main_window.signal_combo.addItems(main_window.csv_data.iloc[:, 1].unique())
    main_window.direction_combo.addItems(main_window.csv_data.iloc[:, 2].unique())
    main_window.behavior_combo.addItems(main_window.csv_data.iloc[:, 3].unique())

    # Populate the fifth combo box with options for parameters
    main_window.parameter_combo.addItems(["saturation", "half_max", "hill_power", "dead", "inactive"])

    # Connect combo boxes to handle selection
    main_window.cell_combo.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window))
    main_window.signal_combo.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window))
    main_window.direction_combo.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window))
    main_window.behavior_combo.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window))
    main_window.parameter_combo.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window))

    # Add combo boxes to the layout
    rule_hbox.addWidget(QLabel("Cell:"))
    rule_hbox.addWidget(main_window.cell_combo)
    rule_hbox.addWidget(QLabel("Signal:"))
    rule_hbox.addWidget(main_window.signal_combo)
    rule_hbox.addWidget(QLabel("Direction:"))
    rule_hbox.addWidget(main_window.direction_combo)
    rule_hbox.addWidget(QLabel("Behavior:"))
    rule_hbox.addWidget(main_window.behavior_combo)
    rule_hbox.addWidget(QLabel("Parameter:"))
    rule_hbox.addWidget(main_window.parameter_combo)

    # Add the rule layout to the main layout
    main_window.rule_section_vbox.addLayout(rule_hbox)

    # Group box for rule details
    rule_details_groupbox = QGroupBox("Rule Details")
    rule_details_layout = QVBoxLayout()
    # Label to display selected rule path
    main_window.selected_rule_label = QLabel("Rule Path: None")
    rule_details_layout.addWidget(main_window.selected_rule_label)
    # Label to display selected rule value
    main_window.selected_rule_value_label = QLabel("Rule Value: None")
    rule_details_layout.addWidget(main_window.selected_rule_value_label)
    # Set the layout for the group box
    rule_details_groupbox.setLayout(rule_details_layout)
    # Add the group box to the main layout
    main_window.rule_section_vbox.addWidget(rule_details_groupbox)

    # Horizontal layout for rule buttons
    rule_buttons_hbox = QHBoxLayout()

    # Add new value input for rules
    main_window.new_value_input_rule = QLineEdit()
    main_window.new_value_input_rule.setPlaceholderText("Enter new value")
    rule_buttons_hbox.addWidget(main_window.new_value_input_rule)

    # Add buttons for setting, adding to analysis, and removing rules parameters
    set_rule_button = QPushButton("Set Rule Parameter")
    set_rule_button.clicked.connect(lambda: main_window.set_rule_parameter(main_window))
    set_rule_button.setStyleSheet("background-color: lightgreen; color: black")
    rule_buttons_hbox.addWidget(set_rule_button)

    add_rule_analysis_button = QPushButton("Add Rule to Analysis")
    add_rule_analysis_button.clicked.connect(lambda: main_window.add_rule_to_analysis(main_window))
    add_rule_analysis_button.setStyleSheet("background-color: lightgreen; color: black")
    rule_buttons_hbox.addWidget(add_rule_analysis_button)

    remove_rule_button = QPushButton("Remove Rule Parameter")
    remove_rule_button.clicked.connect(lambda: main_window.remove_rule_parameter(main_window))
    remove_rule_button.setStyleSheet("background-color: yellow; color: black")
    rule_buttons_hbox.addWidget(remove_rule_button)

    # Add text into new_value_input_rule enables the set parameter button and disables it if empty (oposite for add_analysis_button)
    main_window.new_value_input_rule.textChanged.connect(
        lambda: [
            set_rule_button.setEnabled(main_window.new_value_input_rule.text().strip() != ""),
            set_rule_button.setStyleSheet(
                "background-color: lightgreen; color: black;" if main_window.new_value_input_rule.text().strip() 
                else "background-color: lightgreen; color: darkgray;"
            ),
            add_rule_analysis_button.setEnabled(not set_rule_button.isEnabled()),
            add_rule_analysis_button.setStyleSheet(
                "background-color: lightgreen; color: black;" if not main_window.new_value_input_rule.text().strip() 
                else "background-color: lightgreen; color: darkgray;"
            )
        ]
    )

    # Add the buttons layout to the main layout
    main_window.rule_section_vbox.addLayout(rule_buttons_hbox)

    # Update Rule Path and Rule Value based on combo box selections
    main_window.handle_combo_selection_for_rules(main_window)

def clear_layout(main_window, layout):
    # Recursively clear all items in a layout
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            main_window.clear_layout(main_window, item.layout())

def handle_combo_selection_for_rules(main_window):
    # Update Rule Path and Rule Value based on combo box selections
    try:
        cell = main_window.cell_combo.currentText()
        signal = main_window.signal_combo.currentText()
        direction = main_window.direction_combo.currentText()
        behavior = main_window.behavior_combo.currentText()
        parameter = main_window.parameter_combo.currentText()

        # Find the corresponding row in the CSV
        rule_row = main_window.csv_data[
            (main_window.csv_data.iloc[:, 0] == cell) &
            (main_window.csv_data.iloc[:, 1] == signal) &
            (main_window.csv_data.iloc[:, 2] == direction) &
            (main_window.csv_data.iloc[:, 3] == behavior)
        ]

        if rule_row.empty:
            main_window.selected_rule_label.setText("Rule: None")
            main_window.selected_rule_value_label.setText("Value: None")
            return

        # Extract the respective value for the selected parameter
        if parameter == "saturation":
            value = rule_row.iloc[0, 4]
        elif parameter == "half_max":
            value = rule_row.iloc[0, 5]
        elif parameter == "hill_power":
            value = rule_row.iloc[0, 6]
        elif parameter == "dead":
            value = rule_row.iloc[0, 7]
        elif parameter == "inactive":
            value = rule_row.iloc[0, 8]
        else:
            value = "None"

        # Determine the displayed value
        rule_key = f"{cell},{signal},{direction},{behavior},{parameter}"
        new_value = ''
        if rule_key in main_window.fixed_rules_parameters:
            new_value += f"\u2794 {main_window.fixed_rules_parameters[rule_key]}"
        elif rule_key in main_window.analysis_rules_parameters:
            new_value += "\u2794 <analysis>"

        # Update Rule Path and Rule Value
        main_window.selected_rule_label.setText(f"Rule: {rule_key}")
        main_window.selected_rule_value_label.setText(f"Value: {value} {new_value}")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error handling rule selection: {e}")

def set_rule_parameter(main_window):
    # Set a fixed value for the selected rule parameter
    try:
        cell = main_window.cell_combo.currentText()
        signal = main_window.signal_combo.currentText()
        direction = main_window.direction_combo.currentText()
        behavior = main_window.behavior_combo.currentText()
        parameter = main_window.parameter_combo.currentText()
        new_value = main_window.new_value_input_rule.text()

        if not new_value:
            main_window.update_output_tab1(main_window, "Error: New value is required.")
            return

        # Find the corresponding row in the CSV
        rule_row = main_window.csv_data[
            (main_window.csv_data.iloc[:, 0] == cell) &
            (main_window.csv_data.iloc[:, 1] == signal) &
            (main_window.csv_data.iloc[:, 2] == direction) &
            (main_window.csv_data.iloc[:, 3] == behavior)
        ]

        if rule_row.empty:
            main_window.update_output_tab1(main_window, "Error: No matching rule found in the CSV.")
            return

        # Extract the respective columns for the selected parameter
        if parameter == "saturation":
            column_index = 4
        elif parameter == "half_max":
            column_index = 5
        elif parameter == "hill_power":
            column_index = 6
        elif parameter == "dead":
            column_index = 7
        elif parameter == "inactive":
            column_index = 8
        else:
            main_window.update_output_tab1(main_window, "Error: Invalid parameter selected.")
            return

        # Update the rule parameter
        rule_key = f"{cell},{signal},{direction},{behavior},{parameter}"
        main_window.fixed_rules_parameters[rule_key] = new_value
        old_value = main_window.csv_data.iloc[rule_row.index[0], column_index]
        main_window.selected_rule_label.setText(f"Rule Path: {rule_key}")
        main_window.selected_rule_value_label.setText(f"Rule Value: {old_value} \u2794 {new_value}")
        main_window.update_preview_table(main_window)
        main_window.update_output_tab1(main_window, f"Set rule parameter '{rule_key}' to value '{new_value}'.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error setting rule parameter: {e}")

def add_rule_to_analysis(main_window):
   # Add the selected rule parameter to the analysis
    try:
        cell = main_window.cell_combo.currentText()
        signal = main_window.signal_combo.currentText()
        direction = main_window.direction_combo.currentText()
        behavior = main_window.behavior_combo.currentText()
        parameter = main_window.parameter_combo.currentText()

        # Find the corresponding row in the CSV
        rule_row = main_window.csv_data[
            (main_window.csv_data.iloc[:, 0] == cell) &
            (main_window.csv_data.iloc[:, 1] == signal) &
            (main_window.csv_data.iloc[:, 2] == direction) &
            (main_window.csv_data.iloc[:, 3] == behavior)
        ]

        if rule_row.empty:
            main_window.update_output_tab1(main_window, "Error: No matching rule found in the CSV.")
            return

        # Extract the respective columns for the selected parameter
        if parameter == "saturation":
            old_value = rule_row.iloc[0, 4]
        elif parameter == "half_max":
            old_value = rule_row.iloc[0, 5]
        elif parameter == "hill_power":
            old_value = rule_row.iloc[0, 6]
        elif parameter == "dead":
            old_value = rule_row.iloc[0, 7]
        elif parameter == "inactive":
            old_value = rule_row.iloc[0, 8]
        else:
            main_window.update_output_tab1(main_window, "Error: Invalid parameter selected.")
            return

        # Add the rule to the analysis
        rule_key = f"{cell},{signal},{direction},{behavior},{parameter}"
        # Add the selected parameter to the analysis
        friendly_name, ok = QInputDialog.getText(main_window, "Add Parameter to Analysis", "Enter a friendly name:")
        if ok and friendly_name:
            main_window.analysis_rules_parameters[rule_key] = [None, friendly_name]
            main_window.selected_rule_label.setText(f"Rule Path: {rule_key}")
            main_window.selected_rule_value_label.setText(f"Rule Value: {old_value} \u2794 <analysis>")
            main_window.update_preview_table(main_window)
            main_window.update_output_tab1(main_window, f"Added rule '{rule_key}' to analysis.")
        else:
            main_window.update_output_tab1(main_window, "Error: Friendly name is required.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error adding rule to analysis: {e}")

def remove_rule_parameter(main_window):
    # Remove the selected rule parameter from fixed or analysis parameters
    try:
        cell = main_window.cell_combo.currentText()
        signal = main_window.signal_combo.currentText()
        direction = main_window.direction_combo.currentText()
        behavior = main_window.behavior_combo.currentText()
        parameter = main_window.parameter_combo.currentText()

        rule_key = f"{cell},{signal},{direction},{behavior},{parameter}"

        if rule_key in main_window.fixed_rules_parameters:
            del main_window.fixed_rules_parameters[rule_key]
            main_window.update_output_tab1(main_window, f"Removed fixed rule parameter '{rule_key}'.")

        if rule_key in main_window.analysis_rules_parameters:
            del main_window.analysis_rules_parameters[rule_key]
            main_window.update_output_tab1(main_window, f"Removed analysis rule parameter '{rule_key}'.")

        # Update the selected rule label and value
        main_window.update_preview_table(main_window)
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error removing rule parameter: {e}")

def get_parameter_path_xml(main_window, node):
    # Recursively get the XML tree path of the given node in a format compatible with xml_root.findall()
    path = []
    while node is not None:
        # Skip the root tag 'PhysiCell_settings'
        if node.tag == "PhysiCell_settings":
            break
        # Use 'name' attribute if it exists, otherwise use the tag
        node_name = node.tag
        if "name" in node.attrib:
            node_name += f"[@name='{node.get('name')}']"
        # Append 'index' attribute if it exists
        if "index" in node.attrib:
            node_name += f"[{int(node.get('index')) + 1}]"
        path.insert(0, node_name)
        node = main_window.parent_map.get(node)  # Use the parent map to find the parent
    return ".//" + "/".join(path)

def get_parameter_value_xml(main_window, path):
    # Retrieve the default value from the XML file for a given path
    try:
        element = main_window.xml_tree.find(path)
        if element is not None and element.text:
            return element.text.strip()
        else:
            main_window.update_output_tab1(main_window, f"Warning: No value found for path '{path}' in the XML.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error retrieving XML value for path '{path}': {e}")
    return None

def get_rule_value(main_window, rule_key):
    # Retrieve the default value from the rules CSV for a given rule key
    try:
        cell, signal, direction, behavior, parameter = rule_key.split(",")
        rule_row = main_window.csv_data[
            (main_window.csv_data.iloc[:, 0] == cell) &
            (main_window.csv_data.iloc[:, 1] == signal) &
            (main_window.csv_data.iloc[:, 2] == direction) &
            (main_window.csv_data.iloc[:, 3] == behavior)
        ]
        if not rule_row.empty:
            if parameter == "saturation":
                return rule_row.iloc[0, 4]
            elif parameter == "half_max":
                return rule_row.iloc[0, 5]
            elif parameter == "hill_power":
                return rule_row.iloc[0, 6]
            elif parameter == "dead":
                return rule_row.iloc[0, 7]
            elif parameter == "inactive":
                return rule_row.iloc[0, 8]
        else:
            main_window.update_output_tab1(main_window, f"Warning: No matching rule found for key '{rule_key}' in the CSV.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error retrieving rule value for key '{rule_key}': {e}")
    return None

def set_parameter_value(main_window):
    # Set a fixed value for the selected parameter
    new_value = main_window.new_value_input.text()
    if new_value:
        try:
            path = main_window.get_parameter_path_xml(main_window, main_window.current_leaf_node)
            main_window.fixed_parameters[path] = new_value
            main_window.new_value_input.clear()
            main_window.update_preview_table(main_window)
            main_window.update_selected_param_label(main_window, path, main_window.current_leaf_node.text.strip() if main_window.current_leaf_node.text else "None")
            main_window.update_output_tab1(main_window, f"Set parameter '{path}' to value '{new_value}'")
            # Check if the parameter is a rules file or folder
            if path == ".//cell_rules/rulesets/ruleset/filename" or path == ".//cell_rules/rulesets/ruleset/folder":
                main_window.update_rules_file(main_window)
        except Exception as e:
            main_window.update_output_tab1(main_window, f"Error setting parameter: {e}")

def add_parameter_to_analysis(main_window):
    # Add the selected parameter to the analysis
    friendly_name, ok = QInputDialog.getText(main_window, "Add Parameter to Analysis", "Enter a friendly name:")
    if ok and friendly_name:
        try:
            path = main_window.get_parameter_path_xml(main_window, main_window.current_leaf_node)
            main_window.analysis_parameters[path] = [None, friendly_name]
            main_window.update_preview_table(main_window)
            main_window.update_selected_param_label(main_window, path, main_window.current_leaf_node.text.strip() if main_window.current_leaf_node.text else "None")
            main_window.update_output_tab1(main_window, f"Added parameter '{path}' to analysis with friendly name '{friendly_name}'")
        except Exception as e:
            main_window.update_output_tab1(main_window, f"Error adding parameter to analysis: {e}")

def remove_parameter(main_window):
    # Remove the selected parameter from fixed or analysis parameters
    try:
        path = main_window.get_parameter_path_xml(main_window, main_window.current_leaf_node)
        if path in main_window.fixed_parameters:
            del main_window.fixed_parameters[path]
            main_window.update_output_tab1(main_window, f"Removed fixed parameter '{path}'")
        if path in main_window.analysis_parameters:
            del main_window.analysis_parameters[path]
            main_window.update_output_tab1(main_window, f"Removed analysis parameter '{path}'")
        main_window.update_preview_table(main_window)
        main_window.update_selected_param_label(main_window, path, main_window.current_leaf_node.text.strip() if main_window.current_leaf_node.text else "None")
        # Check if the parameter is a rules file or folder
        if path == ".//cell_rules/rulesets/ruleset/filename" or path == ".//cell_rules/rulesets/ruleset/folder":
            main_window.update_rules_file(main_window)
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error removing parameter: {e}")

def update_preview_table(main_window):
    # Clear the preview table
    main_window.preview_table.setRowCount(0)
    # Update the preview table with the current parameters
    if (main_window.fixed_parameters or main_window.analysis_parameters or main_window.fixed_rules_parameters or main_window.analysis_rules_parameters):
        # Set the number of rows in the preview table
        total_rows = (len(main_window.fixed_parameters) + len(main_window.analysis_parameters) +
                      len(main_window.fixed_rules_parameters) + len(main_window.analysis_rules_parameters))
        main_window.preview_table.setRowCount(total_rows)
        row_position = 0
        # Fixed parameters
        for path, value in main_window.fixed_parameters.items():
            main_window.preview_table.setItem(row_position, 0, QTableWidgetItem(path))
            main_window.preview_table.setItem(row_position, 1, QTableWidgetItem(str(value)))
            main_window.preview_table.setItem(row_position, 2, QTableWidgetItem(""))
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem("XML"))
            row_position += 1

        # Analysis parameters
        for path, value in main_window.analysis_parameters.items():
            main_window.preview_table.setItem(row_position, 0, QTableWidgetItem(path))
            main_window.preview_table.setItem(row_position, 1, QTableWidgetItem("<variable>"))
            main_window.preview_table.setItem(row_position, 2, QTableWidgetItem(str(value[1])))
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem("XML"))
            row_position += 1

        # Fixed rules parameters
        for path, value in main_window.fixed_rules_parameters.items():
            main_window.preview_table.setItem(row_position, 0, QTableWidgetItem(path))
            main_window.preview_table.setItem(row_position, 1, QTableWidgetItem(str(value)))
            main_window.preview_table.setItem(row_position, 2, QTableWidgetItem(""))
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem("CSV"))
            row_position += 1

        # Analysis rules parameters
        for path, value in main_window.analysis_rules_parameters.items():
            main_window.preview_table.setItem(row_position, 0, QTableWidgetItem(path))
            main_window.preview_table.setItem(row_position, 1, QTableWidgetItem("<variable>"))
            main_window.preview_table.setItem(row_position, 2, QTableWidgetItem(str(value[1])))
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem("CSV"))
            row_position += 1



def update_ini_preview(main_window):
    # Update the preview of the .ini file with proper line breaks
    if (main_window.fixed_parameters): preview = "[parameters]\n"
    else: preview = ""
    for path, value in main_window.fixed_parameters.items():
        preview += f'"{path}" = {value}\n'
    if (main_window.analysis_parameters): preview += "\n[parameters_analysis]\n"
    for path, value in main_window.analysis_parameters.items():
        preview += f'"{path}" = {value}\n'
    if (main_window.fixed_rules_parameters): preview += "\n[parameters_rules_fixed]\n"
    for path, value in main_window.fixed_rules_parameters.items():
        preview += f'"{path}" = {value}\n'
    if (main_window.analysis_rules_parameters): preview += "\n[parameters_rules_analysis]\n"
    for path, value in main_window.analysis_rules_parameters.items():
        preview += f'"{path}" = {value}\n'
    main_window.ini_preview_text.setText(preview)  # QTextEdit supports multiline text

def update_output_tab1(main_window, message):
    # Update the output section with a new message
    main_window.output_text.append(message)

def save_ini_file(main_window):
    # Ensure all mandatory fields are filled
    struc_name = main_window.struc_name_input.text().strip()
    executable_path = main_window.executable_path_input.text().strip()
    num_replicates = main_window.num_replicates_input.text().strip()

    if not struc_name:
        main_window.update_output_tab1(main_window, "Error: Structure name is required.")
        return
    if not executable_path:
        main_window.update_output_tab1(main_window, "Error: Executable path is required.")
        return
    if not num_replicates:
        main_window.update_output_tab1(main_window, "Error: Number of replicates is required.")
        return

    # Ensure an XML file has been loaded
    if not hasattr(main_window, 'xml_file_path') or not main_window.xml_file_path:
        main_window.update_output_tab1(main_window, "Error: No XML file loaded.")
        return

    # Save the parameters to a .ini file
    options = QFileDialog.Options()
    file_path, _ = QFileDialog.getSaveFileName(main_window, "Save .ini File", "", "INI Files (*.ini);;All Files (*)", options=options)
    file_path = os.path.relpath(file_path, os.getcwd())  # Convert to relative path
    if file_path:
        try:
            config = configparser.ConfigParser()
            if os.path.exists(file_path):
                config.read(file_path)
                if struc_name in config.sections():
                    # Ask the user if they want to overwrite
                    reply = QMessageBox.question(main_window, "Overwrite Confirmation", 
                                                    f"Structure '{struc_name}' already exists. Do you want to overwrite it?", 
                                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        # Remove the existing section
                        config.remove_section(struc_name)
                    else:
                        main_window.update_output_tab1(main_window, f"Aborted saving. Structure '{struc_name}' was not overwritten.")
                        return

            # Add the new section
            config[struc_name] = {
                "executable":  "./" + os.path.relpath(executable_path, os.getcwd()), # Ensure executable path is relative and prefixed with ./
                "configFile_ref": os.path.relpath(main_window.xml_file_path, os.getcwd()),
                "numReplicates": num_replicates
            }

            # Add parameters if applicable
            if (main_window.fixed_parameters or main_window.analysis_parameters):
                parameters = {path: value for path, value in main_window.fixed_parameters.items()}
                parameters.update({path: value for path, value in main_window.analysis_parameters.items()})
                config[struc_name]["parameters"] = str(parameters)

            # Add rules if applicable
            if (main_window.fixed_rules_parameters or main_window.analysis_rules_parameters):
                config[struc_name]["rulesFile_ref"] = os.path.relpath(main_window.rule_path, os.getcwd())
                rules_parameters = {key: value for key, value in main_window.fixed_rules_parameters.items()}
                rules_parameters.update({key: value for key, value in main_window.analysis_rules_parameters.items()})
                config[struc_name]["parameters_rules"] = str(rules_parameters)

            # Write to the file
            with open(file_path, "w") as ini_file:
                config.write(ini_file)

            main_window.update_output_tab1(main_window, f"Successfully saved .ini file to: {file_path}")
            # Load the .ini file
            main_window.load_ini_file(main_window, filePath=file_path, strucName=struc_name)
        except Exception as e:
            main_window.update_output_tab1(main_window, f"Error saving .ini file: {e}")
