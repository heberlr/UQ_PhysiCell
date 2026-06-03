from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QComboBox, QFileDialog, QInputDialog, QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
import os
import pandas as pd
import configparser
import xml.etree.ElementTree as ET

from .load_files import load_xml_file, load_rules_file, update_rules_file, load_ini_file


def _set_executable_path_from_dialog(main_window):
    selected_path, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select Executable",
        "",
        "Executable Files (*)"
    )
    if not selected_path:
        return
    main_window.executable_path_input.setText(os.path.relpath(selected_path, os.getcwd()))


class PreviewTableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._header_initialized = False

    def showEvent(self, event):
        super().showEvent(event)
        if self._header_initialized:
            return

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.resizeSections(QHeaderView.Stretch)
        for col in range(self.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.Interactive)
        self._header_initialized = True

def create_tab1(main_window):
    # Add the following methods to the main_window instance
    main_window.load_xml_file = load_xml_file
    main_window.create_combo_box = create_combo_box
    main_window.clear_combo_boxes = clear_combo_boxes
    main_window.handle_combo_selection = handle_combo_selection
    main_window.update_selected_param_label = update_selected_param_label
    main_window.load_rules_file = load_rules_file
    main_window.clear_layout = clear_layout
    main_window.clear_rule_section = clear_rule_section
    main_window.create_rule_section = create_rule_section
    main_window._create_rule_combo_box = _create_rule_combo_box
    main_window._clear_rule_combo_boxes = _clear_rule_combo_boxes
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
    main_window.add_analysis_button.setEnabled(True)
    main_window.add_analysis_button.setStyleSheet("background-color: lightgreen; color: black;")
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
    main_window.load_rules_button = QPushButton("Load Rules")
    main_window.load_rules_button.setEnabled(False)
    main_window.load_rules_button.setStyleSheet("background-color: lightgreen; color: black")
    main_window.load_rules_button.clicked.connect(lambda: main_window.load_rules_file(main_window))
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
    main_window.preview_table = PreviewTableWidget()
    main_window.preview_table.setColumnCount(4)
    main_window.preview_table.setHorizontalHeaderLabels(["Parameter Path", "Value", "Name", "Place"])
    main_window.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table non-editable
    header = main_window.preview_table.horizontalHeader()
    for col in range(main_window.preview_table.columnCount()):
        header.setSectionResizeMode(col, QHeaderView.Interactive)
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
    main_window.executable_path_browse_button.clicked.connect(lambda: _set_executable_path_from_dialog(main_window))
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
        # Show the 'index' attribute if it exists
        if "index" in child.attrib:
            display_name = f"{display_name}[{int(child.get('index')) + 1}]"
        # Show the 'ID' attribute if it exists
        elif "ID" in child.attrib:
            display_name = f"{display_name}[@ID='" + child.get('ID') + "']"
        # Generic
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
        elif "ID" in child.attrib:
            display_name = f"{display_name}[@ID='{child.get('ID')}']"
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


def _rule_display_name_xml(node):
    display_name = node.get("name", node.tag)
    if "index" in node.attrib:
        display_name = f"{display_name}[{int(node.get('index')) + 1}]"
    elif "ID" in node.attrib:
        display_name = f"{display_name}[@ID='{node.get('ID')}']"
    return display_name


def _rule_path_xml(main_window, node):
    path = []
    while node is not None:
        if node.tag in ("PhysiCell_settings", "behavior_rulesets"):
            break
        # Use as many attributes as possible to ensure uniqueness of the node in the path
        node_name = node.tag
        if "name" in node.attrib:
            node_name += f"[@name='{node.get('name')}']"
        if "index" in node.attrib:
            node_name += f"[{int(node.get('index')) + 1}]"
        if "ID" in node.attrib:
            node_name += f"[@ID='{node.get('ID')}']"
        path.insert(0, node_name)
        node = main_window.rule_parent_map.get(node)
    return ".//" + "/".join(path)


def _rule_value_xml(node):
    if "enabled" in node.attrib:
        return node.get("enabled")
    if "value" in node.attrib:
        return node.get("value")
    return node.text.strip() if node.text else "None"


def _build_rules_tree_from_csv(csv_data):
    rules_tree = {}
    for _, row in csv_data.iterrows():
        behavior_node = (
            rules_tree
            .setdefault(str(row["cell_type"]), {})
            .setdefault(str(row["signal"]), {})
            .setdefault(str(row["direction"]), {})
            .setdefault(str(row["behavior"]), {})
        )
        for parameter in ["saturation", "half_max", "hill_power", "dead", "inactive"]:
            if parameter in row.index:
                behavior_node[parameter] = {
                    "__leaf__": True,
                    "path": f"{row['cell_type']},{row['signal']},{row['direction']},{row['behavior']},{parameter}",
                    "value": row[parameter],
                    "format": "csv",
                }
    return rules_tree


def _build_rules_tree_from_xml(xml_root, main_window):
    def recurse(node):
        children = list(node)
        if not children:
            path = _rule_path_xml(main_window, node)
            value = _rule_value_xml(node)
            main_window.rule_value_map[path] = value
            return {
                "__leaf__": True,
                "path": path,
                "value": value,
                "format": "xml",
            }
        branch = {}
        for child in children:
            branch[_rule_display_name_xml(child)] = recurse(child)
        return branch

    rules_tree = {}
    for child in list(xml_root):
        rules_tree[_rule_display_name_xml(child)] = recurse(child)
    return rules_tree


def _rules_tree_keys(node):
    return [key for key in node.keys() if key != "__leaf__"]


def _rules_is_leaf(node):
    return isinstance(node, dict) and node.get("__leaf__", False)


def _rules_selected_context(node):
    return {"path": node.get("path"), "value": node.get("value"), "format": node.get("format")}


def _clear_rule_combo_boxes(main_window, starting_index=0):
    while main_window.rule_combo_hbox.count() > starting_index:
        widget = main_window.rule_combo_hbox.takeAt(starting_index).widget()
        if widget:
            widget.deleteLater()


def _create_rule_combo_box(main_window, parent_node, label):
    combo_box = QComboBox()
    combo_box.addItem(f"Select {label}...")
    for child_name in _rules_tree_keys(parent_node):
        combo_box.addItem(child_name)
    combo_box.currentIndexChanged.connect(lambda: main_window.handle_combo_selection_for_rules(main_window, combo_box, parent_node))

    # Add label only for the first combo box
    if main_window.rule_combo_hbox.count() == 0:
        label_widget = QLabel(f"Select {label}:")
        main_window.rule_combo_hbox.addWidget(label_widget)
    else:
        arrow_label = QLabel("\u2794")
        arrow_label.setAlignment(Qt.AlignCenter)
        main_window.rule_combo_hbox.addWidget(arrow_label)

    main_window.rule_combo_hbox.addWidget(combo_box)
    return combo_box

def create_rule_section(main_window):
    # Create the rules section below handle_combo_selection
    if getattr(main_window, "rules_format", "csv") == "csv":
        if not hasattr(main_window, "csv_data") or main_window.csv_data.empty:
            main_window.update_output_tab1(main_window, "Error: No rules loaded.")
            return
        main_window.rules_tree = _build_rules_tree_from_csv(main_window.csv_data)
        main_window.rule_value_map = {
            f"{row['cell_type']},{row['signal']},{row['direction']},{row['behavior']},{parameter}": row[parameter]
            for _, row in main_window.csv_data.iterrows()
            for parameter in ["saturation", "half_max", "hill_power", "dead", "inactive"]
            if parameter in row.index
        }
    else:
        if not hasattr(main_window, 'rules_tree') or not main_window.rules_tree:
            main_window.update_output_tab1(main_window, "Error: No rules loaded.")
            return
        if isinstance(main_window.rules_tree, ET.Element):
            main_window.rule_value_map = {}
            main_window.rules_tree = _build_rules_tree_from_xml(main_window.rules_tree, main_window)
        elif not hasattr(main_window, "rule_value_map"):
            main_window.rule_value_map = {}

    if not hasattr(main_window, 'rules_tree') or not main_window.rules_tree:
        main_window.update_output_tab1(main_window, "Error: No rules loaded.")
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
    main_window.rule_combo_hbox = QHBoxLayout()
    main_window.rule_combo_hbox.setAlignment(Qt.AlignLeft)
    main_window.rule_section_vbox.addLayout(main_window.rule_combo_hbox)

    main_window.current_rule_context = None
    main_window.current_rule_leaf = None
    _create_rule_combo_box(main_window, main_window.rules_tree, "Rule")

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
    main_window.new_value_label_input_rule = QLabel("New Value:")
    main_window.new_value_input_rule = QLineEdit()
    main_window.new_value_input_rule.setPlaceholderText("Enter new value")
    rule_buttons_hbox.addWidget(main_window.new_value_label_input_rule)
    rule_buttons_hbox.addWidget(main_window.new_value_input_rule)

    # Add buttons for setting, adding to analysis, and removing rules parameters
    set_rule_button = QPushButton("Set Rule Parameter")
    set_rule_button.clicked.connect(lambda: main_window.set_rule_parameter(main_window))
    set_rule_button.setStyleSheet("background-color: lightgreen; color: black")
    rule_buttons_hbox.addWidget(set_rule_button)

    add_rule_analysis_button = QPushButton("Add to Analysis")
    add_rule_analysis_button.clicked.connect(lambda: main_window.add_rule_to_analysis(main_window))
    add_rule_analysis_button.setStyleSheet("background-color: lightgreen; color: black")
    rule_buttons_hbox.addWidget(add_rule_analysis_button)

    remove_rule_button = QPushButton("Remove Rule Parameter")
    remove_rule_button.clicked.connect(lambda: main_window.remove_rule_parameter(main_window))
    remove_rule_button.setStyleSheet("background-color: yellow; color: black")
    rule_buttons_hbox.addWidget(remove_rule_button)

    # Keep controls fixed-width and absorb extra space at the end of the row.
    rule_buttons_hbox.addStretch()

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

    # Initialize the labels for the default selection state
    main_window.selected_rule_label.setText("Rule Path: None")
    main_window.selected_rule_value_label.setText("Rule Value: None")

def clear_layout(main_window, layout):
    # Recursively clear all items in a layout
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            main_window.clear_layout(main_window, item.layout())

def handle_combo_selection_for_rules(main_window, combo_box=None, parent_node=None):
    # Update Rule Path and Rule Value based on nested rule selections
    try:
        if combo_box is None or parent_node is None:
            return

        selected_display_name = combo_box.currentText()
        if selected_display_name.startswith("Select") or not selected_display_name:
            return

        child = parent_node[selected_display_name]
        if _rules_is_leaf(child):
            main_window.current_rule_leaf = child
            main_window.current_rule_context = _rules_selected_context(child)
            rule_key = child["path"]
            value = child["value"]
            new_value = ''
            if rule_key in main_window.fixed_rules_parameters:
                new_value += f"\u2794 {main_window.fixed_rules_parameters[rule_key]}"
            elif rule_key in main_window.analysis_rules_parameters:
                new_value += "\u2794 <analysis>"
            main_window.selected_rule_label.setText(f"Rule: {rule_key}")
            main_window.selected_rule_value_label.setText(f"Value: {value} {new_value}")
            return

        main_window.current_rule_leaf = None
        main_window.current_rule_context = None
        main_window._clear_rule_combo_boxes(main_window, starting_index=main_window.rule_combo_hbox.indexOf(combo_box) + 1)
        if _rules_tree_keys(child):
            main_window._create_rule_combo_box(main_window, child, selected_display_name)
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error handling rule selection: {e}")

def set_rule_parameter(main_window):
    # Set a fixed value for the selected rule parameter
    try:
        new_value = main_window.new_value_input_rule.text()

        if not new_value:
            main_window.update_output_tab1(main_window, "Error: New value is required.")
            return

        if not getattr(main_window, "current_rule_context", None):
            main_window.update_output_tab1(main_window, "Error: No rule selected.")
            return

        rule_key = main_window.current_rule_context["path"]
        old_value = main_window.current_rule_context["value"]
        main_window.fixed_rules_parameters[rule_key] = new_value
        main_window.selected_rule_label.setText(f"Rule Path: {rule_key}")
        main_window.selected_rule_value_label.setText(f"Rule Value: {old_value} \u2794 {new_value}")
        main_window.update_preview_table(main_window)
        main_window.update_output_tab1(main_window, f"Set rule parameter '{rule_key}' to value '{new_value}'.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error setting rule parameter: {e}")

def add_rule_to_analysis(main_window):
   # Add the selected rule parameter to the analysis
    try:
        if not getattr(main_window, "current_rule_context", None):
            main_window.update_output_tab1(main_window, "Error: No rule selected.")
            return

        rule_key = main_window.current_rule_context["path"]
        old_value = main_window.current_rule_context["value"]
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
        if not getattr(main_window, "current_rule_context", None):
            main_window.update_output_tab1(main_window, "Error: No rule selected.")
            return

        rule_key = main_window.current_rule_context["path"]

        if rule_key in main_window.fixed_rules_parameters:
            del main_window.fixed_rules_parameters[rule_key]

        if rule_key in main_window.analysis_rules_parameters:
            del main_window.analysis_rules_parameters[rule_key]
        main_window.update_output_tab1(main_window, f"Removed rule parameter '{rule_key}'.")

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
        # Use many attributes as possible to ensure uniqueness of the node in the path
        node_name = node.tag
        if "name" in node.attrib:
            node_name += f"[@name='{node.get('name')}']"
        # Append 'index' attribute if it exists
        if "index" in node.attrib:
            node_name += f"[{int(node.get('index')) + 1}]"
        # Use ID attribute if it exists
        if "ID" in node.attrib:
            node_name += f"[@ID='{node.get('ID')}']"
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
    # Retrieve the default value from the loaded rules file for a given rule key
    try:
        if hasattr(main_window, "rule_value_map") and rule_key in main_window.rule_value_map:
            return main_window.rule_value_map[rule_key]
        main_window.update_output_tab1(main_window, f"Warning: No matching rule found for key '{rule_key}'.")
        print(f"Warning: No matching rule found for key '{rule_key}'.")
    except Exception as e:
        main_window.update_output_tab1(main_window, f"Error retrieving rule value for key '{rule_key}': {e}")
        print(f"Error retrieving rule value for key '{rule_key}': {e}")
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
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem(getattr(main_window, "rules_format", "rules").upper()))
            row_position += 1

        # Analysis rules parameters
        for path, value in main_window.analysis_rules_parameters.items():
            main_window.preview_table.setItem(row_position, 0, QTableWidgetItem(path))
            main_window.preview_table.setItem(row_position, 1, QTableWidgetItem("<variable>"))
            main_window.preview_table.setItem(row_position, 2, QTableWidgetItem(str(value[1])))
            main_window.preview_table.setItem(row_position, 3, QTableWidgetItem(getattr(main_window, "rules_format", "rules").upper()))
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
    if not file_path:
        return

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
                # Add /n line breaks for better readability in the .ini file
                parameters_str = "{ \n"
                for key, value in parameters.items():
                    if isinstance(value, list):
                        parameters_str += f"\t\"{key}\" : {value},\n"
                    else:
                        parameters_str += f"\t\"{key}\" : '{value}',\n"
                config[struc_name]["parameters"] = parameters_str + "}"

            # Add rules if applicable
            if (main_window.fixed_rules_parameters or main_window.analysis_rules_parameters):
                config[struc_name]["rulesFile_ref"] = os.path.relpath(main_window.rule_path, os.getcwd())
                rules_parameters = {key: value for key, value in main_window.fixed_rules_parameters.items()}
                rules_parameters.update({key: value for key, value in main_window.analysis_rules_parameters.items()})
                # Add /n line breaks for better readability in the .ini file
                rules_parameters_str = "{ \n"
                for key, value in rules_parameters.items():
                    if isinstance(value, list):
                        rules_parameters_str += f"\t\"{key}\" : {value},\n"
                    else:
                        rules_parameters_str += f"\t\"{key}\" : '{value}',\n"
                config[struc_name]["parameters_rules"] = rules_parameters_str + "}"

            # Write to the file
            with open(file_path, "w") as ini_file:
                config.write(ini_file)

            main_window.update_output_tab1(main_window, f"Successfully saved .ini file to: {file_path}")
            # Load the .ini file
            main_window.load_ini_file(main_window, filePath=file_path, strucName=struc_name)
        except Exception as e:
            main_window.update_output_tab1(main_window, f"Error saving .ini file: {e}")
