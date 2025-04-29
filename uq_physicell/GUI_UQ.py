import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from uq_physicell import __version__ as uq_physicell_version

from uq_physicell.GUI_UQ_tab1_select_params import create_tab1
from uq_physicell.GUI_UQ_tab2_sensitivity_analysis import create_tab2
from uq_physicell.GUI_UQ_tab3_calibration import create_tab3
from uq_physicell.GUI_UQ_tab4_model_selection import create_tab4
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.parent_map = {}  # Dictionary to store parent-child relationships
        self.analysis_parameters = {}  # Store parameters added to analysis
        self.fixed_parameters = {}  # Store parameters with fixed values
        self.analysis_rules_parameters = {}  # Store rules parameters added to analysis
        self.fixed_rules_parameters = {}  # Store rules parameters with fixed values
        self.xml_file_path = None  # Store the path of the loaded XML file
        self.ini_file_path = None  # Store the path of the loaded .ini file
        self.db_file_path = None  # Store the path of the loaded .db file
        self.csv_data = pd.DataFrame()  # Empty DataFrame for rules
        self.qoi_funcs = {}  # Store the QoI functions
        self.df_qois = pd.DataFrame()  # DataFrame to store the output of the analysis

        self.setWindowTitle("UQ_PhysiCell - GUI")

        # Create tabs
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        # Add tabs to tab widget
        self.tabs.addTab(self.tab1, "Select Parameters")  # Moved to the first tab
        self.tabs.addTab(self.tab2, "Sensitivity Analysis")
        self.tabs.addTab(self.tab3, "Calibration")
        self.tabs.addTab(self.tab4, "Model Selection")

        ##########################################
        # Layout for tab 1 - Select parameters
        ##########################################
        # Create the layout for tab 1
        layout_tab1 = create_tab1(self)
        self.tab1.setLayout(layout_tab1)
        # self.layout_tab1 = layout_tab1  # Store layout for later use

        ##########################################
        # Layout for tab 2 - Sensitivity Analysis
        ##########################################
        layout_tab2 = create_tab2(self)
        self.tab2.setLayout(layout_tab2)

        ##########################################
        # Layout for tab 3 - Calibration
        ##########################################

        # Methods: BO and ABC
        # BO: Bayesian optimization is a powerful black-box optimization framework that efficiently finds the global optimum of an unknown function by iteratively balancing exploration and exploitation using a surrogate model and an acquisition function. 
        # ABC inference approximates Bayesian posterior distributions by comparing simulated data to observed data, avoiding direct likelihood calculations.

        layout_tab3 = QVBoxLayout()
        self.tab3.setLayout(layout_tab3)

        ##########################################
        # Layout for tab 4 - Model Selection
        ##########################################
        layout_tab4 = QVBoxLayout()
        self.tab4.setLayout(layout_tab4)
        
        # Set central widget
        self.setCentralWidget(self.tabs)

def main():
    print(f"UQ_PhysiCell version: {uq_physicell_version}")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 800, 800)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()