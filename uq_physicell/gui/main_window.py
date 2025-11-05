import sys, os
import pandas as pd
import queue
import logging

# All the specific classes we need
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtGui import QIcon, QPalette, QColor, QDesktopServices, QTextCursor
from PyQt5.QtCore import Qt, QUrl, QTimer

# My local modules
from uq_physicell import __version__ as uq_physicell_version
from .tab1_select_params import create_tab1
from .tab2_model_analysis import create_tab2
from .tab3_calibration import create_tab3
from .tab4_model_selection import create_tab4
from .load_files import load_xml_file, load_ini_file, load_db_file
    
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
        self.ma_file_path = None  # Store the path of the model analysis .db file
        self.bo_file_path = None  # Store the path of the bayesian optimization .db file
        self.xml_data = pd.DataFrame
        self.csv_data = pd.DataFrame()  # Empty DataFrame for rules
        self.qoi_funcs = {}  # Store the QoI functions
        self.df_output = pd.DataFrame()  # DataFrame to store the output of the analysis
        self.df_qois = pd.DataFrame()  # DataFrame to store the QoIs of the output of the analysis
        
        self.df_output = pd.DataFrame()  # DataFrame to store the output of the analysis
        self.df_qois = pd.DataFrame()  # DataFrame to store the QoIs of the output of the analysis

        # Logging related attributes - created later after widgets are set up
        self.logger_tab2 = None
        self.logger_tab3 = None
        
        # Thread-safe message queue
        self.message_queue = queue.Queue()
        
        # Set up a timer for processing messages
        self.message_timer = QTimer()
        self.message_timer.setInterval(50)  # Process messages every 50ms
        self.message_timer.timeout.connect(self._process_message_queue)
        
        # Dictionary to store output widgets for different tabs
        self.output_widgets = {}
        
        # Setting up the main window
        self.setWindowTitle("UQ_PhysiCell - GUI")

        # Create a menu bar
        self.menu_bar = self.menuBar()
        self.menu_bar.setStyleSheet("color: black")
        self.menu_bar.setNativeMenuBar(False)  # Set the menu bar to be native

        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet('''
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab:selected {background: orange;}
        ''')
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        # Add tabs to tab widget
        self.tabs.addTab(self.tab1, "Select Parameters")  # Moved to the first tab
        self.tabs.addTab(self.tab2, "Model Analysis")
        self.tabs.addTab(self.tab3, "Calibration")
        self.tabs.addTab(self.tab4, "Model Selection")

        ##########################################
        # Menu bar
        ##########################################
        # UQ_PhysiCell menu bar
        uq_physicell_menu = self.menu_bar.addMenu("&UQ_PhysiCell")
        uq_physicell_menu.addAction("About", lambda: show_about_uq_physicell(self))
        # uq_physicell_menu.addAction("Reset", lambda: reset_all(self))
        uq_physicell_menu.addAction("Exit", self.close)
        # File menu bar
        file_menu = self.menu_bar.addMenu("&File")
        file_menu.addAction("Load XML File", lambda: load_xml_file(self))
        file_menu.addAction("Load INI File", lambda: load_ini_file(self))
        file_menu.addAction("Load DB File", lambda: load_db_file(self))
        # Help menu bar
        help_menu = self.menu_bar.addMenu("&Help")
        help_menu.addAction("User Guide (link)", lambda: show_user_guide(self))
        help_menu.addAction("Documentation (link)", lambda: show_documentation(self))
        help_menu.addAction("Create Issue (link)", lambda: show_create_issue(self))

        ##########################################
        # Layout for tab 1 - Select parameters
        ##########################################
        # Create the layout for tab 1
        scroll_area_tab1 = create_tab1(self)
        layout_tab1 = QVBoxLayout()
        layout_tab1.addWidget(scroll_area_tab1)
        self.tab1.setLayout(layout_tab1)
        # self.layout_tab1 = layout_tab1  # Store layout for later use

        ##########################################
        # Layout for tab 2 - Model Analysis
        ##########################################
        layout_tab2 = create_tab2(self)
        self.tab2.setLayout(layout_tab2)

        ##########################################
        # Layout for tab 3 - Calibration
        ##########################################

        # Methods: BO and ABC
        # BO: Bayesian optimization is a powerful black-box optimization framework that efficiently finds the global optimum of an unknown function by iteratively balancing exploration and exploitation using a surrogate model and an acquisition function. 
        # ABC inference approximates Bayesian posterior distributions by comparing simulated data to observed data, avoiding direct likelihood calculations.
        layout_tab3 = create_tab3(self)
        self.tab3.setLayout(layout_tab3)

        ##########################################
        # Layout for tab 4 - Model Selection
        ##########################################
        layout_tab4 =  create_tab4(self)
        self.tab4.setLayout(layout_tab4)
        
        # Set central widget
        self.setCentralWidget(self.tabs)
        
        # Start the message processing timer after initializing all UI components
        self.start_message_processing()
        
    def _process_message_queue(self):
        """Process messages from the queue and update the appropriate widget safely in the main thread."""
        try:
            # Process up to 10 messages per timer tick to prevent UI freezing
            for _ in range(10):
                if self.message_queue.empty():
                    break
                
                try:    
                    tab_id, message = self.message_queue.get_nowait()
                    
                    # Update the appropriate widget if it exists
                    if tab_id in self.output_widgets and self.output_widgets[tab_id]:
                        try:
                            # Get the text edit widget and append message
                            text_edit = self.output_widgets[tab_id]
                            text_edit.append(message)
                            
                            # Ensure text is visible by scrolling to the bottom
                            # This is safe because we're in the main thread
                            text_edit.ensureCursorVisible()
                            
                        except Exception as e:
                            print(f"Error updating UI: {e}", file=sys.stderr)
                    
                    # Mark task as done even if there was an error
                    self.message_queue.task_done()
                    
                except queue.Empty:
                    # Queue became empty while we were processing
                    break
                    
        except Exception as e:
            print(f"Error processing message queue: {e}", file=sys.stderr)
    
    def start_message_processing(self):
        """Start the message processing timer."""
        if not self.message_timer.isActive():
            self.message_timer.start()
            
    def stop_message_processing(self):
        """Stop the message processing timer."""
        if self.message_timer.isActive():
            self.message_timer.stop()
    
    def add_output_widget(self, tab_id, widget):
        """Register an output widget for a specific tab."""
        self.output_widgets[tab_id] = widget
        # Make sure the message timer is running
        if hasattr(self, 'message_timer') and not self.message_timer.isActive():
            self.message_timer.start()
        
    def post_message(self, tab_id, message):
        """Add a message to the queue to be processed by the main thread."""
        try:
            self.message_queue.put((tab_id, message))
            # Make sure the timer is running
            if hasattr(self, 'message_timer') and not self.message_timer.isActive():
                self.message_timer.start()
        except Exception as e:
            print(f"Error posting message: {e}", file=sys.stderr)
            print(f"Message: {message}", file=sys.stderr)
        
    def closeEvent(self, event):
        """Handle the window close event to properly clean up resources."""
        # Stop message processing timer
        self.stop_message_processing()
        
        # Clear message queue
        try:
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
                self.message_queue.task_done()
        except (AttributeError, queue.Empty):
            pass
            
        # Clear output widgets dictionary
        if hasattr(self, 'output_widgets'):
            self.output_widgets.clear()
        
        # Clean up tab2 logger if it exists
        if hasattr(self, 'logger_tab2') and self.logger_tab2:
            # Remove all handlers to prevent them from accessing deleted Qt objects
            if self.logger_tab2.handlers:
                for handler in self.logger_tab2.handlers[:]:
                    try:
                        handler.close()
                    except:
                        pass
                    self.logger_tab2.removeHandler(handler)
        
        # Clean up tab3 logger if it exists
        if hasattr(self, 'logger_tab3') and self.logger_tab3:
            # Remove all handlers to prevent them from accessing deleted Qt objects
            if self.logger_tab3.handlers:
                for handler in self.logger_tab3.handlers[:]:
                    try:
                        handler.close()
                    except:
                        pass
                    self.logger_tab3.removeHandler(handler)
        
        # Reset root logger configuration to avoid issues on next startup
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Accept the close event
        event.accept()

def show_about_uq_physicell(main_window):
    about_text = f"<b>UQ_PhysiCell v.{uq_physicell_version}</b><br><br>"
    about_text += "UQ_PhysiCell is a tool for uncertainty quantification in PhysiCell models.<br><br>"
    about_text += "For more information, visit the <a href='https://github.com/heberlr/UQ_PhysiCell'>UQ_PhysiCell GitHub repository</a>."
    
    about_box = QMessageBox(main_window)
    about_box.setIcon(QMessageBox.Information)
    about_box.setWindowTitle("About UQ_PhysiCell")
    about_box.setTextFormat(Qt.RichText)
    about_box.setText(about_text)
    about_box.setStandardButtons(QMessageBox.Ok)
    about_box.exec_()

def show_user_guide(main_window):
    user_guide_url = QUrl('https://github.com/heberlr/UQ_PhysiCell')
    if not QDesktopServices.openUrl(user_guide_url):
        QMessageBox.warning(main_window, 'Open Url', 'Could not open URL')

def show_documentation(main_window):
    documentation_url = QUrl('https://uq-physicell.readthedocs.io/en/latest/')
    if not QDesktopServices.openUrl(documentation_url):
        QMessageBox.warning(main_window, 'Open Url', 'Could not open URL')

def show_create_issue(main_window):
    create_issue_url = QUrl('https://github.com/heberlr/UQ_PhysiCell/issues')
    if not QDesktopServices.openUrl(create_issue_url):
        QMessageBox.warning(main_window, 'Open Url', 'Could not open URL')

def reset_all(main_window):
    print("Resetting all parameters and configurations... (this is a placeholder function)")

def main():
    print(f"Opening GUI from UQ_PhysiCell version: {uq_physicell_version}")
    
    # Create the application
    uq_physicell_app = QApplication(sys.argv)

    # Set the application icon
    icon_path = os.path.join(os.path.dirname(__file__), '../doc/icon.png')
    if os.path.exists(icon_path):
        uq_physicell_app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Warning: Icon file not found at {icon_path}")
    
    # Color pattern equal to PhysiCell-Studio
    palette = QPalette()
    rgb = 236
    palette.setColor(QPalette.Window, QColor(rgb, rgb, rgb))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    rgb = 100
    palette.setColor(QPalette.Base, QColor(rgb, rgb, rgb))
    palette.setColor(QPalette.AlternateBase, QColor(rgb, rgb, rgb))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Base, Qt.white)
    palette.setColor(QPalette.Text, Qt.black)
    palette.setColor(QPalette.Button, QColor(255, 255, 255))  # white: affects tree widget header and table headers
    palette.setColor(QPalette.ButtonText, Qt.black)  # e.g., header for tree widget too?!
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(236, 236, 236))   # background when user tabs through QLineEdits
    palette.setColor(QPalette.Highlight, QColor(210, 210, 210))   # background when user tabs through QLineEdits
    palette.setColor(QPalette.HighlightedText, Qt.black)
    uq_physicell_app.setPalette(palette)

    window = MainWindow()
    window.resize(1100, 790)
    window.setMinimumSize(1100, 790)
    window.show()
    
    sys.exit(uq_physicell_app.exec_())

if __name__ == '__main__':
    main()