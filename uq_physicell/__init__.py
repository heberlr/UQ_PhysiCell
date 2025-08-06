"""
UQ-PhysiCell: Uncertainty Quantification for PhysiCell Models

This package provides tools for uncertainty quantification, sensitivity analysis,
and Bayesian optimization of PhysiCell models.
"""

from uq_physicell.VERSION import __version__
from uq_physicell.uq_physicell import (
    PhysiCell_Model, 
    check_parameters_input, 
    setup_model_input,
    RunModel, 
    get_xml_element_value, 
    set_xml_element_value, 
    generate_xml_file, 
    get_rules, 
    get_rule_index_in_csv, 
    generate_csv_file
)

# Import submodules to make them available
try:
    from . import bo
except ImportError:
    bo = None

try:
    from . import model_analysis
except ImportError:
    model_analysis = None

try:
    from . import utils
except ImportError:
    utils = None

try:
    from . import database
except ImportError:
    database = None

try:
    from . import gui
except ImportError:
    gui = None

__all__ = [
    '__version__',
    'PhysiCell_Model',
    'check_parameters_input',
    'setup_model_input', 
    'RunModel',
    'get_xml_element_value',
    'set_xml_element_value',
    'generate_xml_file',
    'get_rules',
    'get_rule_index_in_csv',
    'generate_csv_file',
    'bo',
    'model_analysis',
    'utils',
    'database',
    'gui'
]