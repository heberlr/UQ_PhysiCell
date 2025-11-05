"""
UQ-PhysiCell: Uncertainty Quantification for PhysiCell Models

This package provides tools for uncertainty quantification, sensitivity analysis,
and Bayesian optimization of PhysiCell models.
"""

from uq_physicell.VERSION import __version__
from uq_physicell.pc_model import (
    PhysiCell_Model,
    _run_model as RunModel, # backward compatibility
    get_physicell,
    compile_physicell,
)

# Import submodules to make them available
try:
    from . import bo
except ImportError:
    bo = None

try:
    from . import abc
except ImportError:
    abc = None

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
    'bo',
    'abc',
    'model_analysis',
    'utils',
    'database',
    'gui'
]