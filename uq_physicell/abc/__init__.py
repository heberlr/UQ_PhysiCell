"""
Approximate Bayesian Computation (ABC) for UQ PhysiCell.

This module provides Approximate Bayesian Computation (ABC) for PhysiCell model calibration
with enhanced strategies for model selection using pyABC.
"""

from .abc_context import (
    CalibrationContext,
    ModelSpec,
    run_abc_calibration,
)
from .utils import patch_pyabc_dataframe_csv_fallback, patch_pyabc_nan_particle_weight

# Applied on import so every user of CalibrationContext/run_abc_calibration gets
# them for free -- see each patch function's docstring.
patch_pyabc_dataframe_csv_fallback()
patch_pyabc_nan_particle_weight()

__all__ = [
    'CalibrationContext',
    'ModelSpec',
    'run_abc_calibration',
]