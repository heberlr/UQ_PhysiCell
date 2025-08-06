"""
Bayesian Optimization module for UQ PhysiCell.

This module provides Bayesian optimization capabilities for PhysiCell model calibration
with enhanced strategies for handling parameter non-identifiability.
"""

from .optimize import (
    CalibrationContext,
    run_bayesian_optimization,
    diagnose_identification_issues
)

from .database import (
    create_structure,
    insert_metadata,
    insert_param_space,
    insert_qois,
    insert_gp_models,
    insert_samples,
    insert_output,
    load_structure
)

from .distances import (
    SumSquaredDifferences,
    Manhattan,
    Chebyshev,
)

__all__ = [
    'CalibrationContext',
    'run_bayesian_optimization',
    'diagnose_identification_issues',
    'create_structure',
    'insert_metadata',
    'insert_param_space',
    'insert_qois',
    'insert_gp_models',
    'insert_samples',
    'insert_output',
    'load_structure',
    'SumSquaredDifferences',
    'Manhattan',
    'Chebyshev',
]
