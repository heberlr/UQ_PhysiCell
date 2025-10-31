"""
Bayesian Optimization module for UQ PhysiCell.

This module provides Bayesian optimization capabilities for PhysiCell model calibration
with enhanced strategies for handling parameter non-identifiability.
"""

from .bo_context import (
    CalibrationContext,
    run_bayesian_optimization,
)

from .plots import (
    plot_parameter_space,
    plot_parameter_space_db,
    plot_qoi_param,
    plot_qoi_param_db,
    plot_parameter_vs_fitness,
    plot_parameter_vs_fitness_db,
)

from .utils import (
    normalize_params_df,
    extract_best_parameters,
    extract_best_parameters_db,
    extract_all_pareto_points,
    analyze_pareto_results,
    get_observed_qoi,
)   

__all__ = [
    'CalibrationContext',
    'run_bayesian_optimization',
    'plot_parameter_space',
    'plot_parameter_space_db',
    'plot_qoi_param',
    'plot_qoi_param_db',
    'plot_parameter_vs_fitness',
    'plot_parameter_vs_fitness_db',
    'normalize_params_df',
    'extract_best_parameters',
    'extract_best_parameters_db',
    'extract_all_pareto_points',
    'analyze_pareto_results',
    'get_observed_qoi',
]
