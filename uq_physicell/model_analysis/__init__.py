"""
Model Analysis module for UQ PhysiCell.

This module provides sensitivity analysis capabilities, sampling methods, and database
operations for PhysiCell model analysis and uncertainty quantification.
"""

from .ma_context import (
    ModelAnalysisContext,
    run_simulations
)

from .samplers import (
    run_global_sampler,
    run_local_sampler
)

from .sensitivity_analysis import (
    run_global_sa,
    run_local_sa,
    get_global_SA_parameters,
    get_local_SA_parameters,
    get_sa_results
)

from .visualization import (
    plot_qoi_over_time,
    plot_global_sa_results,
    plot_local_sa_results,
    plot_cells_2D,
)

from .utils import (
    mcds_list_to_qoi_df_for_sa,
    mcds_list_to_qoi_df_for_calib,
    mcds_list_to_qoi_df_long,
    get_qoi_from_db_file,
    calculate_qoi_from_db_file,
    get_mean_std_qois,
    get_relative_mcse_qois,
    get_summary_statistics_qois,
    calculate_qoi_statistics
)

from ..database.ma_db import (
    load_structure,
    load_metadata,
    load_parameter_space,
    load_qois,
    load_samples,
    load_output,

)

__all__ = [
    'ModelAnalysisContext',
    'run_simulations',
    'run_global_sampler',
    'run_local_sampler',
    'run_global_sa',
    'run_local_sa',
    'get_global_SA_parameters',
    'get_local_SA_parameters',
    'get_sa_results',
    'plot_qoi_over_time',
    'plot_global_sa_results',
    'plot_local_sa_results',
    'plot_cells_2D',
    'mcds_list_to_qoi_df_for_sa',
    'mcds_list_to_qoi_df_for_calib',
    'mcds_list_to_qoi_df_long',
    'get_qoi_from_db_file',
    'calculate_qoi_from_db_file',
    'get_mean_std_qois',
    'get_relative_mcse_qois',
    'get_summary_statistics_qois',
    'calculate_qoi_statistics',
    'load_structure',
    'load_metadata',
    'load_parameter_space',
    'load_qois',
    'load_samples',
    'load_output',
]
