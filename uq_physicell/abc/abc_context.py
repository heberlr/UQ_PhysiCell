import os
import logging
import concurrent.futures
from typing import Union, Optional

import numpy as np
import pandas as pd

class ABC_Context:
    """
    Context for Approximate Bayesian Computation (ABC) using pyabc

    This class encapsulates the necessary components and methods for performing ABC
    calibration, including prior distributions, transition functions, and distance metrics.

    Attributes:
        db_path (str): Path to the database file.
        obsData (str or dict): Path or dictionary containing observed data.
        obsData_columns (dict): Dictionary mapping QoIs to their corresponding columns in the observed data.
        model_config (dict): Configuration dictionary for the PhysiCell model, including paths and structure names.
        qoi_functions (dict): Dictionary of functions to compute quantities of interest (QoIs) from model outputs.
        distance_functions (dict): Dictionary of functions to compute distances between model outputs and observed data.
        search_space (dict): Dictionary defining the search space for parameters, including bounds and types.
        abc_options (dict): Options for ABC including max_populations, min_population_size, max_population_size, max_simulations.
        logger (logging.Logger): Logger instance for logging messages during the calibration process.
    """
    def __init__(
        self,
        db_path: str, 
        obsData: Union[str, dict], 
        obsData_columns: dict, 
        model_config: dict, 
        qoi_functions: dict, 
        distance_functions: dict, 
        search_space: dict, 
        abc_options: dict, 
        logger: logging.Logger
    ):
        # Initialize ABC_Context
        self.db_path = db_path
        self.model_config = model_config
        self.qoi_functions = qoi_functions
        self.distance_functions = distance_functions
        self.search_space = search_space
        self.abc_options = abc_options
        self.logger = logger

        # Load and validate observed data
        if isinstance(obsData, dict):
            self.dic_obsData = obsData
            self.obsData_path = None
        else:  # obsData is a path
            try:
                self.obsData_path = obsData
                self.dic_obsData = pd.read_csv(obsData).to_dict('list')
                # Replace column names according to obsData_columns mapping
                for qoi, column_name in obsData_columns.items():
                    if column_name in self.dic_obsData:
                        self.dic_obsData[qoi] = np.array(self.dic_obsData.pop(column_name), dtype=np.float64)
                    else:
                        raise ValueError(f"Column '{column_name}' not found in observed data.")
                self.logger.debug(f"Successfully loaded observed data from {obsData}")
            except Exception as e:
                self.logger.error(f"Error reading observed data from {obsData}: {e}")
                raise

        # Optional model customization
        self.fixed_params = abc_options.get('fixed_params', {})
        self.summary_function = abc_options.get("summary_function", None)
        self.custom_run_single_replicate_func = abc_options.get("custom_run_single_replicate_func", None)
        self.custom_aggregation_func = abc_options.get("custom_aggregation_func", None)
