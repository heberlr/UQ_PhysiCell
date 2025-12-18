import numpy as np
import pandas as pd

from SALib.analyze import fast as fast_analyze, rbd_fast as rbd_fast_analyze, ff as ff_analyze, pawn as pawn_analyze, dgsm as dgsm_analyze, enhanced_hdmr as hdmr_analyze, rsa as rsa_analyze, discrepancy as discrepancy_analyze, delta as delta_analyze, sobol as sobol_analyze

# Compatibility of samplers with methods following the SALib library
# https://salib.readthedocs.io/en/latest/index.html
# https://salib.readthedocs.io/en/latest/user-guide/analysis.html
# https://salib.readthedocs.io/en/latest/user-guide/sampling.html

# Issues with methods:
# Discrepancy Sensitivity Indices: Error Bounds are not consistent 'l_bounds' < 'u_bounds'
# Delta Moment-Independent Measure: Error module 'numpy' has no attribute 'trapezoid' (outdated numpy version)
samplers_to_method = {
    "OAT": [
        "OAT - One-At-A-Time",
    ],
    "Fast": [
        "FAST - Fourier Amplitude Sensitivity Test",
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Delta Moment-Independent Measure",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ],
    "Fractional Factorial": [
        "Fractional Factorial",
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Delta Moment-Independent Measure",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ],
    "Finite Difference": [
        "Derivative-based Global Sensitivity Measure (DGSM)",
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Delta Moment-Independent Measure",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ],
    "Latin hypercube sampling (LHS)": [
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Delta Moment-Independent Measure",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ],
    "Sobol": [
        "Sobol Sensitivity Analysis",
        "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test",
        "Delta Moment-Independent Measure",
        "PAWN Sensitivity Analysis",
        "High-Dimensional Model Representation",
        "Regional Sensitivity Analysis",
        "Discrepancy Sensitivity Indices"
    ]
}

def _set_time_labels(all_times_label: list, df_qois: pd.DataFrame) -> dict:
    """Set and validate time labels for QoI values.
    
    Args:
        all_times_label (list): List of all time labels to be processed.
        df_qois (pd.DataFrame): DataFrame containing the QoI values with time columns.
    
    Returns:
        dict: Dictionary with time labels as keys and their corresponding time values
            as values, sorted by time values.
    
    Raises:
        ValueError: If more than one unique value is found for any time label.
    """
    # Check if qoi_time_values is empty
    qoi_time_values = {}
    
    # Ensure all times are present in the qoi_time_values
    for time_label in all_times_label:
        unique_values = df_qois[time_label].unique()
        if len(unique_values) == 1:
            qoi_time_values[time_label] = unique_values[0]
        else:
            raise ValueError(f"Error: More than one unique value for time label '{time_label}': {unique_values}.")

    # Sort the time values
    qoi_time_values = dict(sorted(qoi_time_values.items(), key=lambda item: item[1]))
    return qoi_time_values

def _get_SA_problem(params_dict: dict) -> dict:
    """Create a SALib problem dictionary from parameter definitions.
    
    This function converts parameter definitions to the format required by
    the SALib (Sensitivity Analysis Library) for conducting sensitivity analysis.

    Args:
        params_dict (dict): Dictionary containing parameter names as keys and
            parameter properties as values. Each parameter should have
            'lower_bound' and 'upper_bound' keys. The special key 'samples'
            is excluded from the problem definition.

    Returns:
        dict: SALib-formatted problem dictionary containing:
            - num_vars (int): Number of parameters
            - names (list): List of parameter names
            - bounds (list): List of tuples with (lower, upper) bounds for each parameter
    """
    param_names = [key for key in params_dict.keys() if key != "samples"]
    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [(params_dict[key]['lower_bound'], params_dict[key]['upper_bound']) for key in param_names]
    }
    return problem

def run_global_sa(params_dict: dict, method: str, all_times_label: list, all_qois_names: list, df_qois: pd.DataFrame, qoi_time_values: dict = None) -> tuple:
    """Run global sensitivity analysis using the specified method.
    
    Args:
        params_dict (dict): Dictionary containing parameter names, properties, and sample values.
            Must include a 'samples' key with parameter sample dictionaries.
        method (str): Name of the sensitivity analysis method to use. Supported methods
            include 'FAST - Fourier Amplitude Sensitivity Test', 'Sobol Sensitivity Analysis',
            'PAWN Sensitivity Analysis', etc.
        all_times_label (list): List of all time labels present in the QoI data.
        all_qois_names (list): List of all Quantity of Interest names to analyze.
        df_qois (pd.DataFrame): DataFrame containing QoI values with columns formatted as
            '{qoi_name}_{time_index}'.
    
    Returns:
        tuple: A tuple containing:
            - sa_results_dict (dict): Nested dictionary with sensitivity analysis results.
              Structure: {qoi_name: {time_label: analysis_results}}
            - qoi_time_values (dict): Dictionary mapping time labels to their values,
              sorted by time.
    
    Raises:
        ValueError: If there's a mismatch between number of samples and QoI results,
            or if the specified method fails during analysis.
    """
    # Get the problem definition for SALib
    problem = _get_SA_problem(params_dict)
    # Generate params_np - it is sorted by sample ID
    params_np = np.array([[param_sample_dic[param] for param in problem['names']] for param_sample_dic in params_dict["samples"].values()])
    # Set the times labels sorted by time values
    if qoi_time_values is None: 
        qoi_time_values = _set_time_labels(all_times_label, df_qois)
    # SA results dictionary
    sa_results_dict = { qoi: {} for qoi in all_qois_names }
    for qoi in all_qois_names: 
        for time_id, time_label in enumerate(qoi_time_values.keys()):
            # Generate qoi_result_np - it is sorted by sample ID
            if qoi_time_values is None: qoi_result_np = df_qois[f"{qoi}_{time_label}"].to_numpy()
            else: qoi_result_np = df_qois[f"{qoi}_{time_id}"].to_numpy()
            if len(qoi_result_np) != len(params_np):
                raise ValueError(f"Error: Mismatch between number of samples ({len(params_np)}) and QoI results ({len(qoi_result_np)})!")
            print(f"Running {method} for QoI: {qoi} and time: {qoi_time_values[time_label]}") 

            try:
                # Run the analysis based on the selected method
                if method == "FAST - Fourier Amplitude Sensitivity Test":
                    sa_results_dict[qoi][time_label] = fast_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "RBD-FAST - Random Balance Designs Fourier Amplitude Sensitivity Test":
                    sa_results_dict[qoi][time_label] = rbd_fast_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Fractional Factorial":
                    sa_results_dict[qoi][time_label] = ff_analyze.analyze(problem, params_np, qoi_result_np, second_order=True)
                elif method == "PAWN Sensitivity Analysis":
                    sa_results_dict[qoi][time_label] = pawn_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Derivative-based Global Sensitivity Measure (DGSM)":
                    sa_results_dict[qoi][time_label] = dgsm_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "High-Dimensional Model Representation":
                    sa_results_dict[qoi][time_label] = hdmr_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Regional Sensitivity Analysis":
                    sa_results_dict[qoi][time_label] = rsa_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Discrepancy Sensitivity Indices":
                    sa_results_dict[qoi][time_label] = discrepancy_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Delta Moment-Independent Measure":
                    sa_results_dict[qoi][time_label] = delta_analyze.analyze(problem, params_np, qoi_result_np)
                elif method == "Sobol Sensitivity Analysis":
                    sa_results_dict[qoi][time_label] = sobol_analyze.analyze(problem, qoi_result_np)
            except Exception as e:
                raise ValueError(f"Error running {method} for QoI: {qoi} and time: {qoi_time_values[time_label]} - {e}")

    return sa_results_dict, qoi_time_values

def _OAT_analyze(dic_samples: dict, dic_qoi: dict) -> dict:
    """Perform One-At-a-Time (OAT) analysis on the simulation results.
    
    Args:
        dic_samples (dict): Dictionary of parameter sample dictionaries, where each key
            is a sample ID and each value is a dictionary of parameter names and values.
        dic_qoi (dict): Dictionary of Quantities of Interest (QoI) values, where each
            key is a sample ID and each value is the corresponding QoI result.
    
    Returns:
        dict: Dictionary containing sensitivity indices for each parameter. Keys are 
            parameter names and values are arrays of sensitivity indices for each
            perturbation.
    
    Note:
        Sample 0 is treated as the reference sample. All other samples are compared
        against this reference to compute sensitivity indices.
    """
    # Remove unused variables ref_pars and qoi_ref
    # Extract parameter samples and QoI samples
    par_samples = np.array([list(sample.values()) for sample in dic_samples.values()])
    # Normalize the parameter samples
    par_samples = (par_samples - par_samples.min(axis=0)) / (par_samples.max(axis=0) - par_samples.min(axis=0))
    # Sample 0 is the reference sample
    ref_pars = par_samples[0]; par_samples = par_samples[1:]
    qoi_ref = dic_qoi[0]
    # Extract QoI samples, excluding the reference sample (SampleID different of 0)
    qoi_samples = np.array([qoi for sample_id, qoi in dic_qoi.items() if sample_id != 0])
    # Initialize the results dictionary
    dic_results = {}
    for id, par in enumerate(dic_samples[0].keys()):
        # Calculate the mean and std deviation of the QoIs for each parameter
        par_var = np.abs(par_samples[:, id] - ref_pars[id])
        non_zero_indices = np.where(par_var != 0)[0]
        dic_results[par] = np.abs(qoi_samples[non_zero_indices] - qoi_ref) / par_var[non_zero_indices]  # Compute SI without skipping

    return dic_results

def run_local_sa(params_dict: dict, all_times_label: list, all_qois_names: list, df_qois: pd.DataFrame, method: str = "OAT") -> tuple:
    """Run local sensitivity analysis using the One-At-a-Time (OAT) method.
    
    Args:
        params_dict (dict): Dictionary containing parameter names, properties, and sample values.
            Must include a 'samples' key with parameter sample dictionaries.
        all_times_label (list): List of all time labels present in the QoI data.
        all_qois_names (list): List of all Quantity of Interest names to analyze.
        df_qois (pd.DataFrame): DataFrame containing QoI values with columns formatted as
            '{qoi_name}_{time_index}'.
        method (str, optional): Local sensitivity analysis method. Currently only 'OAT'
            (One-At-a-Time) is supported. Defaults to "OAT".
    
    Returns:
        tuple: A tuple containing:
            - sa_results_dict (dict): Nested dictionary with sensitivity analysis results.
              Structure: {qoi_name: {time_label: {param_name: sensitivity_index}}}
            - qoi_time_values (dict): Dictionary mapping time labels to their values,
              sorted by time.
    
    Note:
        The OAT method computes sensitivity indices by comparing parameter perturbations
        against a reference sample (sample 0). Results are summed across all perturbations
        for each parameter.
    """
    # Get parameter names
    param_names = [key for key in params_dict.keys() if key != "samples"]
    # Set the times labels sorted by time values
    qoi_time_values = _set_time_labels(all_times_label, df_qois)
    # SA results dictionary
    sa_results_dict = { qoi: {} for qoi in all_qois_names }
    for qoi in all_qois_names:
        for id_time, time_label in enumerate(qoi_time_values.keys()):
            qoi_result_dict = df_qois[f"{qoi}_{id_time}"].to_dict()
            print(f"Running {method} for QoI: {qoi} and time: {qoi_time_values[time_label]}")
            # Return a dictionary of sensitivity indices for each perturbation
            sa_results_dict[qoi][time_label] = _OAT_analyze(params_dict["samples"], qoi_result_dict)
            # Overwrite the results for perturbations by summing them up
            for key in sa_results_dict[qoi][time_label]: sa_results_dict[qoi][time_label][key] = np.sum(sa_results_dict[qoi][time_label][key])

    return sa_results_dict, qoi_time_values