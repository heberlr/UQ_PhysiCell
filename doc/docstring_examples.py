# Example of NumPy-style docstring format for comparison

def example_numpy_style_function(param1: dict, param2: str, param3: list) -> tuple:
    """Run sensitivity analysis using NumPy-style docstring.
    
    This is a longer description that explains what the function does in more detail.
    It can span multiple lines and provide context about the algorithm or method.
    
    Parameters
    ----------
    param1 : dict
        Dictionary containing parameter names and their properties.
        Must include a 'samples' key with parameter sample dictionaries.
    param2 : str
        Name of the sensitivity analysis method to use.
    param3 : list
        List of all time labels present in the data.
    
    Returns
    -------
    tuple
        A tuple containing two elements:
        
        - results : dict
            Dictionary with sensitivity analysis results
        - metadata : dict
            Additional metadata about the analysis
    
    Raises
    ------
    ValueError
        If there's a mismatch between parameters and data.
    ImportError
        If required analysis libraries are not available.
    
    See Also
    --------
    run_global_sa : Global sensitivity analysis function
    OAT_analyze : One-at-a-time analysis implementation
    
    Notes
    -----
    This function implements the methodology described in [1]_.
    
    References
    ----------
    .. [1] Saltelli, A., et al. "Global Sensitivity Analysis: The Primer" (2008)
    
    Examples
    --------
    >>> params = {'param1': {'bounds': [0, 1]}, 'samples': {...}}
    >>> results, meta = example_function(params, 'sobol', ['t0', 't1'])
    >>> print(results.keys())
    """
    pass
