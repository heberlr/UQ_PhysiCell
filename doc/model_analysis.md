# Model Analysis

This section covers the analysis tools available in UQ-PhysiCell for understanding model behavior, parameter importance, and convergence properties.

## Overview

Model analysis is essential for:
- Understanding parameter sensitivity and importance
- Identifying potential issues with parameter identifiability
- Ensuring model convergence and reliability

## Sensitivity Analysis

Sensitivity analysis helps identify which parameters have the most significant impact on model outputs. UQ-PhysiCell provides both global and local sensitivity analysis methods.

### Global Sensitivity Analysis (GSA)

Global sensitivity analysis examines parameter effects across the entire parameter space on an quantity of interest (QoI). UQ Physicell supports multiple methods of GSA, but standard the [SALib python library](https://salib.readthedocs.io/en/latest/index.html) is used here.

Example:
```python
oat_analysis = ModelAnalysisContext(db_file_name = 'LHS_database.db', 
model_config = {'ini_path': <.ini file path>, 'struc_name': <key of model structure in .ini file>}
sampler = 'LHS' # Latin Hypercube Sampling
params_info = {'parA': {'ref_value': 1.0, 'lower_bound': 0.5, 'upper_bound': 1.0}, 'parB': {'ref_value': 2.0, 'lower_bound': 0.0, 'upper_bound': 3.0}}, 
qoi_str, 
num_workers=10)

oat_results = oat_analysis.run_oat_analysis()
oat_analysis.plot_oat_effects(save_path="oat_effects.png")
```

### Local Sensitivity Analysis (LSA)

Local sensitivity analysis examines parameter effects around specific points, we set as the standard LSA the One-at-a-Time (OAT) Analysis, where we change parameters individually according a perturbation, while keeping others fixed:

Example:

```python
oat_analysis = LocalSensitivity(
    model_config=model_config,
    qoi_functions=qoi_functions,
    base_parameters=base_parameters,
    method="oat",
    perturbation_range=0.2  # ±20% variation
)

oat_results = oat_analysis.run_oat_analysis()
oat_analysis.plot_oat_effects(save_path="oat_effects.png")
```