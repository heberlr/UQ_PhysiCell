# UQ-PhysiCell Documentation

Welcome to the UQ-PhysiCell documentation! This project provides uncertainty quantification tools for PhysiCell models.

**Release:** [`v1.2.0`](https://github.com/your-org/UQ_PhysiCell/releases/tag/v1.2.0)

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
model_analysis
bayesian_optimization
api_reference
examples
```

## About UQ-PhysiCell

UQ-PhysiCell is a comprehensive framework for performing uncertainty quantification and parameter calibration of PhysiCell models. It provides sophisticated tools for:

- Parameter sensitivity analysis  
- Model convergence analysis
- Multi-objective Bayesian optimization
- Model calibration with experimental data
- Handling of parameter non-identifiability issues

## Quick Start

```python
from uq_physicell.bo import CalibrationContext, run_bayesian_optimization

# Set up calibration context
calib_context = CalibrationContext(
    db_path="calibration.db",
    obsData="experimental_data.csv",
    # ... additional configuration
)

# Run optimization
run_bayesian_optimization(calib_context)
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
