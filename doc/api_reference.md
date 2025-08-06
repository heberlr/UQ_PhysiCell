# API Reference

## Core UQ-PhysiCell Module

The main module provides the core PhysiCell model interface and utilities.

```{eval-rst}
.. automodule:: uq_physicell
   :members:
   :undoc-members:
   :show-inheritance:
```

### Quick Import Examples

```python
# Core model functionality
from uq_physicell import PhysiCell_Model, RunModel

# Model analysis tools
from uq_physicell.model_analysis import (
    ModelAnalysisContext,
    run_simulations,
    run_global_sa,
    run_local_sa
)

# Bayesian optimization tools  
from uq_physicell.bo import (
    CalibrationContext,
    run_bayesian_optimization,
    SumSquaredDifferences,
    Manhattan,
    Chebyshev
)
```

## Model Analysis Module

Tools for sensitivity analysis, parameter sampling, and model analysis.

```{eval-rst}
.. automodule:: uq_physicell.model_analysis.main
   :members:
   :undoc-members:
   :show-inheritance:
   :no-imported-members:
```

### Sampling Methods

```{eval-rst}
.. automodule:: uq_physicell.model_analysis.samplers
   :members:
   :undoc-members:
```

### Sensitivity Analysis

```{eval-rst}
.. automodule:: uq_physicell.model_analysis.sensitivity_analysis
   :members:
   :undoc-members:
```

## Bayesian Optimization Module

Multi-objective Bayesian optimization for model calibration.

```{eval-rst}
.. automodule:: uq_physicell.bo.optimize
   :members:
   :undoc-members:
   :show-inheritance:
   :no-imported-members:
```

```{note}
The Bayesian Optimization module requires additional dependencies (botorch, gpytorch, torch).
Install them with: `pip install botorch matplotlib plotly seaborn`
```

### Distance Functions

```{eval-rst}
.. automodule:: uq_physicell.bo.distances
   :members:
   :undoc-members:
```

### Plotting and Visualization

```{eval-rst}
.. automodule:: uq_physicell.bo.plot
   :members:
   :undoc-members:
```