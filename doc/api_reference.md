# API Reference

## Core UQ-PhysiCell Module

The main module provides the core PhysiCell model interface and utilities.

```{eval-rst}
.. automodule:: uq_physicell
   :members:
   :undoc-members:
   :show-inheritance:
   :no-imported-members:
```

## Model Analysis Module

Tools for sensitivity analysis, parameter sampling, and model analysis.

```{eval-rst}
.. automodule:: uq_physicell.model_analysis.ma_context
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
.. automodule:: uq_physicell.bo.bo_context
   :members:
   :undoc-members:
   :show-inheritance:
   :no-imported-members:
```

```{note}
The Bayesian Optimization module requires additional dependencies (botorch, gpytorch, torch).
Install them with: `pip install botorch`
```
### Plotting and Visualization

```{eval-rst}
.. automodule:: uq_physicell.bo.plots
   :members:
   :undoc-members:
```

## Utils

### Distances
```{eval-rst}
.. automodule:: uq_physicell.utils.distances
   :members:
   :undoc-members:
```

