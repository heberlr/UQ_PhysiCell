# Model Calibration

This section covers the methods of calibration available in UQ-PhysiCell: {ref}`Bayesian Optimization <bayesian-optimization>` and {ref}`Approximate Bayesian Computation (ABC) <approximate-bayesian-computation-abc>`. Note that this package allow another forms of calibration via API, including optimization algorithms like least-squares, Nelder-Mead, genetic algorithm, MCMC, amoung others.


(bayesian-optimization)=
## Bayesian Optimization

This module provides a comprehensive Bayesian optimization framework for calibrating PhysiCell models using multi-objective optimization to learn the Pareto front (more details in [BoTorch](https://botorch.org/docs/multi_objective/)). The framework is designed to efficiently find optimal parameter configurations that minimize the discrepancy between model predictions and observed experimental data.

The optimization problem is formulated as a multi-objective maximization:

$$
\max_{\theta \in \Theta} F(\theta) = [f_1(\theta),\quad f_2(\theta),\quad ..., \quad f_k(\theta)]
$$

Where:
- $\Theta$ is the feasible parameter space defined by bounds
- $\theta$ is the parameter vector to be optimized
- $f_i(\theta)$ is the fitness value for the i-th quantity of interest (QoI), measuring agreement between model predictions and observed data
- The goal is to find the Pareto-optimal set of parameters that maximizes agreement across all QoIs

### Parameter Space ($\Theta$)
The search space is defined by parameter bounds and types:
- **Real parameters**: Continuous variables with lower and upper bounds
- **Integer parameters**: Discrete variables with specified ranges
- **Categorical parameters**: Discrete choices from predefined sets

### Quantities of Interest (QoIs)
QoIs are model outputs that correspond to experimental observables:
- **Time series data**: Cell counts, concentrations, spatial metrics over time
- **Aggregate metrics**: Final values, peak values, areas under curves
- **Derived quantities**: Ratios, differences, or complex functions of raw outputs

### Distance Metrics
The discrepancy between model predictions and observed data is quantified using distance metrics:

- **Sum of Squared Differences** ($L_2^2$ norm): 
  $d(\text{QoI}, \text{Obs}) = \sum_{i=1}^{n}(\text{QoI}_i - \text{Obs}_i)^2$
  Penalizes large deviations heavily

- **Manhattan Distance** ($L_1$ norm): 
  $d(\text{QoI}, \text{Obs}) = \sum_{i=1}^{n}|\text{QoI}_i - \text{Obs}_i|$
  Robust to outliers

- **Chebyshev Distance** ($L_\infty$ norm): 
  $d(\text{QoI}, \text{Obs}) = \max_{i=1,...,n}|\text{QoI}_i - \text{Obs}_i|$
  Focuses on maximum deviation

### Fitness Functions
Distance values are transformed into fitness values (to be maximized) using one of two methods:

**Standard Transformation (default):**
$
f(\theta) = \frac{1}{1 + d(\text{QoI}, \text{Obs})}
$

**Exponential Transformation:**
$
f(\theta) = \exp(-d(\text{QoI}, \text{Obs}))
$

Both transformations ensure:
- All objectives are in (0, 1] range (values closer to 1 = better fit, closer to 0 = worse fit)
- Numerical stability for optimization algorithms
- Proper hypervolume computation for multi-objective optimization

The exponential transformation provides steeper gradients for small distances, which can be beneficial when fine-tuning parameters near optimal values. Enable exponential transformation by setting `use_exponential_fitness: True` in the `bo_options`.

### Example

The entry point is `CalibrationContext` (from `uq_physicell.bo`), which pairs a parameter search space and observed data with the PhysiCell model, plus `run_bayesian_optimization` to drive the optimization loop:

```python
from uq_physicell.bo import CalibrationContext, run_bayesian_optimization

model_config = {"ini_path": "uq_pc_struc.ini", "struc_name": "Model_struc_Calib"}

# df_cell is the cell DataFrame pcdl builds from each simulation's output.
qoi_functions = {
    "epi_":         lambda df_cell: len(df_cell[df_cell['cell_type'] == 'epithelial']),
    "epi_infected": lambda df_cell: len(df_cell[df_cell['cell_type'] == 'epithelial_infected']),
}

# Maps each QoI to the corresponding column in the observed-data CSV, plus the time column.
obs_data_columns = {
    "time":         "Time",
    "epi_":         "Healthy Epithelial Cells",
    "epi_infected": "Infected Epithelial Cells",
}

search_space = {
    "mac_phag_rate_infected": {"type": "real", "lower_bound": 0.7, "upper_bound": 1.5},
    "epi2infected_hfm":       {"type": "real", "lower_bound": 0.1, "upper_bound": 0.5},
}

bo_options = {
    "num_initial_samples": 10,  # random evaluations before GP fitting starts
    "num_iterations": 30,       # BO iterations after the initial samples
}

calib_context = CalibrationContext(
    db_path="results.db",
    obsData="observed_data.csv",
    obsData_columns=obs_data_columns,
    model_config=model_config,
    qoi_functions=qoi_functions,
    search_space=search_space,
    bo_options=bo_options,
)
run_bayesian_optimization(calib_context)
```

Results (samples, GP models, Pareto front) are stored in `results.db` and can be reloaded with `uq_physicell.database.bo_db.load_structure`, then explored with the plotting helpers in `uq_physicell.bo.plots` (`plot_parameter_space`, `plot_qoi_param`, `plot_parameter_vs_fitness`) and the analysis helpers in `uq_physicell.bo.utils` (`analyze_pareto_results`, `get_observed_qoi`).

**Full worked example:** {doc}`examples/virus-mac-new/ex7_Calib_BO`

(approximate-bayesian-computation-abc)=
## Approximate Bayesian Computation (ABC)

This module helps to solve the problem of parameter inference using a Sequential Monte Carlo scheme (more details in [pyabc documentation](https://pyabc.readthedocs.io/en/latest/what.html)). The ABC-SMC creates a sequence of intermediate posterior distributions that gradually approach the true, intractable posterior distribution. The ABC posterior in iteration $t$ is:

$$
\pi_{t}(\theta|y_0) \propto \pi(\theta)\cdot f_{\epsilon_t}(y_0|\theta)
$$

Where:
- $\pi(\theta)$ is the prior distribution of parameters.
- $\theta$ is the parameter vector to be inferred.
- $f_{\epsilon_t}(y_0|\theta)$ is the approximate likelihood based on the distance between simulated and observed data, with tolerance threshold $\epsilon_t$.
- $y_0$ is the observed data.
- $\epsilon_t$ is the tolerance threshold at iteration $t$, which typically decreases as $t$ increases.
- The goal is to obtain an approximation of the posterior distribution of parameters given the observational data $y_0$.

### Example

The entry point is `CalibrationContext` (from `uq_physicell.abc`), which pairs a prior, distance functions, and observed data with the PhysiCell model, plus `run_abc_calibration` to drive the ABC-SMC loop:

```python
from uq_physicell.abc import CalibrationContext, run_abc_calibration
from pyabc import RV, Distribution
import numpy as np

model_config = {"ini_path": "uq_pc_struc.ini", "struc_name": "Model_struc_Calib"}

qoi_functions = {
    "epi_":         lambda df_cell: len(df_cell[df_cell['cell_type'] == 'epithelial']),
    "epi_infected": lambda df_cell: len(df_cell[df_cell['cell_type'] == 'epithelial_infected']),
}

obs_data_columns = {
    "time":         "Time",
    "epi_":         "Healthy Epithelial Cells",
    "epi_infected": "Infected Epithelial Cells",
}

def euclidean_distance_epi(data1, data2):
    return np.sum((np.array(data1['epi_']) - np.array(data2['epi_'])) ** 2)

distance_functions = {
    "epi_": {"function": euclidean_distance_epi},
}

# Uniform prior: mac_phag_rate_infected ~ U[0.7, 1.5]
prior = Distribution(mac_phag_rate_infected=RV("uniform", 0.7, 0.8))

abc_options = {
    "max_populations": 2,
    "max_simulations": 100,
    "sampler": "multicore",   # or "dask" — requires the `dask` package (see installation)
    "num_workers": 6,
}

calib_context = CalibrationContext(
    db_path="results.db",
    obsData="observed_data.csv",
    obsData_columns=obs_data_columns,
    model_config=model_config,
    qoi_functions=qoi_functions,
    distance_functions=distance_functions,
    prior=prior,
    abc_options=abc_options,
)
history = run_abc_calibration(calib_context=calib_context)
```

`run_abc_calibration` returns the [pyabc](https://pyabc.readthedocs.io/en/latest/what.html) `History` object, which gives direct access to `pyabc.visualization` (posterior KDE matrices, epsilon schedules, etc.) in addition to the results stored in `results.db`.

**Full worked example:** {doc}`examples/virus-mac-new/ex8_ABC_Calib`

## References

1. [Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization" *Advances in Neural Information Processing Systems* 33 (2020)](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)

2. [Schalte, Y., et al. "pyABC: Efficient and robust easy-to-use approximate Bayesian computation" *Journal of Open Source Software* 7(74), 4304 (2022).](https://doi.org/10.21105/joss.04304)