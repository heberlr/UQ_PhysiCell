# Model Calibration

This section covers the methods of calibration available in UQ-PhysiCell: {ref}`Bayesian Optimization <bayesian-optimization>` and {ref}`Approximate Bayesian Computation (ABC) <approximate-bayesian-computation-abc>`, plus {ref}`Model Selection <model-selection>` (ABC-SMC over several competing models). Note that this package allow another forms of calibration via API, including optimization algorithms like least-squares, Nelder-Mead, genetic algorithm, MCMC, amoung others.


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

Results (samples, GP models, Pareto front) are stored in `results.db` and can be reloaded with `uq_physicell.database.bo_db.load_structure`, then explored with the plotting helpers in `uq_physicell.bo.plots` (`plot_parameter_space`, `plot_qoi_param`, `plot_parameter_vs_fitness`) and the analysis helpers in `uq_physicell.bo.utils` (`analyze_pareto_results`, `get_observed_qoi`). See {doc}`databases` for the database layout and recipes (best parameters, convergence history).

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
    # data1 (the simulated summary) is None when that particle's PhysiCell run
    # failed; returning np.inf rejects the particle instead of crashing on
    # None['epi_'] and killing the whole calibration run.
    if data1 is None:
        return np.inf
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

**Handling a failed simulation.** A PhysiCell crash (segfault, malformed output, a transient filesystem hiccup, ...) is a property of one particle, not of the whole calibration. When that happens, `CalibrationContext` catches it and passes `None` as the simulated data instead of letting the exception kill the pyABC worker process — every custom distance function must therefore check for `None` (as above) and return `np.inf` to reject that particle, the same convention used for a malformed or short simulation output. The built-in distance functions in `uq_physicell.utils.distances` (`SumSquaredDifferences`, `Manhattan`, `Chebyshev`) already do this.

**Parameters on very different scales.** Between generations, ABC-SMC proposes new parameters from a kernel fitted to the previous population. Parameters often differ by orders of magnitude (a rate of `1e-5` 1/min next to a half-max of `20` mmHg), and a kernel fitted in those raw units can collapse: it stops perturbing the small parameters and gives most new proposals zero density. The default `transition_strategy="multivariate"` therefore uses `ScaledMultivariateNormalTransition`, which fits the kernel on standardized parameters and reports densities in the original units, so no rescaling of your priors is needed. `transition_strategy="local"` selects pyABC's `LocalTransition` instead. Each candidate model gets its own kernel.

**Safeguards between generations.** Before every generation, the run checks that each model's fitted kernel gives positive density to its own proposals, and that the previous generation's importance weights are finite. If a model's kernel is degenerate, or more than `abc_options["max_zero_weight_fraction"]` (default `0.05`) of its particles ended up with a zero or non-finite weight, the run stops with an `ABCSafeguardError` naming the model, before any simulation of the next generation runs. Continuing would silently give that model a wrong probability and posterior. Completed generations stay in the database.

**Full worked example:** {doc}`examples/virus-mac-new/ex8_ABC_Calib`

(model-selection)=
## Model Selection

ABC-SMC can also ask *which model* the data support, not only which parameters. Several candidate models, each with its own PhysiCell configuration (settings file, rules file, free parameters) and its own prior, are calibrated against the same observed data in a single run. pyABC then returns a **posterior probability for each model**:

$$
P(m \mid y_0) \propto P(m) \int P(y_0 \mid \theta, m)\, P(\theta \mid m)\, d\theta
$$

Where:
- $P(m)$ is the prior probability of model $m$ (uniform over the candidates by default).
- The integral is the marginal likelihood of the data under model $m$. It acts as a built-in Occam's razor: a richer model that spreads its prior-predictive mass over a wider range of outcomes is penalized unless its extra mechanism genuinely improves the fit.
- In ABC the likelihood is approximated through the distance between simulated and observed data, so model probabilities are the share of accepted particles that came from each model in the final population.

### Example

Candidates are passed through `abc_options["models"]`, a list of dictionaries. Each requires a `name`, a `model_config` and a `prior`; `fixed_params` and `num_replicates` are optional. When `models` is given, the top-level `model_config` and `prior` arguments of `CalibrationContext` are omitted (each candidate carries its own):

```python
from uq_physicell.abc import CalibrationContext, run_abc_calibration
from pyabc import RV, Distribution

# Three hypotheses for what limits growth, defined as sections of uq_config.ini
models = [
    {"name": "ModelA",  # mechanical feedback: pressure decreases cycle entry
     "model_config": {"ini_path": "uq_config.ini", "struc_name": "ModelA"},
     "prior": Distribution(cell_cycle_entry=RV("uniform", 1111, 555),
                           apoptosis_rate=RV("uniform", 4.25e-5, 2.13e-5))},
    {"name": "ModelB",  # nutrient limitation: oxygen increases cycle entry
     "model_config": {"ini_path": "uq_config.ini", "struc_name": "ModelB"},
     "prior": Distribution(apoptosis_rate=RV("uniform", 4.25e-5, 2.13e-5),
                           oxygen_cell_cycle_entry_sat=RV("uniform", 5.76e-4, 2.88e-4),
                           oxygen_cell_cycle_entry_hfm=RV("uniform", 10, 20))},
    # ... ModelC combines both mechanisms
]

abc_options = {"models": models, "max_populations": 5, "population_size": 250, "sampler": "multicore"}

calib_context = CalibrationContext(
    db_path="model_selection.db",
    obsData="observed_data.csv",
    obsData_columns=obs_data_columns,
    qoi_functions=qoi_functions,
    distance_functions=distance_functions,   # the same distances are used for every model
    abc_options=abc_options,
)
history = run_abc_calibration(calib_context=calib_context)
probs = history.get_model_probabilities()    # one row per population, one column per model
```

Results are stored in a regular ABC database. In addition to pyABC's own tables, UQ-PhysiCell writes a `CandidateModels` table with one row per candidate (name, structure, number of replicates and the PhysiCell configuration fingerprint), which maps pyABC's integer model index to the model name. It is a separate table because pyABC already owns a lowercase `models` table, and SQLite table names are case-insensitive. See {doc}`databases`.

**Interpreting the result.** Posterior probabilities are relative to the candidate set: they say which of the proposed mechanisms the data favor, not that the winner is correct. In the worked example the observed data were generated from ModelA, and the final population gives ModelA about 0.75, ModelC (which contains ModelA's mechanism plus another) about 0.18 and ModelB about 0.07. Check that the result is stable across populations and consider repeating the run with other random seeds before drawing conclusions. Distance functions must treat a failed simulation the same way as in single-model ABC (return `np.inf`).

**Current limitations.** Model selection is available through the Python API. The GUI does not currently support ABC calibration or model selection.

**Full worked example:** {doc}`examples/template/ex11_ABC_ModelSelection`

## References

1. [Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization" *Advances in Neural Information Processing Systems* 33 (2020)](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)

2. [Schalte, Y., et al. "pyABC: Efficient and robust easy-to-use approximate Bayesian computation" *Journal of Open Source Software* 7(74), 4304 (2022).](https://doi.org/10.21105/joss.04304)