# Model Calibration

This section covers the methods of calibration available in UQ-PhysiCell: [Bayesian Optimization](#bayesian-optimization) and [Approximate Bayes Computation](#approximate-bayesian-computation-abc). Note that this package allow another forms of calibration via API, including optimization algorithms like least-squares, Nelder-Mead, genetic algorithm, MCMC, amoung others.


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

### 1. Parameter Space ($\Theta$)
The search space is defined by parameter bounds and types:
- **Real parameters**: Continuous variables with lower and upper bounds
- **Integer parameters**: Discrete variables with specified ranges
- **Categorical parameters**: Discrete choices from predefined sets

### 2. Quantities of Interest (QoIs)
QoIs are model outputs that correspond to experimental observables:
- **Time series data**: Cell counts, concentrations, spatial metrics over time
- **Aggregate metrics**: Final values, peak values, areas under curves
- **Derived quantities**: Ratios, differences, or complex functions of raw outputs

### 3. Distance Metrics
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

### 4. Fitness Functions
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

## References

1. [Balandat, M., et al. "BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization" *Advances in Neural Information Processing Systems* 33 (2020)](https://proceedings.neurips.cc/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html)

2. [Schalte, Y., et al. "pyABC: Efficient and robust easy-to-use approximate Bayesian computation" *Journal of Open Source Software* 7(74), 4304 (2022).](https://doi.org/10.21105/joss.04304)