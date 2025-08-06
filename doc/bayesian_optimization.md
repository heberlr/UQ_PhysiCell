# Bayesian Optimization for PhysiCell Model Calibration

## Overview

This module provides a comprehensive Bayesian optimization framework for calibrating PhysiCell models using multi-objective optimization with sophisticated handling of parameter non-identifiability issues. The framework is designed to efficiently find optimal parameter configurations that minimize the discrepancy between model predictions and observed experimental data.

## The Optimization Problem

### Problem Definition

The optimization problem is formulated as a multi-objective minimization:

```
minimize f(θ) = [d₁(θ), d₂(θ), ..., dₖ(θ)]
subject to θ ∈ Θ
```

Where:
- `θ` is the parameter vector to be optimized
- `Θ` is the feasible parameter space defined by bounds
- `dᵢ(θ)` is the distance between model predictions and observed data for the i-th quantity of interest (QoI)
- The goal is to find the Pareto-optimal set of parameters

### Key Components

#### 1. Parameter Space (Θ)
The search space is defined by parameter bounds and types:
- **Real parameters**: Continuous variables with lower and upper bounds
- **Integer parameters**: Discrete variables with specified ranges
- **Categorical parameters**: Discrete choices from predefined sets

#### 2. Quantities of Interest (QoIs)
QoIs are model outputs that correspond to experimental observables:
- **Time series data**: Cell counts, concentrations, spatial metrics over time
- **Aggregate metrics**: Final values, peak values, areas under curves
- **Derived quantities**: Ratios, differences, or complex functions of raw outputs

#### 3. Distance Functions
Multiple distance metrics are available to quantify model-data discrepancy:
- **Sum of Squared Differences**: L₂ norm, penalizes large deviations heavily
- **Manhattan Distance**: L₁ norm, robust to outliers
- **Chebyshev Distance**: L∞ norm, focuses on maximum deviation

#### 4. Objective Function Transformation
Distance values are transformed into fitness values using one of two methods:

**Standard Transformation (default):**
```
fitness = 1 / (1 + distance)
```

**Exponential Transformation:**
```
fitness = exp(-distance)
```

Both transformations ensure:
- All objectives are in [0, 1] range (0 = worst, 1 = best)
- Numerical stability for optimization algorithms
- Proper hypervolume computation for multi-objective optimization

The exponential transformation provides steeper gradients for small distances, which can be beneficial when fine-tuning parameters near optimal values. Enable exponential transformation by setting `use_exponential_fitness: True` in the `bo_options`.

## Model Analysis

### Sensitivity Analysis

Sensitivity analysis in UQ-PhysiCell helps identify which parameters have the most significant impact on model outputs. This is crucial for:

#### Global Sensitivity Analysis (GSA)
- **Sobol indices**: Quantify the contribution of each parameter to output variance
- **Morris method**: Efficient screening for parameter importance
- **Variance decomposition**: Understand parameter interactions

#### Local Sensitivity Analysis
- **Finite difference**: Compute gradients around specific parameter values
- **One-at-a-time (OAT)**: Vary one parameter while keeping others fixed
- **Parameter perturbation**: Assess local stability of model predictions

#### Implementation
```python
from uq_physicell.model_analysis import run_sensitivity_analysis

# Global sensitivity analysis
sobol_results = run_sensitivity_analysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    method="sobol",
    n_samples=1000
)

# Local sensitivity analysis
local_results = run_sensitivity_analysis(
    model_config=model_config,
    base_parameters=best_parameters,
    qoi_functions=qoi_functions,
    method="finite_difference",
    perturbation=0.01
)
```

### Model Convergence

*[To be implemented in future versions]*

Model convergence analysis will provide tools to assess:

#### Temporal Convergence
- **Steady-state detection**: Identify when model dynamics reach equilibrium
- **Transient analysis**: Characterize initial model behavior
- **Oscillation detection**: Identify periodic behavior patterns

#### Parameter Convergence
- **Optimization convergence**: Monitor parameter estimation stability
- **Cross-validation**: Assess parameter generalizability
- **Identifiability analysis**: Detect parameter correlation issues

#### Spatial Convergence
- **Grid independence**: Verify spatial discretization adequacy
- **Boundary effects**: Assess domain size influence
- **Resolution sensitivity**: Determine optimal spatial resolution

#### Implementation (Planned)
```python
# Future API design
from uq_physicell.model_analysis import analyze_convergence

convergence_results = analyze_convergence(
    simulation_data=simulation_output,
    convergence_criteria={
        "temporal": {"tolerance": 1e-6, "window": 100},
        "spatial": {"refinement_levels": 3},
        "parameter": {"cv_folds": 5}
    }
)
```

## Algorithm Architecture

### Multi-Objective Bayesian Optimization

The framework uses the **qNEHVI** (q-Noisy Expected Hypervolume Improvement) acquisition function, which:
1. **Builds surrogate models**: Gaussian Processes (GPs) model each objective function
2. **Selects promising candidates**: Acquisition function balances exploration vs exploitation
3. **Handles noise**: Accounts for simulation stochasticity through multiple replicates
4. **Optimizes multiple objectives**: Uses hypervolume improvement for Pareto optimization

### Enhanced Identification Strategies

To address parameter non-identifiability issues, the framework provides several enhancement strategies:

#### 1. Diversity Bonus (`diversity_bonus`)
Promotes exploration of unexplored parameter regions by adding a bonus term to the acquisition function based on distance to existing samples.

#### 2. Uncertainty Weighting (`uncertainty_weighting`)
Emphasizes regions with high model uncertainty, encouraging sampling where the GP is least confident.

#### 3. Soft Constraints (`soft_constraints`)
Provides gentle guidance toward preferred parameter ranges without hard constraints.

#### 4. Adaptive Scaling (`adaptive_scaling`)
Dynamically adjusts exploration based on optimization progress and convergence indicators.

#### 5. Combined Strategy (`combined`)
**[RECOMMENDED]** Uses multiple strategies together for robust performance across different problem types.

### Convergence and Restart Mechanisms

#### Convergence Detection
The algorithm monitors several metrics:
- **Hypervolume stability**: Plateau detection in hypervolume improvement
- **Parameter convergence**: Changes in best parameter estimates
- **Acquisition function values**: Diminishing expected improvement

#### Auto-Restart Capability
When stagnation is detected:
1. **Diagnose issues**: Analyze parameter identifiability and model behavior
2. **Adaptive restart**: Increase exploration parameters and continue optimization
3. **Progress preservation**: All previous samples and models are retained

## Workflow

### 1. Setup Phase
```python
# Create calibration context with acquisition strategy in bo_options
calib_context = CalibrationContext(
    db_path="calibration.db",
    obsData="experimental_data.csv",
    obsData_columns={"cell_count": "Total_Cells", "viability": "Viability"},
    model_config={
        "ini_path": "PhysiCell_settings.xml",
        "struc_name": "tumor_growth",
        "numReplicates": 3
    },
    qoi_functions={
        "cell_count": "lambda df: df['total_cells'].values",
        "viability": "lambda df: df['live_cells'].sum() / df['total_cells'].sum()"
    },
    distance_functions={
        "cell_count": {"function": SumSquaredDifferences, "weight": 1e-5},
        "viability": {"function": Manhattan, "weight": 1e-3}
    },
    search_space={
        "proliferation_rate": {"type": "real", "lower_bound": 0.1, "upper_bound": 2.0},
        "apoptosis_rate": {"type": "real", "lower_bound": 0.01, "upper_bound": 0.5},
        "migration_speed": {"type": "real", "lower_bound": 0.5, "upper_bound": 5.0}
    },
    bo_options={
        "num_initial_samples": 50,
        "num_iterations": 100,
        "acq_func_strategy": "combined",  # Enhanced strategy for non-identifiability
        "diversity_weight": 0.08,
        "uncertainty_weight": 0.12,
        "use_exponential_fitness": True  # Use exponential transformation
    }
)
```

### 2. Optimization Execution
```python
# Run the optimization
run_bayesian_optimization(calib_context)
```

### 3. Results Analysis
```python
# Extract best parameters
best_params, best_iteration = extract_best_parameters("calibration.db")

# Visualize parameter space and convergence
plot_parameter_space("calibration.db")
plot_convergence("calibration.db")

# Diagnose identification issues if needed
diagnose_identification_issues(calib_context)
```

## Database Structure

The framework uses SQLite for persistent storage with six main tables:

1. **Metadata**: Optimization configuration and hyperparameters
2. **ParameterSpace**: Parameter definitions and bounds
3. **QoIs**: Quantities of interest and distance function specifications
4. **GP_Models**: Gaussian Process models and hypervolume history
5. **Samples**: Parameter samples across all iterations
6. **Output**: Simulation results and objective function values

## Key Features

### Robustness
- **Noise handling**: Multiple replicates and noise-aware GPs
- **Numerical stability**: Careful scaling and hypervolume computation
- **Error recovery**: Graceful handling of simulation failures

### Scalability
- **Parallel evaluation**: Batch acquisition and concurrent simulation
- **Memory efficiency**: Streaming data processing and database storage
- **Resume capability**: Continue optimization from saved state

### Diagnostics
- **Convergence monitoring**: Real-time progress tracking
- **Identifiability analysis**: Detection of parameter correlation issues
- **Visualization tools**: Parameter space plots and convergence curves

### Flexibility
- **Custom QoIs**: User-defined functions for extracting model outputs
- **Multiple distance metrics**: Various options for model-data comparison
- **Configurable strategies**: Tunable enhancement methods for different problems

## Best Practices

1. **Initial Sampling**: Use 20-50 initial samples to build reliable surrogate models
2. **Scale Factor**: Adjust based on typical distance values (aim for fitness in [0.1, 0.9])
3. **QoI Weights**: Balance different objectives based on measurement uncertainty
4. **Strategy Selection**: Use "combined" strategy for most problems
5. **Replicates**: Use 3-5 replicates for stochastic models
6. **Convergence**: Monitor hypervolume and parameter stability
7. **Fitness Transformation**: Use exponential transformation for fine-tuning near optimal values
8. **Sensitivity Analysis**: Perform sensitivity analysis before optimization to identify key parameters

## Troubleshooting

### Common Issues
- **Zero hypervolume**: Increase scale_factor or check distance values
- **Poor convergence**: Try different identification strategies or increase iterations
- **Parameter correlation**: Use soft_constraints or diversity_bonus strategies
- **Simulation failures**: Check model configuration and parameter bounds

### Performance Optimization
- **Batch size**: Use q=1-5 for acquisition function optimization
- **GP fitting**: Monitor marginal log-likelihood for model quality
- **Memory usage**: Limit database size for very long optimizations

## References

1. Balandat, M., et al. "BoTorch: A framework for efficient Monte-Carlo Bayesian optimization." *Advances in Neural Information Processing Systems* 33 (2020).

2. Daulton, S., et al. "Differentiable expected hypervolume improvement for parallel multi-objective Bayesian optimization." *Advances in Neural Information Processing Systems* 33 (2020).

3. Ghahramani, A., et al. "UQ-PhysiCell: Quantifying and reducing uncertainties of PhysiCell models." *Bioinformatics* (2023).
