# Model Analysis

This section covers the analysis tools available in UQ-PhysiCell for understanding model behavior, parameter importance, and convergence properties.

## Overview

Model analysis is essential for:
- Understanding parameter sensitivity and importance
- Validating model predictions and stability
- Identifying potential issues with parameter identifiability
- Optimizing computational resources
- Ensuring model convergence and reliability

## Sensitivity Analysis

Sensitivity analysis helps identify which parameters have the most significant impact on model outputs. UQ-PhysiCell provides both global and local sensitivity analysis methods.

### Global Sensitivity Analysis (GSA)

Global sensitivity analysis examines parameter effects across the entire parameter space:

#### Sobol Indices
Sobol indices quantify the contribution of each parameter to output variance:
- **First-order indices (S₁)**: Individual parameter contributions
- **Total-order indices (Sₜ)**: Parameter contributions including interactions
- **Second-order indices (S₂)**: Pairwise parameter interactions

```python
from uq_physicell.model_analysis import SobolAnalysis

# Set up Sobol analysis
sobol_analysis = SobolAnalysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    n_samples=1000,  # Number of samples for analysis
    calc_second_order=True  # Include interaction effects
)

# Run analysis
sobol_results = sobol_analysis.run()

# Extract results
first_order = sobol_results['S1']
total_order = sobol_results['ST']
second_order = sobol_results['S2']

# Visualize results
sobol_analysis.plot_indices(save_path="sobol_indices.png")
```

#### Morris Method
The Morris method provides efficient screening for parameter importance:

```python
from uq_physicell.model_analysis import MorrisAnalysis

morris_analysis = MorrisAnalysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    num_levels=4,  # Discretization levels
    grid_jump=2,   # Grid jump size
    num_trajectories=50  # Number of Morris trajectories
)

morris_results = morris_analysis.run()

# Extract sensitivity measures
mu_star = morris_results['mu_star']  # Mean absolute effects
sigma = morris_results['sigma']      # Standard deviations
mu = morris_results['mu']            # Mean effects

# Plot Morris screening
morris_analysis.plot_screening(save_path="morris_screening.png")
```

### Local Sensitivity Analysis

Local sensitivity analysis examines parameter effects around specific points:

#### Finite Difference Method
Compute gradients using finite differences:

```python
from uq_physicell.model_analysis import LocalSensitivity

local_sa = LocalSensitivity(
    model_config=model_config,
    qoi_functions=qoi_functions,
    base_parameters=best_parameters,  # Point of interest
    perturbation=0.01,  # Relative perturbation size
    method="central"    # 'forward', 'backward', or 'central'
)

# Compute local sensitivities
gradients = local_sa.compute_gradients()
elasticities = local_sa.compute_elasticities()

# Visualize local sensitivity
local_sa.plot_tornado(save_path="tornado_plot.png")
```

#### One-at-a-Time (OAT) Analysis
Vary parameters individually while keeping others fixed:

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

### Sensitivity Analysis Workflow

Complete workflow for comprehensive sensitivity analysis:

```python
from uq_physicell.model_analysis import comprehensive_sensitivity_analysis

# Run comprehensive analysis
sa_results = comprehensive_sensitivity_analysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    base_parameters=calibrated_parameters,
    methods=['sobol', 'morris', 'local'],
    n_samples=1000,
    save_results=True,
    output_dir="sensitivity_analysis"
)

# Generate comprehensive report
sa_results.generate_report("sensitivity_report.html")
```

## Model Convergence

*[To be implemented in future versions]*

Model convergence analysis provides tools to assess when models reach stable, reliable states.

### Temporal Convergence

Analyze convergence in time domain:

#### Steady-State Detection
Identify when model dynamics reach equilibrium:

```python
# Future API design
from uq_physicell.model_analysis import TemporalConvergence

temporal_conv = TemporalConvergence(
    simulation_data=time_series_data,
    variables=['total_cells', 'live_cells'],
    detection_method='variance_threshold',
    window_size=100,
    tolerance=1e-6
)

steady_state_time = temporal_conv.detect_steady_state()
convergence_metrics = temporal_conv.compute_metrics()
```

#### Transient Analysis
Characterize initial model behavior:

```python
transient_analysis = temporal_conv.analyze_transients(
    phases=['initialization', 'growth', 'equilibrium'],
    characteristic_times=True
)
```

### Parameter Convergence

Assess parameter estimation stability:

#### Optimization Convergence
Monitor parameter convergence during Bayesian optimization:

```python
from uq_physicell.model_analysis import ParameterConvergence

param_conv = ParameterConvergence(database_path="calibration.db")

# Analyze parameter evolution
evolution_metrics = param_conv.analyze_evolution()
convergence_diagnostics = param_conv.convergence_diagnostics()

# Plot convergence
param_conv.plot_parameter_traces(save_path="param_convergence.png")
param_conv.plot_hypervolume_evolution(save_path="hypervolume.png")
```

#### Cross-Validation Analysis
Assess parameter generalizability:

```python
cv_analysis = param_conv.cross_validation_analysis(
    k_folds=5,
    validation_metrics=['rmse', 'r_squared', 'mae']
)
```

### Spatial Convergence

*[Planned for future implementation]*

Verify spatial discretization adequacy:

#### Grid Independence Study
```python
# Future implementation
spatial_conv = SpatialConvergence(
    model_config=model_config,
    grid_refinements=[1, 2, 4, 8],
    convergence_variables=['cell_density', 'oxygen_concentration']
)

grid_independence = spatial_conv.grid_independence_study()
```

## Analysis Utilities

### Data Processing
Tools for handling and processing analysis results:

```python
from uq_physicell.model_analysis.utils import (
    process_sensitivity_data,
    normalize_parameters,
    compute_ranking_metrics
)

# Process and rank parameters by importance
processed_data = process_sensitivity_data(sobol_results)
parameter_ranking = compute_ranking_metrics(processed_data)
```

### Visualization
Comprehensive plotting functions for analysis results:

```python
from uq_physicell.model_analysis.plotting import (
    plot_sensitivity_heatmap,
    plot_parameter_correlations,
    plot_convergence_diagnostics
)

# Create comprehensive plots
plot_sensitivity_heatmap(
    sensitivity_results,
    parameters=list(search_space.keys()),
    qois=list(qoi_functions.keys()),
    save_path="sensitivity_heatmap.png"
)
```

## Best Practices

### Sensitivity Analysis
1. **Sample Size**: Use at least 1000 samples for reliable Sobol indices
2. **QoI Selection**: Focus on the most important model outputs
3. **Parameter Ranges**: Use realistic parameter bounds based on literature
4. **Method Selection**: Use Morris for screening, Sobol for detailed analysis
5. **Validation**: Cross-validate results with different methods

### Convergence Analysis
1. **Multiple Metrics**: Use several convergence criteria simultaneously
2. **Temporal Windows**: Choose appropriate time windows for analysis
3. **Baseline Comparison**: Compare with known analytical solutions when available
4. **Documentation**: Record convergence settings for reproducibility

### Computational Efficiency
1. **Parallel Execution**: Leverage multiple cores for sensitivity analysis
2. **Adaptive Sampling**: Use efficient sampling strategies for large parameter spaces
3. **Caching**: Store intermediate results to avoid recomputation
4. **Progressive Analysis**: Start with coarse analysis, refine as needed

## Integration with Bayesian Optimization

Model analysis and Bayesian optimization work synergistically:

### Pre-Optimization Analysis
- Use sensitivity analysis to identify important parameters
- Reduce parameter space dimensionality
- Set appropriate parameter bounds

### During Optimization
- Monitor parameter convergence
- Detect potential identifiability issues
- Adjust optimization strategy based on sensitivity

### Post-Optimization Analysis
- Validate optimized parameters with sensitivity analysis
- Assess parameter identifiability
- Evaluate model robustness around optimal solutions

```python
# Integrated workflow example
def integrated_analysis_workflow(model_config, search_space, obs_data):
    """Complete analysis workflow combining SA and BO."""
    
    # 1. Initial sensitivity analysis
    sa_results = run_sensitivity_analysis(
        model_config, search_space, method='morris'
    )
    
    # 2. Reduce parameter space based on sensitivity
    important_params = filter_important_parameters(sa_results, threshold=0.1)
    
    # 3. Run Bayesian optimization with reduced space
    calib_results = run_bayesian_optimization(
        model_config, important_params, obs_data
    )
    
    # 4. Post-optimization analysis
    final_sa = run_sensitivity_analysis(
        model_config, important_params, 
        base_params=calib_results.best_parameters
    )
    
    return {
        'calibration': calib_results,
        'pre_sensitivity': sa_results,
        'post_sensitivity': final_sa
    }
```

This integrated approach ensures robust, well-understood model calibration with comprehensive uncertainty quantification.
