# Examples

This section provides practical examples of using UQ-PhysiCell for different scenarios, starting with model analysis and then moving to Bayesian optimization.

## Model Analysis Examples

### Basic Global Sensitivity Analysis

```python
from uq_physicell import PhysiCell_Model
from uq_physicell.model_analysis import (
    ModelAnalysisContext,
    run_global_sa,
    get_SA_problem
)

# Define the model configuration
model_config = {
    "ini_path": "PhysiCell_settings.xml",
    "struc_name": "tumor_growth",
    "numReplicates": 3
}

# Define parameter space for sensitivity analysis
search_space = {
    "proliferation_rate": {"type": "real", "lower_bound": 0.5, "upper_bound": 2.0},
    "apoptosis_rate": {"type": "real", "lower_bound": 0.01, "upper_bound": 0.3},
    "oxygen_threshold": {"type": "real", "lower_bound": 5.0, "upper_bound": 25.0}
}

# Define quantities of interest
qoi_functions = {
    "total_cells": "lambda df: df['total_cells'].iloc[-1]",
    "live_cells": "lambda df: df['live_cells'].iloc[-1]"
}

# Set up analysis context
analysis_context = ModelAnalysisContext(
    db_path="sensitivity_analysis.db",
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    sa_options={
        "method": "sobol",
        "n_samples": 1000,
        "calc_second_order": True
    }
)

# Run global sensitivity analysis
sa_results = run_global_sa(analysis_context)

# Extract results
first_order = sa_results['S1']
total_order = sa_results['ST']

print("Parameter Sensitivities:")
for param in search_space.keys():
    print(f"{param}: S1={first_order[param]:.3f}, ST={total_order[param]:.3f}")
```

### Morris Screening for Parameter Importance

```python
from uq_physicell.model_analysis import run_local_sa

# Quick parameter screening using Morris method
morris_context = ModelAnalysisContext(
    db_path="morris_screening.db",
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    sa_options={
        "method": "morris",
        "num_levels": 4,
        "grid_jump": 2,
        "num_trajectories": 50
    }
)

morris_results = run_local_sa(morris_context)

# Extract Morris measures
mu_star = morris_results['mu_star']  # Mean absolute effects
sigma = morris_results['sigma']      # Standard deviations

print("Morris screening results:")
for param in search_space.keys():
    print(f"{param}: μ*={mu_star[param]:.3f}, σ={sigma[param]:.3f}")
```

## Bayesian Optimization Examples

## Bayesian Optimization Examples

### Basic Bayesian Optimization

Here's a simple example of setting up and running Bayesian optimization for a PhysiCell model:

```python
from uq_physicell import PhysiCell_Model
from uq_physicell.bo import (
    CalibrationContext, 
    run_bayesian_optimization,
    SumSquaredDifferences,
    Manhattan
)

# Define the calibration context
calib_context = CalibrationContext(
    db_path="tumor_growth_calibration.db",
    obsData="experimental_tumor_data.csv",
    obsData_columns={
        "total_cells": "Cell_Count",
        "live_cells": "Live_Count"
    },
    model_config={
        "ini_path": "PhysiCell_settings.xml",
        "struc_name": "tumor_spheroid",
        "numReplicates": 3
    },
    qoi_functions={
        "total_cells": "lambda df: df['total_cells'].iloc[-1]",
        "live_cells": "lambda df: df['live_cells'].iloc[-1]"
    },
    distance_functions={
        "total_cells": {"function": SumSquaredDifferences, "weight": 1e-6},
        "live_cells": {"function": Manhattan, "weight": 1e-5}
    },
    search_space={
        "proliferation_rate": {"type": "real", "lower_bound": 0.5, "upper_bound": 2.0},
        "apoptosis_rate": {"type": "real", "lower_bound": 0.01, "upper_bound": 0.3},
        "oxygen_threshold": {"type": "real", "lower_bound": 5.0, "upper_bound": 25.0}
    },
    bo_options={
        "num_initial_samples": 30,
        "num_iterations": 100,
        "acq_func_strategy": "combined",
        "use_exponential_fitness": True  # Use exponential fitness transformation
    }
)

# Run the optimization
run_bayesian_optimization(calib_context)
```

## Multi-Objective Optimization with Time Series Data

This example shows how to calibrate against time series experimental data:

```python
# Time series QoI functions
qoi_functions = {
    "growth_curve": """
    def extract_growth_curve(df):
        # Extract cell count at specific time points
        time_points = [0, 24, 48, 72, 96]  # hours
        counts = []
        for t in time_points:
            closest_time = df.iloc[(df['time'] - t).abs().argsort()[:1]]
            counts.append(closest_time['total_cells'].values[0])
        return np.array(counts)
    """,
    
    "viability_trend": """
    def extract_viability(df):
        # Calculate viability over time
        df['viability'] = df['live_cells'] / df['total_cells']
        # Return viability at key time points
        time_points = [24, 48, 72, 96]
        viabilities = []
        for t in time_points:
            closest_time = df.iloc[(df['time'] - t).abs().argsort()[:1]]
            viabilities.append(closest_time['viability'].values[0])
        return np.array(viabilities)
    """
}

# Corresponding experimental data structure
obsData_columns = {
    "growth_curve": ["T0_cells", "T24_cells", "T48_cells", "T72_cells", "T96_cells"],
    "viability_trend": ["T24_viability", "T48_viability", "T72_viability", "T96_viability"]
}
```

## Handling Parameter Non-Identifiability

When dealing with parameters that might be highly correlated or non-identifiable:

```python
# Enhanced options for non-identifiability issues
bo_options = {
    "num_initial_samples": 50,  # More initial samples
    "num_iterations": 150,
    "acq_func_strategy": "combined",  # Use combined strategy
    "diversity_weight": 0.1,          # Promote diversity
    "uncertainty_weight": 0.15,       # Focus on uncertain regions
    "soft_constraints": {             # Gentle parameter guidance
        "proliferation_rate": {"preferred_range": [0.8, 1.5], "weight": 0.05},
        "migration_speed": {"preferred_range": [1.0, 3.0], "weight": 0.03}
    }
}
```

## Custom Distance Functions

Define custom distance metrics for specific use cases:

```python
# Custom distance function for asymmetric penalties
custom_distance_functions = {
    "cell_count": {
        "function": """
        def asymmetric_loss(predicted, observed):
            diff = predicted - observed
            # Penalize over-prediction more heavily
            loss = np.where(diff > 0, 2 * diff**2, diff**2)
            return np.sum(loss)
        """,
        "weight": 1e-5
    }
}
```

## Results Analysis and Visualization

After optimization, analyze the results:

```python
from uq_physicell.bo.analysis import (
    extract_best_parameters, 
    plot_convergence, 
    plot_parameter_space,
    analyze_pareto_front
)

# Extract best parameters
best_params, best_iteration = extract_best_parameters("calibration.db")
print(f"Best parameters found at iteration {best_iteration}:")
for param, value in best_params.items():
    print(f"  {param}: {value:.4f}")

# Plot convergence
plot_convergence("calibration.db", save_path="convergence.png")

# Visualize parameter space exploration
plot_parameter_space("calibration.db", save_path="parameter_space.png")

# Analyze Pareto front for multi-objective problems
pareto_solutions = analyze_pareto_front("calibration.db")
```

## Resuming Interrupted Optimizations

UQ-PhysiCell automatically handles resumption of interrupted optimizations:

```python
# If the database already exists, optimization will resume
# from the last completed iteration
calib_context = CalibrationContext(
    db_path="existing_calibration.db",  # Existing database
    # ... same configuration as before
)

# This will automatically detect existing progress and continue
run_bayesian_optimization(calib_context)
```

## Advanced Configuration

For complex scenarios with custom model setups:

```python
# Advanced model configuration
model_config = {
    "ini_path": "custom_settings.xml",
    "struc_name": "complex_model",
    "numReplicates": 5,
    "custom_executable": "./custom_physicell",
    "environment_vars": {
        "OMP_NUM_THREADS": "4",
        "PHYSICELL_OUTPUT_DIR": "./custom_output"
    },
    "timeout": 3600,  # 1 hour timeout per simulation
    "cleanup_files": ["*.svg", "*.mat"]  # Files to clean up after each run
}
```

These examples demonstrate the flexibility and power of the UQ-PhysiCell framework for various calibration scenarios.

## Sensitivity Analysis Examples

### Basic Global Sensitivity Analysis

```python
from uq_physicell.model_analysis import SobolAnalysis

# Define the same model setup as for optimization
sobol_analysis = SobolAnalysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions={
        "total_cells": "lambda df: df['total_cells'].iloc[-1]",
        "live_cells": "lambda df: df['live_cells'].iloc[-1]"
    },
    n_samples=1000
)

# Run Sobol sensitivity analysis
results = sobol_analysis.run()

# Extract first-order and total-order indices
first_order = results['S1']
total_order = results['ST']

print("Parameter Sensitivities:")
for param in search_space.keys():
    print(f"{param}: S1={first_order[param]:.3f}, ST={total_order[param]:.3f}")

# Plot results
sobol_analysis.plot_indices(save_path="sobol_sensitivity.png")
```

### Morris Screening for Parameter Importance

```python
from uq_physicell.model_analysis import MorrisAnalysis

# Quick parameter screening
morris_analysis = MorrisAnalysis(
    model_config=model_config,
    search_space=search_space,
    qoi_functions=qoi_functions,
    num_trajectories=50
)

morris_results = morris_analysis.run()

# Identify most important parameters
important_params = morris_analysis.rank_parameters(threshold=0.1)
print(f"Most important parameters: {important_params}")

# Visualize screening results
morris_analysis.plot_screening(save_path="morris_screening.png")
```

### Integrated Sensitivity and Optimization Workflow

```python
from uq_physicell.model_analysis import ModelAnalysisContext, run_global_sa
from uq_physicell.bo import CalibrationContext, run_bayesian_optimization

def sensitivity_guided_optimization():
    """Example of using sensitivity analysis to guide optimization."""
    
    # Step 1: Initial parameter screening with Morris method
    morris_context = ModelAnalysisContext(
        db_path="morris_screening.db",
        model_config=model_config,
        search_space=full_search_space,
        qoi_functions=qoi_functions,
        sa_options={
            "method": "morris",
            "num_trajectories": 50
        }
    )
    
    screening_results = run_global_sa(morris_context)
    
    # Step 2: Filter important parameters based on Morris screening
    mu_star = screening_results['mu_star']
    important_params = {
        param: bounds for param, bounds in full_search_space.items() 
        if mu_star[param] > 0.05  # Threshold for importance
    }
    
    print(f"Reduced parameter space to: {list(important_params.keys())}")
    
    # Step 3: Reduced Bayesian optimization
    calib_context = CalibrationContext(
        db_path="sensitivity_guided_calibration.db",
        obsData=obs_data,
        search_space=important_params,
        model_config=model_config,
        qoi_functions=qoi_functions,
        bo_options={
            "num_initial_samples": 25,
            "num_iterations": 75,
            "use_exponential_fitness": True
        }
    )
    
    # Step 4: Run optimization
    run_bayesian_optimization(calib_context)
    
    return important_params

# Run the integrated workflow
important_parameters = sensitivity_guided_optimization()
```
```
