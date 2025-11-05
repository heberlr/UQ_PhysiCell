# Model Analysis

This section covers the analysis tools available in UQ-PhysiCell for understanding model behavior, parameter importance, and convergence properties.

## Overview

Model analysis is essential for:
- Understanding parameter sensitivity and importance
- Identifying potential issues with parameter identifiability
- Ensuring model convergence and reliability

```{note}
Executions of simulations based on PhysiCell structures can be performed in multiple ways: 
- Serial: Single-threaded execution, suitable for small analyses.
- Inter-process: Multi-processing on a single node using concurrent.futures.
- Inter-node: Distributed execution across multiple nodes using MPI.  
Simulations can also be executed in a cluster environment. An example SLURM script demonstrating how to run simulations on a SLURM-managed cluster is provided in [slurm_script](../examples/slurm_script.sh).
```

## Sensitivity Analysis

Sensitivity analysis helps identify which parameters have the most significant impact on model outputs. UQ-PhysiCell provides both global and local sensitivity analysis methods.

### Global Sensitivity Analysis (GSA)

Global sensitivity analysis examines parameter effects across the entire parameter space on an quantity of interest (QoI). UQ Physicell supports multiple methods of GSA, but standard the [SALib python library](https://salib.readthedocs.io/en/latest/index.html) is used here.


### Local Sensitivity Analysis (LSA)

Local sensitivity analysis examines parameter effects around specific points, we set as the standard LSA the One-at-a-Time (OAT) Analysis, where we change parameters individually according a perturbation, while keeping others fixed.
