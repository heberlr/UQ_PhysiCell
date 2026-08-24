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
- Inter-node: Distributed execution across multiple nodes using MPI (requires the `mpi` extra, see {doc}`installation`).  
Simulations can also be executed in a cluster environment. An example SLURM script demonstrating how to run simulations on a SLURM-managed cluster is provided in [slurm_script](../examples/slurm_script.sh).
```

## Quick Example

The entry point for sensitivity analysis (and model analysis in general) is `ModelAnalysisContext`. It manages the parameter samples, runs the underlying PhysiCell simulations, and stores the samples plus each run's results in a SQLite database that the analysis and plotting functions read back from.

```python
from uq_physicell.model_analysis import ModelAnalysisContext, calculate_qoi_statistics, get_sa_results

model_config = {"ini_path": "Model_Struct.ini", "struc_name": "physicell_model_2"}

# Global samplers (Sobol, LHS, ...) use only lower_bound/upper_bound for sampling.
params_info = {
    "viral_replication_rate": {"lower_bound": 0.05, "upper_bound": 0.20},
    "min_virion_count":        {"lower_bound": 0.5,  "upper_bound": 1.5},
}

# qois_info={} stores the raw simulation state (the full list of MCDS timesteps per run)
# instead of precomputed QoI values — see "Data Storage" below.
context = ModelAnalysisContext(
    "results.db", model_config,
    sampler="Sobol",
    params_info=params_info,
    qois_info={},
    num_workers=8,
)
context.generate_samples(N=8)  # Sobol needs N x (2D+2) samples, D = number of parameters
context.run()

# QoIs are computed after the fact, straight from the raw data already stored in results.db —
# no re-simulation needed, and you're free to change or add QoIs later without rerunning.
# df_cell / df_subs are the cell/substrate DataFrames pcdl builds from each simulation's output.
qoi_funcs = {
    "epithelial_live": lambda df_cell: len(df_cell[(df_cell['dead'] == False) & (df_cell['cell_type'] == 'epithelial cell')]),
    "interferon_mean": lambda df_subs: df_subs['interferon'].mean(),
}

# Aggregate replicates into per-sample statistics (mean, std, relative MCSE)
df_mean, df_std, df_mcse = calculate_qoi_statistics("results.db", qoi_funcs)

# Compute Sobol indices (S1, ST) from the aggregated QoIs
sa_results, qoi_time_values = get_sa_results("results.db", list(qoi_funcs.keys()), df_mean, "Sobol Sensitivity Analysis")
```

`ModelAnalysisContext` also accepts `'OAT'` (one-at-a-time local SA), `'LHS'`, and other [SALib](https://salib.readthedocs.io/en/latest/index.html)-backed samplers, as well as a `'User-defined'` sampler where you supply your own samples via `context.set_samples(...)` instead of `generate_samples(...)`.

### Data Storage: Raw MCDS vs. Precomputed QoIs

Whether `context.run()` stores the full simulation state or precomputed QoI values depends entirely on `qois_info` — the QoI dictionary passed to `ModelAnalysisContext`:

- **`qois_info={}`** (empty), as in the example above — the full `list[pcdl.TimeStep]` (MCDS) objects are stored, preserving the complete simulation state. This is the safer default: `calculate_qoi_statistics` computes QoIs from this raw data afterward, so you can change your mind about what to measure — or add a QoI you hadn't thought of — without rerunning any simulations.
- **`qois_info` populated** with your QoI functions — each one runs at simulation time, only the resulting values are stored, and the raw PhysiCell output folder is deleted afterward. Smallest storage footprint, but you commit to those QoIs up front. Switch to this mode once your QoIs are settled and you need to scale to many runs (e.g. a sensitivity analysis with hundreds of samples), where the storage savings matter.

`calculate_qoi_statistics` works transparently with either mode — it detects what is stored and computes the requested QoIs either way, always returning the same long-format `(SampleID, time)` DataFrame. See {doc}`examples/ex2_storage_modes` for a full side-by-side comparison, including database size.

**Full worked examples:**
- {doc}`examples/ex1_context_setup` — inspecting a `ModelAnalysisContext` and using the `'User-defined'` sampler
- {doc}`examples/ex2_storage_modes` — precomputed QoIs vs. raw MCDS storage, and querying both the same way
- {doc}`examples/ex3_runSA_MultiTask` — global Sobol SA with multi-task (single-node) parallelization
- {doc}`examples/ex4_runSA_MPI` — the same Sobol SA distributed across nodes with MPI
- {doc}`examples/virus-mac-new/ex5_OAT_example` — local (OAT) sensitivity analysis
- {doc}`examples/virus-mac-new/ex6_LHS_example` — global SA with Latin Hypercube Sampling

## Sensitivity Analysis

Sensitivity analysis helps identify which parameters have the most significant impact on model outputs. UQ-PhysiCell provides both global and local sensitivity analysis methods.

### Global Sensitivity Analysis (GSA)

Global sensitivity analysis examines parameter effects across the entire parameter space on an quantity of interest (QoI). UQ Physicell supports multiple methods of GSA, but standard the [SALib python library](https://salib.readthedocs.io/en/latest/index.html) is used here.


### Local Sensitivity Analysis (LSA)

Local sensitivity analysis examines parameter effects around specific points, we set as the standard LSA the One-at-a-Time (OAT) Analysis, where we change parameters individually according a perturbation, while keeping others fixed.
