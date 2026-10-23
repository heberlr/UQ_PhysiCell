# Working with Result Databases

Every study you run with UQ-PhysiCell — a sensitivity analysis, a Bayesian optimization calibration, or an ABC calibration — writes **one SQLite file** (`.db`) holding everything needed to reproduce and analyze it: the configuration, the parameter samples, and each simulation's results.

This page explains what is inside those files and how to read them. The short version:

```{important}
The tables that describe a study (metadata, parameter space, samples, seeds) are plain SQL and can be read with any SQLite tool. **The simulation results are not.** They are stored as serialized Python objects (pickled, and for model-analysis databases also compressed) in `BLOB` columns. Running `SELECT Data FROM Output` in a SQLite browser, `sqlite3`, or `pandas.read_sql` returns raw bytes, not numbers.

Always read results through the loader functions in `uq_physicell.database` (or the higher-level analysis functions built on them), which decompress and deserialize the data for you.
```

## Which kind of database do I have?

| Created by | Database kind | Read with |
|---|---|---|
| `ModelAnalysisContext` (SA, OAT, LHS, user-defined samples) | Model analysis (`MA`) | `uq_physicell.database.ma_db` |
| `CalibrationContext` + `run_bayesian_optimization` | Bayesian optimization (`BO`) | `uq_physicell.database.bo_db` |
| `CalibrationContext` + `run_abc_calibration` | ABC-SMC | pyABC's functions: [`pyabc.History`](https://pyabc.readthedocs.io/en/latest/api/pyabc.storage.html) (not `ma_db` / `bo_db`) |

You can check a file programmatically:

```python
from uq_physicell.database import get_database_type

get_database_type("results.db")   # 'MA', 'BO', 'ABC', or None
```

`get_database_type` reads only the `Metadata` table (nothing is deserialized, so it is safe on any file). It returns `None` for a missing file, a file that is not SQLite, or a database that UQ-PhysiCell did not create.

## Schema at a glance

Columns marked **BLOB** cannot be read meaningfully with SQL — use the loader listed next to them.

### Model-analysis databases (`ma_db`)

| Table | Columns | Readable with SQL? | Notes |
|---|---|---|---|
| `Metadata` | `Sampler`, `Ini_File_Path`, `StructureName`, `uq_physicell_version`, `pcdl_version`, `*_Hash` | Yes | One row. The hashes fingerprint the `.ini`, XML, and rules files used for the run. |
| `ParameterSpace` | `ParamName`, `Lower_Bound`, `Upper_Bound`, `ReferenceValue`, `Perturbation` | Yes | `load_parameter_space` |
| `QoIs` | `QOI_Name`, `QOI_Function` | Yes | `QOI_Function` is the QoI lambda stored as **source text**. QoI names with a `NULL` function mean a custom summary function computed them. An empty table or a single row of `NULL`s means raw-MCDS storage (see below). |
| `Samples` | `SampleID`, `ParamName`, `ParamValue` | Yes | Long format, one row per (sample, parameter). `load_samples` returns `{SampleID: {param: value}}`. |
| `Output` | `SampleID`, `ReplicateID`, `Seed`, **`Data`** | `Data`: **no** | One row per simulation. `Data` is a **zstd-compressed pickle** (older databases may use zlib). `load_output` |

What `Output.Data` contains depends on the storage mode chosen when the study ran (see {doc}`model_analysis`). `ma_db.get_storage_mode(db)` tells you which one a database uses, by deserializing a single stored run:

| `get_storage_mode` | How the study was run | `QoIs` table | `Data` after loading |
|---|---|---|---|
| `'qoi'` | QoI functions in `qois_info` | One row per QoI | `pandas.DataFrame`, one row per output time, columns `time`, `sampleID`, `replicateID`, one column per QoI, plus the sampled parameter values |
| `'qoi'` | Custom summary function returning such a DataFrame | QoI names, `NULL` functions | Same as above (at least `time` and one column per QoI) |
| `'raw_mcds'` | No QoI functions (`qois_info={}` or `None`) | Empty, or one `NULL` row | `list` of `pcdl.TimeStep` objects (the full simulation state at each saved time) |
| `'custom'` | Custom summary function returning anything else (including a DataFrame without a `time` column) | Depends | Whatever your summary function returned. `calculate_qoi_statistics` cannot process it; load it with `load_output` and handle it yourself. |

It returns `None` when no simulation has been stored yet.

The storage mode describes **what is stored**, not who computed it. A study that used QoI lambdas and one that used a custom summary function both report `'qoi'` as long as they store the same layout, and `calculate_qoi_statistics` reads both the same way. To find out **where the QoIs came from**, look at the `QoIs` table (plain SQL, nothing is deserialized):

```python
import sqlite3
import pandas as pd

with sqlite3.connect("results.db") as conn:
    print(pd.read_sql("SELECT QOI_Name, QOI_Function FROM QoIs", conn))
```

- **`QOI_Function` holds lambda text**: the built-in summary function ran these lambdas, and their source is stored with the results.
- **`QOI_Function` is `NULL` for named QoIs**: a custom summary function computed them. **Its code is not stored in the database**, and in `'qoi'` mode the raw simulation output is deleted after each run. Keep the summary function's source under version control next to the database, or the QoIs cannot be traced back to how they were computed.

```{tip}
**Writing a custom summary function whose results `calculate_qoi_statistics` can read:** return a `pandas.DataFrame` with one row per output time, a `time` column, and one column per QoI, named as in `qois_info`. Pass `qois_info={name: None, ...}` to `ModelAnalysisContext` so the QoI names are recorded in the `QoIs` table. Any other return value is stored as-is and reported as `'custom'`.
```

### Bayesian optimization databases (`bo_db`)

| Table | Columns | Readable with SQL? | Notes |
|---|---|---|---|
| `Metadata` | `BO_Method`, `ObsData_Path`, `Ini_File_Path`, `StructureName`, versions, `BO_Options`, `*_Hash` | Yes | `BO_Options` is a JSON string with the resolved BO configuration. |
| `ParameterSpace` | `ParamName`, `Type`, `Lower_Bound`, `Upper_Bound`, `Regulates` | Yes | `load_parameter_space` |
| `QoIs` | `QOI_Name`, `QOI_Function`, `ObsData_Column`, `QoI_distanceFunction`, `QoI_distanceWeight` | Yes | Links each QoI to its column in the observed data. |
| `GP_Models` | `IterationID`, **`GP_Model`**, `Score`, `ConvergenceStatus` | `GP_Model`: **no** | `GP_Model` is a `torch.save`d BoTorch model — `load_gp_models` (requires PyTorch). `Score` (hypervolume for multi-objective, best fitness for single-objective) and `ConvergenceStatus` (JSON) are plain SQL. |
| `Samples` | `IterationID`, `SampleID`, `ParamName`, `ParamValue` | Yes | Long format. `IterationID` is the BO iteration that proposed the sample. |
| `Output` | `SampleID`, **`ObjFunc`**, **`Noise_Std`**, **`Data`**, `Seeds` | Blobs: **no** | One row per sample (replicates aggregated). Blobs are **pickled** but not compressed: BO stores only QoI values per sample, which are small, while model-analysis databases may hold the full raw simulation state and therefore compress it. `Seeds` is a JSON list of per-replicate seeds. `load_output` |

After `bo_db.load_output`:

- `ObjFunc` — `dict` of `{qoi_name: fitness}`, averaged over replicates. Fitness is in `[0, 1]`, **higher is better** (it is a transformed distance to the observed data).
- `Noise_Std` — `dict` of `{qoi_name: standard error}` across replicates.
- `Data` — `dict` of `{replicate_id: {column: numpy array}}` with the simulated time series (`time`, each QoI, the parameter values) for every replicate.
- `Seeds` — `list` of the PhysiCell random seeds, ordered by replicate.

### ABC databases

ABC calibrations store their results in [pyABC](https://pyabc.readthedocs.io/)'s own database schema (`abc_smc`, `populations`, `models`, `particles`, `parameters`, `samples`, `summary_statistics`), which UQ-PhysiCell does not modify. UQ-PhysiCell adds:

| Table | Notes |
|---|---|
| `Metadata` | One row with `Method='ABC'`, observed data path, model configuration, and package versions. |
| `CandidateModels` | One row per candidate model (model selection), with its configuration, fixed parameters, prior summary, and config hashes. |
| `AdaptiveDistance` | Per-population weights when an adaptive distance is used. |

```{important}
**Read ABC results with pyABC's own functions, not with `ma_db` / `bo_db`.** UQ-PhysiCell has no loader for ABC databases: the particles, weights, distances, and summary statistics live in pyABC's tables and are serialized in pyABC's own format, so the `ma_db` / `bo_db` loaders (and plain SQL) cannot interpret them. Open the file with [`pyabc.History`](https://pyabc.readthedocs.io/en/latest/api/pyabc.storage.html) and use its methods and the `pyabc.visualization` module. Only the three UQ-PhysiCell tables above are plain SQL.
```

```python
import uq_physicell.abc   # import first: applies UQ-PhysiCell's pyABC compatibility patches
import pyabc

history = pyabc.History("sqlite:///results.db")
```

`run_abc_calibration` also returns this `History` object directly. Commonly used methods:

| I want… | Use |
|---|---|
| Posterior samples and weights of model `m` at the last generation | `df, w = history.get_distribution(m=0, t=history.max_t)` |
| Number of generations, total simulations run | `history.n_populations`, `history.max_t`, `history.total_nr_simulations` |
| Epsilon schedule, accepted particles, and samples per generation | `history.get_all_populations()` |
| Model posterior probabilities (model selection) | `history.get_model_probabilities()` |
| Summary statistics of accepted particles | `w, sum_stats = history.get_weighted_sum_stats_for_model(m=0, t=history.max_t)` |
| Plots (KDE matrix, epsilons, credible intervals, model probabilities) | `pyabc.visualization.plot_kde_matrix(df, w)`, `plot_epsilons(history)`, ... |

Two practical notes:

- **Import `uq_physicell.abc` before reading.** It patches pyABC so that databases whose summary statistics were written on a machine with a different `pyarrow` setup (for example, started on a cluster and resumed on a workstation) still load, instead of failing with `ArrowInvalid: Parquet magic bytes not found`.
- **ABC databases are tied to pyABC's storage format.** pyABC versions its database schema, and a newer pyABC refuses to open a database written in an older format, failing with *"Database has version 1, latest format version is 2"*. The pyABC version that wrote a database is recorded in `Metadata.pyabc_version`. Either read it with a compatible pyABC, or upgrade a copy with pyABC's migration tool: `abc-migrate --src old.db --dst new.db`. UQ-PhysiCell cannot read ABC databases on its own, so this limitation applies to every ABC result.

See {doc}`calibration` and {doc}`examples/virus-mac-new/ex8_ABC_Calib` for a worked example.

## Which function answers which question?

### Model-analysis databases

| I want… | Use |
|---|---|
| Sampler, model, versions | `ma_db.load_metadata(db)` |
| Parameter names and ranges | `ma_db.load_parameter_space(db)` |
| Parameter values of each sample | `ma_db.load_samples(db)` |
| Which (sample, replicate) runs exist, and their seeds — without loading results | `ma_db.load_output(db, load_data=False, load_seed=True)` |
| What kind of results are stored (QoIs or raw simulation state) | `ma_db.get_storage_mode(db)` |
| Raw results of specific runs | `ma_db.load_output(db, sample_ids=[...], replicate_ids=[...])` |
| QoI time series flattened into columns (precomputed-QoI mode only) | `ma_db.load_data_unserialized(db)` |
| Everything at once | `ma_db.load_structure(db)` |
| **Per-sample mean / std / MCSE of QoIs (either storage mode)** | `model_analysis.calculate_qoi_statistics(db, qoi_funcs)` |
| Sensitivity indices | `model_analysis.get_sa_results(db, qoi_names, df_mean, method)` |
| Check that every sample has output | `ma_db.check_db_consistency(db)` |

For most analyses, start from `calculate_qoi_statistics`: it detects the storage mode, computes the QoIs if needed, aggregates replicates, and returns tidy DataFrames indexed by `(SampleID, time)`.

```{tip}
Loading raw-MCDS results is memory-hungry: every row holds the full simulation state. Filter with `sample_ids` / `replicate_ids`, or use `calculate_qoi_statistics(..., chunk_size=..., n_jobs=...)`, which streams through the database in chunks.
```

### BO databases

| I want… | Use |
|---|---|
| Metadata, parameter space, QoIs | `bo_db.load_metadata`, `bo_db.load_parameter_space`, `bo_db.load_qois` |
| Samples and the iteration that proposed them | `bo_db.load_samples(db, iteration_ids=None)` |
| Fitness, noise, simulated time series, seeds | `bo_db.load_output(db, sample_ids=None)` |
| Convergence history | `SELECT IterationID, Score, ConvergenceStatus FROM GP_Models` (no PyTorch needed) |
| Fitted GP surrogate models | `bo_db.load_gp_models(db)` |
| Everything at once | `bo_db.load_structure(db)` |
| Pareto front | `bo.utils.analyze_pareto_results(df_qois, df_samples, df_output)` |
| Observed data aligned with QoI names | `bo.utils.get_observed_qoi(obs_csv, df_qois)` |

## Key concepts

- **`SampleID`** identifies one parameter set. In model-analysis databases it indexes the `Samples` table; in BO databases it is unique across all iterations.
- **`ReplicateID`** identifies one stochastic repetition of a sample (`0 … n_replicates-1`). Model-analysis databases store one `Output` row per (sample, replicate). BO databases store one row per sample and keep the replicates inside `Data`.
- **Seeds** record the PhysiCell random seed of every replicate (`Output.Seed` in model-analysis databases, the `Output.Seeds` JSON list in BO databases), so any single simulation can be rerun exactly. They are `NULL` for runs made before seed tracking was added.
- **`IterationID`** (BO only) is the optimization iteration. Iteration 0 holds the initial space-filling samples.
- **Configuration hashes** (`Ini_Hash`, `XML_Hash`, `Rules_Hash`, `Structure_Config_Hash`, `Effective_Run_Hash`) fingerprint the exact model configuration. Resuming a study against a changed configuration is refused because of them, and they let you confirm two databases came from the same model.

## Recipes

### Look inside any database without deserializing anything

Safe for files you did not create yourself (see {ref}`security`):

```python
import sqlite3
import pandas as pd

with sqlite3.connect("results.db") as conn:
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    print(tables)
    print(pd.read_sql("SELECT * FROM Metadata", conn).T)
    print(pd.read_sql("SELECT * FROM ParameterSpace", conn))
```

### Final-time QoI values per sample (model analysis)

```python
from uq_physicell.model_analysis import calculate_qoi_statistics

# Precomputed-QoI databases: pass the stored QoI names with None.
# Raw-MCDS databases: pass the lambdas to compute, e.g. {"live_cells": lambda df_cell: ...}
df_mean, df_std, df_mcse = calculate_qoi_statistics("results.db", {"live_cells": None, "interferon_mean": None})

t_final = df_mean.index.get_level_values("time").max()
final = df_mean.xs(t_final, level="time")   # one row per SampleID
```

### Join parameters, seeds, and results into one table (model analysis)

```python
import pandas as pd
from uq_physicell.database import ma_db

params = pd.DataFrame.from_dict(ma_db.load_samples("results.db"), orient="index")
params.index.name = "SampleID"
runs = ma_db.load_output("results.db", load_data=False, load_seed=True)[["SampleID", "ReplicateID", "Seed"]]

table = runs.merge(params, left_on="SampleID", right_index=True).merge(final, left_on="SampleID", right_index=True)
table.to_csv("results_summary.csv", index=False)
```

### Best calibrated parameters (BO)

```python
import pandas as pd
from uq_physicell.database import bo_db

out = bo_db.load_output("results.db")
fitness = pd.DataFrame(out["ObjFunc"].tolist(), index=out["SampleID"])          # higher is better
params = bo_db.load_samples("results.db").pivot(index="SampleID", columns="ParamName", values="ParamValue")

best = fitness.mean(axis=1).idxmax()   # for multi-objective runs, prefer the Pareto front
print(best, fitness.loc[best].to_dict(), params.loc[best].to_dict())
```

### Has my BO run converged?

Works without PyTorch, while the run is still going:

```python
import json, sqlite3
import pandas as pd

with sqlite3.connect("results.db") as conn:
    hist = pd.read_sql("SELECT IterationID, Score, ConvergenceStatus FROM GP_Models ORDER BY IterationID", conn)
hist["ConvergenceStatus"] = hist["ConvergenceStatus"].map(lambda s: json.loads(s) if s else None)
print(hist[["IterationID", "Score"]].tail())
print(hist["ConvergenceStatus"].iloc[-1])
```

## Maintenance

- **Schema upgrades.** Databases created by older versions are upgraded in place, additively, when a study resumes or when `bo_db.load_structure` is called. To upgrade explicitly: `ma_db.migrate_ma_database(db)` or `bo_db.migrate_bo_database(db)`. Pass `migrate=False` to `bo_db.load_structure` to read a read-only or archived file without touching it.
- **Compression.** `compression.migrate_to_zstd(input_db, output_db)` rewrites an older model-analysis database with zstd compression, often shrinking it substantially. Reading handles zstd, zlib, and uncompressed data transparently.
- **Incomplete studies.** `ma_db.check_db_consistency(db)` reports samples with no output. Rerunning the same context resumes only the missing simulations.

(security)=
## Portability and security

```{warning}
Loading a result (`load_output`, `load_structure`, `load_gp_models`, `calculate_qoi_statistics`, ...) **unpickles** data, and unpickling can execute arbitrary code. Only load results from databases you trust — your own runs, or files from a trusted source such as the project's Zenodo records. The plain-SQL tables can always be inspected safely, as in the first recipe above.
```

Pickled results also depend on the library versions that wrote them. Raw-MCDS databases contain `pcdl.TimeStep` objects, and many results contain pandas or NumPy objects; very different versions of `pcdl` or pandas may fail to load them. `Metadata` records `uq_physicell_version` and `pcdl_version` so you can recreate a compatible environment. For long-term archiving or sharing with non-Python tools, also export the analysis tables you need (e.g. the output of `calculate_qoi_statistics`) to CSV or Parquet.
