"""Setup + runner for ex11_ABC_ModelSelection.ipynb.

This module is the single source of truth for "what ex11 runs": the three
candidate models (ModelA/B/C), their priors, the QoI/distance functions, and
the ABC-SMC options. It is used two ways:

- As a script, for the real (cluster-scale, multi-hour) run:
    python uq_script.py <num_workers>        # or via uq_slurm.sh on SLURM
- As a module, imported by ex11_ABC_ModelSelection.ipynb so the notebook never
  redefines this setup -- it only loads the resulting database and visualizes
  it (`import uq_script`, then e.g. `uq_script.db_path`, `uq_script.calib_context`,
  `uq_script.get_model_probabilities()`).

The ABC "multicore" sampler runs on a single node: it forks `num_workers` worker
processes, so the SLURM job asks for one task with many CPUs (see uq_slurm.sh).
Each PhysiCell run stays single-threaded (omp_num_threads = 1 in uq_config.ini).
"""

import os, sys

# Cap BLAS/thread-pool libraries to 1 thread *before* numpy/scipy/pandas are
# imported. Parallelism here comes from running many independent PhysiCell
# worker processes, not from multi-threaded BLAS; left at its default, OpenBLAS
# spins up one thread per detected CPU core for every Python process (main or
# worker), which can blow through a shared machine's process/thread budget
# (RLIMIT_NPROC) long before any simulation runs. Set here so it also covers
# ad-hoc runs (e.g. `python uq_script.py 48`) that don't go through uq_slurm.sh.
for _blas_env in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_env, "1")

import sqlite3
import logging

import pandas as pd

from uq_physicell.abc import CalibrationContext, run_abc_calibration
from uq_physicell.utils import relative_rmse
from pyabc import RV, Distribution, History

# Timestamped so a plain `nohup python -u uq_script.py <n> > output.log 2>&1 &`
# is enough on its own -- no need to pipe through `awk` for timestamps, which is
# fragile under nohup: a pipeline run with `&` is one job sharing one process
# group, but `nohup` only protects the process it directly wraps. On SSH
# disconnect, the shell forwards SIGHUP to that whole job; an unprotected `awk`
# downstream dies, and every subsequent log write then fails silently (Python's
# logging.Handler.handleError swallows the resulting BrokenPipeError when its
# own fallback report to stderr fails too) -- the run keeps going, just mute.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

db_path = "ex11_ABC_ModelSelection.db"
obs_data_path = "ObsData.csv"

# Ground-truth parameters behind ObsData.csv (ModelA only).
dic_real_value = {"cell_cycle_entry": 1440.0, "apoptosis_rate": 5.787e-05}

# df_cell -> cell DataFrame per snapshot ; mcds_ts -> full time series
qoi_functions = {
    "live_cell_count":            lambda df_cell: len(df_cell[df_cell['dead'] == False]),
    "integrated_dead_cell_count": lambda mcds_ts: sum(len(mcds.get_cell_df()[mcds.get_cell_df()['dead'] == True]) for mcds in mcds_ts),
}

obs_data_columns = {
    "time":                       "Time",
    "live_cell_count":            "Live_Cells",
    "integrated_dead_cell_count": "Sum_Dead_Cells",
}

# ---------------------------------------------------------------------------
# The three candidate models (structure sections of uq_config.ini)
# ---------------------------------------------------------------------------
model_config_A = {"ini_path": "uq_config.ini", "struc_name": "ModelA"}
model_config_B = {"ini_path": "uq_config.ini", "struc_name": "ModelB"}
model_config_C = {"ini_path": "uq_config.ini", "struc_name": "ModelC"}

prior_A = Distribution(
    cell_cycle_entry=RV("uniform", 1111, 555),          # uniform on [1111, 1666] min
    apoptosis_rate=RV("uniform", 4.25e-5, 2.13e-5),     # uniform on [4.25e-5, 6.38e-5] 1/min
)
prior_B = Distribution(
    apoptosis_rate=RV("uniform", 4.25e-5, 2.13e-5),
    oxygen_cell_cycle_entry_sat=RV("uniform", 5.76e-4, 2.88e-4),  # uniform on [5.76e-4, 8.64e-4] 1/min
    oxygen_cell_cycle_entry_hfm=RV("uniform", 10, 20),            # uniform on [10, 30] mmHg
)
prior_C = Distribution(
    apoptosis_rate=RV("uniform", 4.25e-5, 2.13e-5),
    cell_cycle_entry=RV("uniform", 0, 5.76e-4),         # uniform on [0, 5.76e-4] 1/min (baseline rate)
    oxygen_cell_cycle_entry_sat=RV("uniform", 5.76e-4, 2.88e-4),
    oxygen_cell_cycle_entry_hfm=RV("uniform", 10, 20),
)

# Production run: drop `num_replicates` so the uq_config.ini value (5) is used.
models = [
    {"name": "ModelA", "model_config": model_config_A, "prior": prior_A},
    {"name": "ModelB", "model_config": model_config_B, "prior": prior_B},
    {"name": "ModelC", "model_config": model_config_C, "prior": prior_C},
]

# ---------------------------------------------------------------------------
# Distance functions (relative RMSE; inf -> particle rejected)
# ---------------------------------------------------------------------------
def distance_live_cells(sim, obs):
    return relative_rmse(sim, obs, "live_cell_count")

def distance_dead_cells(sim, obs):
    return relative_rmse(sim, obs, "integrated_dead_cell_count")

distance_functions = {
    "live_cell_count":            {"function": distance_live_cells, "weight": 1.0},
    "integrated_dead_cell_count": {"function": distance_dead_cells, "weight": 1.0},
}

# ---------------------------------------------------------------------------
# ABC-SMC configuration (scaled up for a cluster run)
# ---------------------------------------------------------------------------
def _resolve_num_workers():
    """CLI arg (e.g. `python uq_script.py 48`) takes priority, then
    UQ_NUM_WORKERS / SLURM_CPUS_PER_TASK (see uq_slurm.sh), then all detected
    CPUs. The CLI arg is only consulted when this file is run as the entry-point
    script (__name__ == "__main__") -- not when it's imported (e.g. `import
    uq_script` from the ex11 notebook), where sys.argv belongs to the importing
    process (Jupyter's own kernel launch args, pytest's args, ...) and is not
    ours to parse.
    """
    if __name__ == "__main__" and len(sys.argv) > 1:
        return int(sys.argv[1])
    return int(os.environ.get(
        "UQ_NUM_WORKERS",
        os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()),
    ))

num_workers = _resolve_num_workers()

abc_options = {
    "models": models,
    "max_populations": 5,
    "max_simulations": 5000,
    "max_population_size": 250, # with "fixed", this is the population size
    "population_strategy": "fixed",
    "adaptive_distance": False,
    "sampler": "multicore",
    "num_workers": num_workers,
}

calib_context = CalibrationContext(
    db_path=db_path,
    obsData=obs_data_path,
    obsData_columns=obs_data_columns,
    qoi_functions=qoi_functions,
    distance_functions=distance_functions,
    abc_options=abc_options,
    logger=logger,
)


def get_model_probabilities(db_path=db_path):
    """Re-open `db_path` and return (probs, models_tbl, history):

    - probs: pyABC's per-population model probabilities (history.get_model_probabilities()),
      with columns renamed from pyABC's integer ModelIndex to each candidate's
      human-readable Name (read from the CandidateModels table this package writes).
    - models_tbl: the CandidateModels table (one row per candidate, with its
      PhysiCell config fingerprint). Named CandidateModels, not Models, because
      pyABC's own storage already has a (lowercase) `models` table in the same
      file, and SQLite resolves table names case-insensitively.
    - history: the underlying pyabc.History object (e.g. for history.max_t,
      history.get_distribution(...)).

    Works whether the calibration was run via this script, uq_slurm.sh, or the
    notebook's own (optional) calibration cell -- only db_path is read.
    """
    history = History("sqlite:///" + db_path, _id=1)
    # Extract model names from the CandidateModels table, so the returned probs DataFrame
    models_tbl = pd.read_sql(
        "SELECT ModelIndex, Name, StructureName, NumReplicates, Effective_Run_Hash "
        "FROM CandidateModels ORDER BY ModelIndex", sqlite3.connect(db_path))
    index_to_name = {int(i): n for i, n in zip(models_tbl["ModelIndex"], models_tbl["Name"])}
    # Rename the columns of the model probabilities DataFrame from pyABC's integer ModelIndex to each candidate's human-readable Name.
    probs = history.get_model_probabilities()
    # Column labels may be scalars or 1-tuples ('m',) depending on pyABC version.
    probs.columns = [index_to_name.get(int(c[-1] if isinstance(c, tuple) else c), c) for c in probs.columns]
    return probs, models_tbl, history


if __name__ == "__main__":
    logger.info(f"ABC-SMC model selection with num_workers={num_workers}")
    history = run_abc_calibration(calib_context=calib_context)
    logger.info(
        f"Completed - {history.n_populations} populations, "
        f"{history.total_nr_simulations} total simulations"
    )

    probs, models_tbl, history = get_model_probabilities()
    final = probs.iloc[-1].sort_values(ascending=False)
    winner = final.index[0]

    print("\nFinal model probabilities:")
    for name, p in final.items():
        print(f"  {name}: {p:.3f}")
    print(f"\n=> ABC-SMC favours {winner} (P = {final.iloc[0]:.3f})")
