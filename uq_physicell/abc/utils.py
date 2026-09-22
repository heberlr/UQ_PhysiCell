
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc_context import CalibrationContext
from uq_physicell import __version__ as uq_physicell_version
from pcdl import __version__ as pcdl_version
from pyabc import __version__ as pyabc_version

logger = logging.getLogger(__name__)


def patch_pyabc_dataframe_csv_fallback():
    """Make pyABC's DataFrame deserialization tolerate a database written with
    a different pyarrow availability than the one reading it now.

    pyABC's df_to_bytes/df_from_bytes (pyabc/storage/dataframe_bytes_storage.py)
    pick parquet vs. CSV purely from whether *this* process has pyarrow
    installed -- a blob is never tagged with which format it was actually
    written in. A calibration resumed on a different machine (e.g. started on
    a cluster node without pyarrow, resumed on a workstation that has it) ends
    up with a mix of CSV- and parquet-encoded summary-statistics blobs in the
    same database. Reading then assumes "pyarrow available -> must be
    parquet" and crashes with pyarrow.lib.ArrowInvalid ("Parquet magic bytes
    not found in footer") on the CSV-written rows, even though pyABC's own CSV
    reader (df_from_bytes_csv) reads them fine.

    Patches the parquet reader to fall back to the CSV reader specifically on
    a parquet-format error -- the same spirit as pyABC's own (narrower)
    legacy-msgpack fallback for pre-0.9.14 databases. Called automatically on
    `import uq_physicell.abc`; safe to call more than once. A no-op if
    pyarrow isn't installed here at all (pyABC then uses CSV exclusively
    already, so there is nothing to patch).
    """
    try:
        import pyarrow
        from pyabc.storage import dataframe_bytes_storage as _dbs
    except ImportError:
        return

    if getattr(_dbs.df_from_bytes_parquet, "_uq_physicell_csv_fallback", False):
        return  # already patched

    _orig_df_from_bytes_parquet = _dbs.df_from_bytes_parquet

    def _df_from_bytes_parquet_with_csv_fallback(bytes_):
        try:
            return _orig_df_from_bytes_parquet(bytes_)
        except (pyarrow.lib.ArrowInvalid, pyarrow.lib.ArrowIOError):
            logger.debug(
                "pyABC summary-statistics blob is not parquet (likely written "
                "in an environment without pyarrow available); falling back "
                "to pyABC's CSV reader."
            )
            return _dbs.df_from_bytes_csv(bytes_)

    _df_from_bytes_parquet_with_csv_fallback._uq_physicell_csv_fallback = True
    _dbs.df_from_bytes_parquet = _df_from_bytes_parquet_with_csv_fallback


def patch_pyabc_nan_particle_weight():
    """Make pyABC treat a non-finite particle importance weight as zero instead
    of letting it poison the whole population.

    pyABC computes each particle's weight as
    ``acceptance_weight * prior_pd / transition_pd``
    (pyabc/inference_util/inference_util.py:_weight_function). A proposed
    parameter that lands far enough into a transition kernel's tail -- or just
    outside a uniform prior's support -- can make both the prior density and
    the perturbation kernel's density underflow to exactly 0.0 in floating
    point, producing ``0.0 / 0.0 = nan``. pyABC's own normalize_weights() only
    guards against an exact total of 0.0 (``total_weight_accepted == 0.0``),
    which nan fails (``nan == 0.0`` is False in Python) -- so this one
    degenerate particle's nan weight silently poisons the entire population's
    normalization, crashing with "AssertionError: The population total weight
    nan is not normalized." only once every simulation in that population has
    already finished (observed after a 2+ day cluster run). A weight of 0.0 --
    an unmeasurably-unlikely proposal contributes nothing -- is the
    mathematically sensible outcome, so that's substituted instead.

    Called automatically on `import uq_physicell.abc`; safe to call more than
    once.
    """
    from pyabc.inference_util import inference_util as _iu

    if getattr(_iu._weight_function, "_uq_physicell_nan_guard", False):
        return  # already patched

    _orig_weight_function = _iu._weight_function

    def _weight_function_with_nan_guard(m_ss, theta_ss, acceptance_weight, prior_pdf, transition_pdf):
        try:
            weight = _orig_weight_function(m_ss, theta_ss, acceptance_weight, prior_pdf, transition_pdf)
            is_degenerate = not math.isfinite(weight)
        except ZeroDivisionError:
            # prior_pd/transition_pd as plain Python floats raise here instead of
            # silently producing nan/inf the way numpy floats do -- same outcome.
            is_degenerate = True
        if is_degenerate:
            logger.debug(
                f"pyABC importance weight was non-finite for model {m_ss}, "
                f"parameter {theta_ss} -- treating as zero weight instead of letting it "
                "propagate to the whole population's normalization."
            )
            return 0.0
        return weight

    _weight_function_with_nan_guard._uq_physicell_nan_guard = True
    _iu._weight_function = _weight_function_with_nan_guard


def insert_adaptive_weights_db(db_file, dict_distances, dict_adaptive_weights):
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create AdaptiveDistance Table
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS AdaptiveDistance (
                Population INTEGER PRIMARY KEY,
                {', '.join([f'{distance} DOUBLE' for distance in dict_distances.keys()])})""")
    conn.commit()
    conn.close()
    # Insert the Data
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    for pop_id, weights in dict_adaptive_weights.items():
        placeholders = ', '.join(['?'] * (len(dict_distances)))
        cursor.execute(f"""INSERT OR IGNORE INTO AdaptiveDistance (Population, {', '.join(dict_distances.keys())}) VALUES (?, {placeholders})""", (pop_id, *weights))
    conn.commit()
    conn.close()

def _prior_summary(prior) -> dict:
    """Best-effort JSON-serializable summary of a pyABC prior Distribution."""
    try:
        out = {}
        for pname in prior.get_parameter_names():
            rv = prior[pname]
            out[pname] = {
                "distribution": getattr(rv, "name", None),
                "args": list(getattr(rv, "args", []) or []),
                "kwargs": dict(getattr(rv, "kwargs", {}) or {}),
            }
        return out
    except Exception:
        return {"repr": repr(prior)}


def _model_config_fingerprint(spec) -> dict:
    """Best-effort PhysiCell effective-config fingerprint for one model spec.

    Returns ``{}`` (every hash resolves to None) if the model cannot be instantiated.
    """
    try:
        from uq_physicell import PhysiCell_Model
        pc_model = PhysiCell_Model(spec.model_config["ini_path"], spec.model_config["struc_name"])
        return pc_model.build_effective_config_fingerprint()
    except Exception:
        return {}


def insert_models_db(db_file: str, abc_context: "CalibrationContext"):
    """Store the candidate ABC-SMC model specs in a ``CandidateModels`` table.

    One row per model in ``abc_context.models`` (a single row for a plain
    single-model calibration). Keeps pyABC's own schema untouched -- note the
    name: pyABC's own storage backend already creates a lowercase ``models``
    table in the same database file, and SQLite resolves table names
    case-insensitively, so naming this table ``Models`` collides with it (silently
    skipping table creation, since it already "exists", then failing every insert
    with "no such column"). The per-model effective-config fingerprint hashes let
    a reader tell candidates that differ in their PhysiCell configuration
    (``struc_name``, XML, rules) apart from ones that only differ by prior / fixed
    parameters.
    """
    import json
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS CandidateModels (
                    ModelIndex INTEGER PRIMARY KEY,
                    Name TEXT,
                    Ini_File_Path TEXT,
                    StructureName TEXT,
                    InputFolder TEXT,
                    OutputFolder TEXT,
                    NumReplicates INTEGER,
                    FixedParams TEXT,
                    PriorSummary TEXT,
                    Ini_Hash TEXT,
                    XML_Hash TEXT,
                    Rules_Hash TEXT,
                    Structure_Config_Hash TEXT,
                    Effective_Run_Hash TEXT)""")
    for idx, spec in enumerate(abc_context.models):
        fp = _model_config_fingerprint(spec)
        cursor.execute(
            """INSERT OR REPLACE INTO CandidateModels
               (ModelIndex, Name, Ini_File_Path, StructureName, InputFolder, OutputFolder,
                NumReplicates, FixedParams, PriorSummary,
                Ini_Hash, XML_Hash, Rules_Hash, Structure_Config_Hash, Effective_Run_Hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                idx,
                spec.name,
                spec.model_config.get("ini_path"),
                spec.model_config.get("struc_name"),
                spec.model_config.get("input_folder"),
                spec.model_config.get("output_folder"),
                int(spec.num_replicates) if spec.num_replicates is not None else None,
                json.dumps(spec.fixed_params, default=str),
                json.dumps(_prior_summary(spec.prior), default=str),
                fp.get("ini_file_hash"),
                fp.get("xml_file_hash"),
                fp.get("rules_file_hash"),
                fp.get("structure_config_hash"),
                fp.get("effective_run_hash"),
            ),
        )
    conn.commit()
    conn.close()


def insert_metadata_db(db_file: str, abc_context: "CalibrationContext"):
    """Write the run-level ``Metadata`` row (one row, ``Method='ABC'``).

    ``Ini_File_Path`` / ``StructureName`` describe a single calibrated model. For a
    model-selection run (more than one candidate) they are written as ``NULL`` and
    the per-model configuration lives in the ``CandidateModels`` table instead.
    """
    import sqlite3
    model_selection = len(abc_context.models) > 1
    ini_path = None if model_selection else abc_context.models[0].model_config.get('ini_path')
    struc_name = None if model_selection else abc_context.models[0].model_config.get('struc_name')

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Create Metadata Table
    cursor.execute(f"""CREATE TABLE IF NOT EXISTS Metadata (
                    Method TEXT PRIMARY KEY,
                    ObsData_Path TEXT,
                    Ini_File_Path TEXT,
                    StructureName TEXT,
                    uq_physicell_version TEXT,
                    pcdl_version TEXT,
                    pyabc_version TEXT)""")
    conn.commit()
    conn.close()
    # Insert the Data
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"""INSERT OR REPLACE INTO Metadata (Method, ObsData_Path, Ini_File_Path, StructureName, uq_physicell_version, pcdl_version, pyabc_version) VALUES (?, ?, ?, ?, ?, ?, ?)""", (
        "ABC",
        abc_context.obsData_path,
        ini_path,
        struc_name,
        uq_physicell_version,
        pcdl_version,
        pyabc_version
    ))
    conn.commit()
    conn.close()