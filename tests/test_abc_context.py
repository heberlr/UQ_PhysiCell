"""
Test script for ABC CalibrationContext to verify basic functionality.
This tests the import and initialization without requiring a full PhysiCell setup.
"""

import sys
import os
import logging
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Add the path to import uq_physicell
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _minimal_kwargs(**overrides):
    """Baseline kwargs for a valid single-model CalibrationContext.

    model_config carries numReplicates directly so no ini file needs to
    exist on disk -- _resolve_num_replicates short-circuits before ever
    opening ini_path. Callers override whatever they need to exercise.
    """
    from pyabc import Distribution, RV
    kwargs = dict(
        obsData={'QoI1': np.array([1.0, 1.1])},
        obsData_columns={'QoI1': 'QoI1_data'},
        qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
        distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
        model_config={'ini_path': 'unused.ini', 'struc_name': 'strucA', 'numReplicates': 1},
        prior=Distribution(param1=RV('uniform', 0, 1.0)),
        abc_options={},
    )
    kwargs.update(overrides)
    return kwargs


def test_imports():
    """Test that the new CalibrationContext can be imported."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV
    # Basic sanity assertions: imports resolved
    assert CalibrationContext is not None
    assert Distribution is not None
    assert RV is not None

def test_initialization():
    """Test CalibrationContext initialization with minimal configuration."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Setup minimal configuration
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Synthetic observed data
    obs_data = {
        'QoI1': np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        'QoI2': np.array([0.2, 0.21, 0.22, 0.23, 0.24])
    }

    obs_data_columns = {
        'QoI1': 'QoI1_data',
        'QoI2': 'QoI2_data'
    }

    # Create a temporary INI file for testing
    ini_content = """[model_struc_name]
    numReplicates = 1
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_file_path = ini_file.name

    model_config = {
        'ini_path': ini_file_path,
        'struc_name': 'model_struc_name',
    }

    qoi_functions = {
        'QoI1': 'lambda df: df["QoI1"].values',
        'QoI2': 'lambda df: df["QoI2"].values'
    }

    distance_functions = {
        'QoI1': {'function': 'euclidean', 'weight': 1.0},
        'QoI2': {'function': 'euclidean', 'weight': 1.0}
    }

    lb1 = 0.0; ub1 = 5.0
    lb2 = 0.0; ub2 = 10.0; loc2 = 5.0; scale2 = 2.0 # lb and ub are bounds, loc is mean, scale is stddev
    prior = Distribution(
        param1 = RV('uniform', lb1, ub1-lb1),
        param2 = RV('truncnorm', lb2, ub2, loc2, scale2)
    )

    abc_options = {
        'max_populations': 5,
        'max_simulations': 50,
        'sampler': 'multicore',
        'num_workers': 2,
        'mode': 'local'
    }

    # Create temporary database file
    db_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name

        # Initialize CalibrationContext
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData=obs_data,
            obsData_columns=obs_data_columns,
            model_config=model_config,
            qoi_functions=qoi_functions,
            distance_functions=distance_functions,
            prior=prior,
            abc_options=abc_options,
            logger=logger
        )

        # Sanity-check a few attributes
        assert hasattr(calib_context, 'db_path')
        assert list(calib_context.qoi_functions.keys()) == ['QoI1', 'QoI2']
        # prior is a pyabc Distribution — ensure parameter names property exists if available
        if hasattr(calib_context.prior, 'get_parameter_names'):
            assert callable(calib_context.prior.get_parameter_names)
        assert hasattr(calib_context, 'sampler_type')
        assert hasattr(calib_context, 'num_workers')

    finally:
        # Cleanup
        if db_path and os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except Exception:
                pass
        if ini_file_path and os.path.exists(ini_file_path):
            try:
                os.unlink(ini_file_path)
            except Exception:
                pass

def test_configuration_validation():
    """Test that configuration validation works correctly."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    # Test with missing required keys
    invalid_model_config = {
        'numReplicates': 2
        # Missing 'config_file' and 'model_name'
    }

    with pytest.raises(ValueError):
        CalibrationContext(
            db_path="dummy.db",
            obsData={'dummy': [1, 2, 3]},
            obsData_columns={'dummy': 'dummy'},
            model_config=invalid_model_config,  # Invalid config
            qoi_functions={'dummy': 'lambda x: x'},
            distance_functions={'dummy': {'function': 'euclidean', 'weight': 1.0}},
            prior= Distribution(param1 = RV('uniform', 0, 1.0)),
            abc_options={'max_populations': 5}
        )

def test_model_selection_specs():
    """abc_options['models'] builds one ModelSpec per candidate with per-model IO subfolders."""
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 4\n\n[strucB]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name

    obs_data = {'QoI1': np.array([1.0, 1.1, 1.2])}
    obs_data_columns = {'QoI1': 'QoI1_data'}
    qoi_functions = {'QoI1': 'lambda df: df["QoI1"].values'}
    distance_functions = {'QoI1': {'function': 'euclidean', 'weight': 1.0}}
    prior_a = Distribution(param1=RV('uniform', 0, 1.0))
    prior_b = Distribution(param1=RV('uniform', 0, 1.0), param2=RV('uniform', 0, 2.0))

    try:
        ctx = CalibrationContext(
            db_path="dummy.db",
            obsData=obs_data,
            obsData_columns=obs_data_columns,
            qoi_functions=qoi_functions,
            distance_functions=distance_functions,
            abc_options={
                'max_populations': 2,
                'sampler': 'multicore',
                'num_workers': 2,
                'models': [
                    {'name': 'hypA', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucA'},
                     'prior': prior_a},
                    {'name': 'hypB', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucB'},
                     'prior': prior_b, 'fixed_params': {'k': 1.0}, 'num_replicates': 3},
                ],
            },
        )

        assert ctx.model_selection is True
        assert ctx.num_models == 2
        assert [m.name for m in ctx.models] == ['hypA', 'hypB']
        assert isinstance(ctx.models[0], ModelSpec)
        # replicate counts: from ini / explicit override
        assert ctx.models[0].num_replicates == 4
        assert ctx.models[1].num_replicates == 3
        # per-model IO isolation auto-applied for multi-model runs
        assert ctx.models[0].model_config['output_folder'] == 'hypA/'
        assert ctx.models[1].model_config['input_folder'] == 'hypB/'
        # back-compat accessors delegate to models[0]
        assert ctx.prior is ctx.models[0].prior
        assert ctx.fixed_params == {}
        # one pyABC wrapper per model, uniquely named
        names = {ctx.create_model_wrapper(m, ctx.workers_inner).__name__ for m in ctx.models}
        assert names == {'run_physicell_hypA', 'run_physicell_hypB'}
    finally:
        if os.path.exists(ini_path):
            os.unlink(ini_path)


def test_model_selection_validation():
    """Invalid model-selection specs raise ValueError."""
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    common = dict(
        db_path="dummy.db",
        obsData={'QoI1': [1, 2, 3]},
        obsData_columns={'QoI1': 'QoI1'},
        qoi_functions={'QoI1': 'lambda x: x'},
        distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
    )
    prior = Distribution(param1=RV('uniform', 0, 1.0))
    mc = {'ini_path': 'x.ini', 'struc_name': 's', 'numReplicates': 1}

    # neither models nor model_config/prior
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={}, **common)
    # model entry missing required 'prior'
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [{'name': 'a', 'model_config': mc}]}, **common)
    # non-unique names
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [
            {'name': 'a', 'model_config': mc, 'prior': prior},
            {'name': 'a', 'model_config': mc, 'prior': prior},
        ]}, **common)
    # unsafe name
    with pytest.raises(ValueError):
        CalibrationContext(abc_options={'models': [
            {'name': 'a/b', 'model_config': mc, 'prior': prior},
        ]}, **common)


def test_model_selection_db_tables():
    """Metadata NULLs the single-model columns for a selection run; CandidateModels has one row per candidate."""
    import sqlite3
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.utils import insert_metadata_db, insert_models_db
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 2\n\n[strucB]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name

    common = dict(
        obsData={'QoI1': np.array([1.0, 1.1])},
        obsData_columns={'QoI1': 'QoI1_data'},
        qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
        distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
    )
    prior = Distribution(param1=RV('uniform', 0, 1.0))

    try:
        # single model -> Metadata carries ini/struc
        ctx1 = CalibrationContext(db_path=db_path, model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
                                  prior=prior, abc_options={}, **common)
        insert_metadata_db(db_path, ctx1)
        insert_models_db(db_path, ctx1)
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT Ini_File_Path, StructureName FROM Metadata").fetchone() == (ini_path, 'strucA')
        assert conn.execute("SELECT COUNT(*) FROM CandidateModels").fetchone()[0] == 1
        conn.close()

        # model selection -> Metadata NULLs ini/struc, CandidateModels has both rows
        ctx2 = CalibrationContext(db_path=db_path, abc_options={'models': [
            {'name': 'A', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucA'}, 'prior': prior},
            {'name': 'B', 'model_config': {'ini_path': ini_path, 'struc_name': 'strucB'}, 'prior': prior},
        ]}, **common)
        insert_metadata_db(db_path, ctx2)
        insert_models_db(db_path, ctx2)
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT Ini_File_Path, StructureName FROM Metadata").fetchone() == (None, None)
        rows = conn.execute("SELECT ModelIndex, Name, StructureName FROM CandidateModels ORDER BY ModelIndex").fetchall()
        assert rows == [(0, 'A', 'strucA'), (1, 'B', 'strucB')]
        # fingerprint columns exist (values may be NULL when the model can't be instantiated)
        cols = {c[1] for c in conn.execute("PRAGMA table_info(CandidateModels)")}
        assert {'Ini_Hash', 'XML_Hash', 'Rules_Hash', 'Structure_Config_Hash', 'Effective_Run_Hash'} <= cols
        conn.close()
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def test_insert_models_db_does_not_collide_with_pyabc_models_table():
    """Regression test: pyABC's own storage creates a lowercase `models` table in
    the same database file. SQLite resolves table names case-insensitively, so a
    table literally named `Models` would silently fail to be created ("IF NOT
    EXISTS" sees pyABC's own table) and every insert would then raise
    "no such column: ModelIndex" against pyABC's table instead. insert_models_db
    must use a non-colliding name (CandidateModels) and leave pyABC's table alone.
    """
    import sqlite3
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.utils import insert_models_db
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name

    try:
        # Stand in for pyABC's own schema, already present by the time
        # insert_models_db runs in a real calibration (abc_smc.new() creates it
        # before run_abc_calibration ever calls insert_models_db).
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE models (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO models VALUES (0, 'pyabc_internal_name')")
        conn.commit()
        conn.close()

        prior = Distribution(param1=RV('uniform', 0, 1.0))
        ctx = CalibrationContext(
            db_path=db_path,
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior, abc_options={},
            obsData={'QoI1': np.array([1.0, 1.1])},
            obsData_columns={'QoI1': 'QoI1_data'},
            qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
            distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
        )
        insert_models_db(db_path, ctx)  # must not raise

        conn = sqlite3.connect(db_path)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {'models', 'CandidateModels'} <= tables
        # pyABC's own table must be untouched.
        assert conn.execute("SELECT id, name FROM models").fetchall() == [(0, 'pyabc_internal_name')]
        assert conn.execute("SELECT ModelIndex, Name FROM CandidateModels").fetchall() == [(0, 'Model_0')]
        conn.close()
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def test_metadata_persisted_before_calibration_runs():
    """Regression test: Metadata/CandidateModels are pure static configuration
    and must be persisted before run_calibration() executes, not after.

    run_calibration() is a single, blocking abc_smc.run(max_nr_populations=...)
    call that does not return until pyABC has completed every requested
    population (or hit the simulation cap) -- for a real, multi-hour/day run
    with no convergence_check_func configured (the common case, e.g. ex11's
    uq_script.py), that is the entire calibration. Persisting only after it
    returns meant a crash, timeout, or kill anywhere during that call -- the
    likely place for exactly that on a long run -- left pyABC's own History
    file fully populated but Metadata/CandidateModels completely empty.
    """
    import sqlite3
    from unittest.mock import patch
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.abc_context import run_abc_calibration
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 2\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    os.remove(db_path)  # must not exist yet -- abc_smc.new() is what creates it

    try:
        prior = Distribution(param1=RV('uniform', 0, 1.0))
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData={'QoI1': np.array([1.0, 1.1])},
            obsData_columns={'QoI1': 'QoI1_data'},
            qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
            distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior,
            abc_options={},
        )

        # Simulate a crash inside the (normally multi-hour) calibration call.
        with patch.object(CalibrationContext, "run_calibration", side_effect=RuntimeError("simulated crash mid-run")):
            with pytest.raises(RuntimeError, match="simulated crash mid-run"):
                run_abc_calibration(calib_context=calib_context)

        assert os.path.exists(db_path), "abc_smc.new() should have created the database before the simulated crash"
        conn = sqlite3.connect(db_path)
        assert conn.execute("SELECT Ini_File_Path, StructureName FROM Metadata").fetchone() == (ini_path, 'strucA')
        assert conn.execute("SELECT COUNT(*) FROM CandidateModels").fetchone()[0] == 1
        conn.close()
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def test_sequential_replicate_aggregation_and_seeds():
    """Regression test for two bugs in _run_physicell_model_sequential:

    1. The replicate loop reused the `replicate_id` parameter as its loop
       variable, shadowing it -- so `if replicate_id is None:` was never true
       after the loop ran, and aggregation_func was silently skipped whenever
       this is called with replicate_id=None (i.e. the non-multicore / "dask"
       sampler path).
    2. Replicates were run without a distinct random_seed per replicate, so
       parallel replicates of the same particle could collide on PhysiCell's
       default system-clock seed.
    """
    from uq_physicell.abc import CalibrationContext
    from uq_physicell.abc.abc_context import ModelSpec
    from pyabc import Distribution, RV

    ini_content = "[strucA]\nnumReplicates = 3\n"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as ini_file:
        ini_file.write(ini_content)
        ini_path = ini_file.name
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    prior = Distribution(param1=RV('uniform', 0, 1.0))

    try:
        calib_context = CalibrationContext(
            db_path=db_path,
            obsData={'QoI1': np.array([1.0, 1.1])},
            obsData_columns={'QoI1': 'QoI1_data'},
            qoi_functions={'QoI1': 'lambda df: df["QoI1"].values'},
            distance_functions={'QoI1': {'function': 'euclidean', 'weight': 1.0}},
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior,
            abc_options={},
        )
        model_spec = ModelSpec(
            name='strucA',
            model_config={'ini_path': ini_path, 'struc_name': 'strucA'},
            prior=prior,
            num_replicates=3,
        )

        fake_model = MagicMock()
        fake_model.XML_parameters_variable = {}
        fake_model.parameters_rules_variable = {}

        aggregation_calls = []
        def fake_aggregation_func(dic_all_replicates):
            aggregation_calls.append(dic_all_replicates)
            return "AGGREGATED"

        seen_seeds = []
        def fake_run_replicate_serializable(**kwargs):
            seen_seeds.append(kwargs["random_seed"])
            return (None, None, pd.DataFrame({"sampleID": [1], "time": [0]}))

        with patch.object(calib_context, "_instantiate_model", return_value=fake_model), \
             patch.object(calib_context, "aggregation_func", side_effect=fake_aggregation_func), \
             patch("uq_physicell.abc.abc_context.run_replicate_serializable", side_effect=fake_run_replicate_serializable):
            result = calib_context._run_physicell_model_sequential(
                pars={}, model_spec=model_spec, sample_id=1, replicate_id=None
            )

        assert aggregation_calls, "aggregation_func was never called -- replicate_id shadowing regression"
        assert result == "AGGREGATED"

        assert len(seen_seeds) == 3
        assert len(set(seen_seeds)) == 3, "replicates must not share a random seed"
    finally:
        for p in (ini_path, db_path):
            if os.path.exists(p):
                os.unlink(p)


def test_missing_qoi_or_distance_functions_raises():
    """qoi_functions/distance_functions=None must raise, not silently proceed."""
    from uq_physicell.abc import CalibrationContext

    with pytest.raises(ValueError, match="qoi_functions and distance_functions are required"):
        CalibrationContext(
            db_path="dummy.db",
            obsData={'QoI1': [1, 2]},
            obsData_columns={'QoI1': 'QoI1'},
            qoi_functions=None,
            distance_functions=None,
        )


def test_obsdata_loaded_from_csv_path(tmp_path):
    """obsData as a CSV path: columns renamed per obsData_columns and dropped."""
    from uq_physicell.abc import CalibrationContext

    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"t": [0, 1, 2], "y_obs": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)

    ctx = CalibrationContext(
        db_path="dummy.db",
        **_minimal_kwargs(obsData=str(csv_path), obsData_columns={'QoI1': 'y_obs'}),
    )
    assert ctx.obsData_path == str(csv_path)
    np.testing.assert_allclose(ctx.dic_obsData['QoI1'], [1.0, 2.0, 3.0])
    assert 'y_obs' not in ctx.dic_obsData


def test_obsdata_csv_missing_column_raises(tmp_path):
    from uq_physicell.abc import CalibrationContext

    csv_path = tmp_path / "obs.csv"
    pd.DataFrame({"t": [0, 1]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="not found in observed data"):
        CalibrationContext(
            db_path="dummy.db",
            **_minimal_kwargs(obsData=str(csv_path), obsData_columns={'QoI1': 'missing_col'}),
        )


def test_obsdata_csv_bad_path_raises(tmp_path):
    """A nonexistent CSV path re-raises the underlying exception (logged first)."""
    from uq_physicell.abc import CalibrationContext

    with pytest.raises(FileNotFoundError):
        CalibrationContext(
            db_path="dummy.db",
            **_minimal_kwargs(obsData=str(tmp_path / "does_not_exist.csv"),
                               obsData_columns={'QoI1': 'y'}),
        )


def test_property_setters_delegate_to_models0():
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    new_mc = {'ini_path': 'unused.ini', 'struc_name': 'strucA', 'numReplicates': 1, 'tag': 'x'}
    ctx.model_config = new_mc
    assert ctx.models[0].model_config is new_mc
    assert ctx.model_config is new_mc  # getter delegates too

    new_prior = Distribution(param1=RV('uniform', 0, 2.0))
    ctx.prior = new_prior
    assert ctx.models[0].prior is new_prior
    assert ctx.prior is new_prior

    ctx.fixed_params = {'k': 1.0}
    assert ctx.models[0].fixed_params == {'k': 1.0}
    assert ctx.fixed_params == {'k': 1.0}


def test_deprecated_and_conflicting_abc_options_warn(caplog):
    """abc_options['num_models']/['model_selection'] are deprecated; specifying
    both abc_options['models'] and top-level model_config/prior warns that the
    top-level ones are ignored.
    """
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    prior = Distribution(param1=RV('uniform', 0, 1.0))
    mc = {'ini_path': 'unused.ini', 'struc_name': 's', 'numReplicates': 1}

    with caplog.at_level(logging.WARNING):
        CalibrationContext(
            db_path="dummy.db",
            model_config=mc,
            prior=prior,
            abc_options={'num_models': 2, 'models': [{'name': 'a', 'model_config': mc, 'prior': prior}]},
            **{k: v for k, v in _minimal_kwargs().items() if k not in ('model_config', 'prior', 'abc_options')},
        )
    messages = [r.message for r in caplog.records]
    assert any("deprecated" in m for m in messages)
    assert any("top-level model_config/prior are ignored" in m for m in messages)


def test_num_replicates_resolved_from_model_config_key():
    """numReplicates read straight from model_config, bypassing the ini file entirely."""
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(
        db_path="dummy.db",
        **_minimal_kwargs(model_config={'ini_path': 'unused.ini', 'struc_name': 'irrelevant', 'numReplicates': 7}),
    )
    assert ctx.models[0].num_replicates == 7


def test_instantiate_model_applies_io_folders():
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    spec = ModelSpec(
        name='m',
        model_config={'ini_path': 'x.ini', 'struc_name': 's', 'input_folder': 'in/', 'output_folder': 'out/'},
        prior=Distribution(param1=RV('uniform', 0, 1.0)),
        num_replicates=1,
    )
    fake_pc_model = MagicMock()
    fake_pc_model.input_folder = "base_in/"
    fake_pc_model.output_folder = "base_out/"

    with patch("uq_physicell.abc.abc_context.PhysiCell_Model", return_value=fake_pc_model) as mock_ctor:
        result = ctx._instantiate_model(spec)

    mock_ctor.assert_called_once_with('x.ini', 's')
    assert result.input_folder == "base_in/in/"
    assert result.output_folder == "base_out/out/"


def test_setup_parallelization_dask():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(abc_options={'sampler': 'dask', 'num_workers': 4}))
    assert ctx.workers_inner is None
    assert ctx.workers_outer == 4


def test_setup_parallelization_unsupported_sampler_raises():
    from uq_physicell.abc import CalibrationContext

    with pytest.raises(ValueError, match="not supported"):
        CalibrationContext(db_path="dummy.db", **_minimal_kwargs(abc_options={'sampler': 'bogus'}))


def test_validate_configuration_errors():
    from uq_physicell.abc import CalibrationContext
    from pyabc import Distribution, RV

    with pytest.raises(ValueError, match="prior cannot be empty"):
        CalibrationContext(db_path="dummy.db", **_minimal_kwargs(prior=Distribution()))

    with pytest.raises(ValueError, match="qoi_functions cannot be empty"):
        CalibrationContext(db_path="dummy.db", **_minimal_kwargs(qoi_functions={}))

    with pytest.raises(ValueError, match="distance_functions cannot be empty"):
        CalibrationContext(db_path="dummy.db", **_minimal_kwargs(distance_functions={}))

    with pytest.raises(ValueError, match="Distance function not defined for QoI"):
        CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
            distance_functions={'other': {'function': 'euclidean', 'weight': 1.0}}
        ))


def test_setup_sampler_dask_missing_cluster_func_raises():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(abc_options={'sampler': 'dask', 'num_workers': 2}))
    with pytest.raises(ValueError, match="cluster_setup_func must be provided"):
        ctx.setup_sampler()


def test_setup_sampler_dask_success():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(abc_options={'sampler': 'dask', 'num_workers': 2}))

    with patch("uq_physicell.abc.abc_context.Client") as mock_client_cls, \
         patch("uq_physicell.abc.abc_context.sampler.DaskDistributedSampler") as mock_dask_sampler_cls:
        mock_client_cls.return_value = "the_client"
        mock_dask_sampler_cls.return_value = "the_sampler"
        result = ctx.setup_sampler(cluster_setup_func=lambda: "fake_cluster")

    mock_client_cls.assert_called_once_with("fake_cluster")
    mock_dask_sampler_cls.assert_called_once_with("the_client")
    assert result == "the_sampler"


def test_setup_sampler_unsupported_raises():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())
    ctx.sampler_type = "bogus"
    with pytest.raises(ValueError, match="is not supported"):
        ctx.setup_sampler()


def test_setup_population_strategy_fixed():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'population_strategy': 'fixed', 'max_population_size': 250}
    ))
    assert ctx.setup_population_strategy() == 250


def test_setup_distance_function_multi_qoi_branches():
    from uq_physicell.abc import CalibrationContext
    from pyabc import AggregatedDistance, AdaptiveAggregatedDistance

    two_qoi = dict(
        qoi_functions={'QoI1': 'lambda df: df["QoI1"].values', 'QoI2': 'lambda df: df["QoI2"].values'},
        distance_functions={
            'QoI1': {'function': 'euclidean', 'weight': 1.0},
            'QoI2': {'function': 'euclidean', 'weight': 2.0},
        },
    )

    # Fixed (non-adaptive) weighting.
    ctx_fixed = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        **two_qoi, abc_options={'adaptive_distance': False}
    ))
    dist_fixed = ctx_fixed.setup_distance_function(ctx_fixed.distance_functions)
    assert isinstance(dist_fixed, AggregatedDistance)

    # Adaptive weighting (db doesn't exist -> fresh AdaptiveAggregatedDistance).
    ctx_adaptive = CalibrationContext(db_path="dummy_nonexistent.db", **_minimal_kwargs(
        **two_qoi, abc_options={'adaptive_distance': True}
    ))
    dist_adaptive = ctx_adaptive.setup_distance_function(ctx_adaptive.distance_functions)
    assert isinstance(dist_adaptive, AdaptiveAggregatedDistance)


def test_setup_transition_function_local():
    from uq_physicell.abc import CalibrationContext
    from pyabc import LocalTransition

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'transition_strategy': 'local'}
    ))
    assert isinstance(ctx.setup_transition_function(), LocalTransition)


def test_setup_epsilon_function_unsupported_raises():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'epsilon_strategy': 'bogus'}
    ))
    with pytest.raises(ValueError, match="not supported"):
        ctx.setup_epsilon_function()


def test_create_model_wrapper_invokes_run_physicell_model():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())
    spec = ctx.models[0]
    wrapper = ctx.create_model_wrapper(spec, ctx.workers_inner)

    with patch.object(ctx, "_run_physicell_model", return_value={'foo': 1}) as mock_run:
        result = wrapper({'p': 1})

    mock_run.assert_called_once_with({'p': 1}, spec, ctx.workers_inner)
    assert result == {'foo': 1}


def test_run_physicell_model_dispatch_and_error_handling():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'sampler': 'multicore', 'num_workers': 2}
    ))
    spec = ctx.models[0]

    with patch.object(ctx, "_run_replicates_parallel", return_value="PARALLEL") as mock_par, \
         patch.object(ctx, "_run_physicell_model_sequential") as mock_seq:
        result = ctx._run_physicell_model(pars={'a': 1}, model_spec=spec, workers_inner=2)
    mock_par.assert_called_once_with(2, {'a': 1}, spec)
    mock_seq.assert_not_called()
    assert result == "PARALLEL"

    with patch.object(ctx, "_run_replicates_parallel") as mock_par2, \
         patch.object(ctx, "_run_physicell_model_sequential", return_value="SEQ") as mock_seq2:
        result2 = ctx._run_physicell_model(pars={'a': 1}, model_spec=spec, workers_inner=None)
    mock_par2.assert_not_called()
    mock_seq2.assert_called_once_with({'a': 1}, spec)
    assert result2 == "SEQ"

    with patch.object(ctx, "_run_physicell_model_sequential", side_effect=RuntimeError("boom")):
        result3 = ctx._run_physicell_model(pars={'a': 1}, model_spec=spec, workers_inner=None)
    assert result3 is None


def test_default_aggregation_func_success_and_failures():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    replicate_results = {
        0: pd.DataFrame({'sampleID': [1, 1], 'time': [0, 1], 'val': [10.0, 20.0]}),
        1: pd.DataFrame({'sampleID': [1, 1], 'time': [0, 1], 'val': [12.0, 22.0]}),
    }
    result = ctx._default_aggregation_func(replicate_results)
    assert list(result['val']) == [11.0, 21.0]

    # Empty dict: the error handler must not itself crash with an unrelated
    # TypeError/IndexError from indexing a dict_values view (see abc_context.py
    # fix in _default_aggregation_func).
    with pytest.raises(ValueError, match="unknown \\(no replicates\\)"):
        ctx._default_aggregation_func({})

    # Non-empty but malformed (missing 'time' -> pivot_table KeyError), sampleID
    # still extractable for the error message.
    malformed = {0: pd.DataFrame({'sampleID': [1], 'val': [1.0]})}
    with pytest.raises(ValueError, match="Error in _default_aggregation_func for sampleID"):
        ctx._default_aggregation_func(malformed)


def test_run_physicell_model_sequential_fixed_param_override_and_raw_replicate_return():
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    fake_model = MagicMock()
    fake_model.XML_parameters_variable = {'x': 'paramA'}
    fake_model.parameters_rules_variable = {'r': 'paramB'}

    calls = []
    def fake_run_replicate(**kwargs):
        calls.append(kwargs)
        return (None, None, pd.DataFrame({'sampleID': [1], 'time': [0]}))

    model_spec = ModelSpec(
        name='m', model_config={'ini_path': 'x.ini', 'struc_name': 's'},
        prior=Distribution(param1=RV('uniform', 0, 1.0)),
        fixed_params={'paramA': 42.0, 'paramB': 7.0}, num_replicates=1,
    )

    with patch.object(ctx, "_instantiate_model", return_value=fake_model), \
         patch.object(ctx, "_get_worker_id", return_value=77) as mock_worker_id, \
         patch("uq_physicell.abc.abc_context.run_replicate_serializable", side_effect=fake_run_replicate):
        result = ctx._run_physicell_model_sequential(pars={}, model_spec=model_spec, sample_id=None, replicate_id=0)

    mock_worker_id.assert_called_once()
    assert calls[0]["sample_id"] == 77
    assert calls[0]["ParametersXML"] == {'paramA': 42.0}
    assert calls[0]["ParametersRules"] == {'paramB': 7.0}
    # replicate_id given (not None) -> raw dict returned, no aggregation.
    assert list(result.keys()) == [0]


def test_run_physicell_model_sequential_missing_param_raises():
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())
    fake_model = MagicMock()
    fake_model.XML_parameters_variable = {'x': 'paramA'}
    fake_model.parameters_rules_variable = {}
    model_spec = ModelSpec(
        name='m', model_config={'ini_path': 'x.ini', 'struc_name': 's'},
        prior=Distribution(param1=RV('uniform', 0, 1.0)), num_replicates=1,
    )

    with patch.object(ctx, "_instantiate_model", return_value=fake_model):
        with pytest.raises(ValueError, match="Some parameters are None"):
            ctx._run_physicell_model_sequential(pars={}, model_spec=model_spec, sample_id=1, replicate_id=0)


def test_run_physicell_model_sequential_run_replicate_errors():
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())
    fake_model = MagicMock()
    fake_model.XML_parameters_variable = {}
    fake_model.parameters_rules_variable = {}
    model_spec = ModelSpec(
        name='m', model_config={'ini_path': 'x.ini', 'struc_name': 's'},
        prior=Distribution(param1=RV('uniform', 0, 1.0)), num_replicates=1,
    )

    with patch.object(ctx, "_instantiate_model", return_value=fake_model), \
         patch("uq_physicell.abc.abc_context.run_replicate_serializable",
               side_effect=RuntimeError("physicell crashed")):
        with pytest.raises(RuntimeError, match="Error in RunModel"):
            ctx._run_physicell_model_sequential(pars={}, model_spec=model_spec, sample_id=1, replicate_id=0)

    with patch.object(ctx, "_instantiate_model", return_value=fake_model), \
         patch("uq_physicell.abc.abc_context.run_replicate_serializable",
               return_value=(None, None, pd.DataFrame())):
        with pytest.raises(RuntimeError, match="empty or invalid DataFrame"):
            ctx._run_physicell_model_sequential(pars={}, model_spec=model_spec, sample_id=1, replicate_id=0)


def test_run_replicates_parallel_merges_replicate_dicts():
    from uq_physicell.abc import CalibrationContext, ModelSpec
    from pyabc import Distribution, RV

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'sampler': 'multicore', 'num_workers': 2}
    ))
    model_spec = ModelSpec(
        name='m', model_config={'ini_path': 'x.ini', 'struc_name': 's'},
        prior=Distribution(param1=RV('uniform', 0, 1.0)), num_replicates=3,
    )

    def fake_seq(params, spec, worker_id, replicate_id):
        return {replicate_id: f"data{replicate_id}"}

    agg_calls = []
    def fake_agg(dic_all):
        agg_calls.append(dict(dic_all))
        return "AGG_RESULT"

    with patch.object(ctx, "_run_physicell_model_sequential", side_effect=fake_seq), \
         patch.object(ctx, "_get_worker_id", return_value=5), \
         patch.object(ctx, "aggregation_func", side_effect=fake_agg):
        result = ctx._run_replicates_parallel(workers_inner=2, params={'p': 1}, model_spec=model_spec)

    assert result == "AGG_RESULT"
    assert agg_calls == [{0: "data0", 1: "data1", 2: "data2"}]


def test_get_worker_id_variants():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    numeric_worker = MagicMock()
    numeric_worker.name = "3"
    with patch("uq_physicell.abc.abc_context.get_worker", return_value=numeric_worker):
        assert ctx._get_worker_id() == 3

    hyphenated_worker = MagicMock()
    hyphenated_worker.name = "Worker-7"
    with patch("uq_physicell.abc.abc_context.get_worker", return_value=hyphenated_worker):
        assert ctx._get_worker_id() == 7

    with patch("uq_physicell.abc.abc_context.get_worker", side_effect=ValueError("no worker")):
        assert ctx._get_worker_id() == os.getpid()


def test_load_or_create_database_branches(tmp_path):
    from uq_physicell.abc import CalibrationContext

    db_path = tmp_path / "cal.db"
    ctx = CalibrationContext(db_path=str(db_path), **_minimal_kwargs())

    fake_abc_smc = MagicMock()
    resumed, n_pop, n_sim = ctx.load_or_create_database(fake_abc_smc)
    assert (resumed, n_pop, n_sim) == (False, 0, 0)
    fake_abc_smc.new.assert_called_once_with("sqlite:///" + str(db_path), observed_sum_stat=ctx.dic_obsData)

    db_path.write_text("")  # simulate an existing database file
    fake_abc_smc2 = MagicMock()
    fake_abc_smc2.history.n_populations = 4
    fake_abc_smc2.history.total_nr_simulations = 99
    resumed2, n_pop2, n_sim2 = ctx.load_or_create_database(fake_abc_smc2, abc_id=2)
    assert (resumed2, n_pop2, n_sim2) == (True, 4, 99)
    fake_abc_smc2.load.assert_called_once_with("sqlite:///" + str(db_path), abc_id=2)

    fake_abc_smc3 = MagicMock()
    fake_abc_smc3.load.side_effect = ValueError("corrupt db")
    with pytest.raises(ValueError, match="corrupt db"):
        ctx.load_or_create_database(fake_abc_smc3)


def test_run_calibration_branches():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'max_populations': 10, 'max_simulations': 100}
    ))

    fake_abc_smc = MagicMock()
    ctx.run_calibration(fake_abc_smc, resume_db=True, current_populations=2, current_simulations=20)
    fake_abc_smc.run.assert_called_once_with(max_nr_populations=10, max_total_nr_simulations=100)

    fake_abc_smc2 = MagicMock()
    ctx.run_calibration(fake_abc_smc2, resume_db=True, current_populations=10, current_simulations=100)
    fake_abc_smc2.run.assert_not_called()

    ctx.adaptive_distance = True
    ctx.adaptive_distance_file = "fake_log.json"
    fake_abc_smc3 = MagicMock()
    with patch("uq_physicell.abc.abc_context.insert_adaptive_weights_db") as mock_insert, \
         patch("uq_physicell.abc.abc_context.load_dict_from_json", return_value={"QoI1": 1.0}) as mock_load:
        ctx.run_calibration(fake_abc_smc3, resume_db=False)
    fake_abc_smc3.run.assert_called_once_with(max_nr_populations=10, max_total_nr_simulations=100)
    mock_load.assert_called_once_with("fake_log.json")
    mock_insert.assert_called_once_with(
        ctx.db_path, dict_distances=ctx.distance_functions, dict_adaptive_weights={"QoI1": 1.0}
    )


def test_check_convergence_and_include_additional_metadata():
    from uq_physicell.abc import CalibrationContext

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())
    assert ctx.check_convergence(MagicMock()) is False
    ctx.include_additional_metadata(foo="bar", n=1)  # must not raise


def test_run_abc_calibration_full_flow_with_convergence_and_model_selection(tmp_path):
    """Integration-style test of run_abc_calibration's post-run_calibration flow:
    model-selection logging, the convergence retry loop, final adaptive-weight
    persistence, and per-model IO cleanup -- none of which are exercised by
    test_metadata_persisted_before_calibration_runs (which crashes exactly at
    run_calibration).
    """
    from uq_physicell.abc import CalibrationContext, run_abc_calibration
    from pyabc import Distribution, RV

    prior = Distribution(param1=RV('uniform', 0, 1.0))
    mc_a = {'ini_path': 'unused.ini', 'struc_name': 'strucA', 'numReplicates': 1}
    mc_b = {'ini_path': 'unused.ini', 'struc_name': 'strucB', 'numReplicates': 1}

    convergence_calls = {"n": 0}
    def convergence_check_func(history):
        convergence_calls["n"] += 1
        return convergence_calls["n"] >= 2

    adaptive_file = tmp_path / "adaptive_log.json"
    adaptive_file.write_text("{}")

    ctx = CalibrationContext(
        db_path="dummy.db",
        **_minimal_kwargs(
            model_config=None, prior=None,
            abc_options={
                'models': [
                    {'name': 'A', 'model_config': mc_a, 'prior': prior},
                    {'name': 'B', 'model_config': mc_b, 'prior': prior},
                ],
                'convergence_check_func': convergence_check_func,
                'adaptive_distance': True,
                'adaptive_distance_file': str(adaptive_file),
                'max_populations': 1,
                'max_simulations': 5,
            },
        ),
    )

    # Both >= the initial max_populations/max_simulations so the loop's
    # "extend limits by one" branches (not just the retry-run itself) fire.
    fake_history = MagicMock()
    fake_history.n_populations = 2
    fake_history.total_nr_simulations = 10
    fake_history.get_model_probabilities.return_value = "PROBS"

    fake_abc_smc = MagicMock()
    fake_abc_smc.history = fake_history

    fake_pc_model = MagicMock()

    with patch.object(ctx, "setup_sampler", return_value=MagicMock()), \
         patch.object(ctx, "setup_population_strategy", return_value=100), \
         patch.object(ctx, "setup_distance_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_transition_function", return_value=None), \
         patch.object(ctx, "setup_epsilon_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_abc_smc", return_value=fake_abc_smc), \
         patch.object(ctx, "load_or_create_database", return_value=(False, 0, 0)), \
         patch.object(ctx, "run_calibration") as mock_run_calibration, \
         patch.object(ctx, "_instantiate_model", return_value=fake_pc_model), \
         patch("uq_physicell.abc.abc_context.insert_metadata_db") as mock_insert_meta, \
         patch("uq_physicell.abc.abc_context.insert_models_db") as mock_insert_models, \
         patch("uq_physicell.abc.abc_context.insert_adaptive_weights_db") as mock_insert_adaptive, \
         patch("uq_physicell.abc.abc_context.load_dict_from_json", return_value={"QoI1": 1.0}):

        result = run_abc_calibration(calib_context=ctx)

    assert result is fake_history
    mock_insert_meta.assert_called_once()
    mock_insert_models.assert_called_once()
    mock_run_calibration.assert_called_once()
    # Loop ran once (not converged), called abc_smc.run() directly, then converged.
    assert convergence_calls["n"] == 2
    # max_populations/max_simulations were extended by one past history's values.
    fake_abc_smc.run.assert_called_once_with(max_nr_populations=3, max_total_nr_simulations=11)
    fake_history.get_model_probabilities.assert_called_once()
    # Adaptive weights persisted once inside the loop iteration, once more at the end.
    assert mock_insert_adaptive.call_count == 2
    assert fake_pc_model.remove_io_folders.call_count == 2  # one per candidate model


def test_run_abc_calibration_metadata_failure_logs_warning_and_continues(caplog):
    """A failure persisting Metadata/CandidateModels must not abort the run --
    see the comment above insert_metadata_db's call site in run_abc_calibration.
    """
    from uq_physicell.abc import CalibrationContext, run_abc_calibration

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs())

    fake_history = MagicMock()
    fake_history.n_populations = 1
    fake_history.total_nr_simulations = 1
    fake_abc_smc = MagicMock()
    fake_abc_smc.history = fake_history

    with patch.object(ctx, "setup_sampler", return_value=MagicMock()), \
         patch.object(ctx, "setup_population_strategy", return_value=100), \
         patch.object(ctx, "setup_distance_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_transition_function", return_value=None), \
         patch.object(ctx, "setup_epsilon_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_abc_smc", return_value=fake_abc_smc), \
         patch.object(ctx, "load_or_create_database", return_value=(False, 0, 0)), \
         patch.object(ctx, "run_calibration") as mock_run_calibration, \
         patch.object(ctx, "_instantiate_model", return_value=MagicMock()), \
         patch("uq_physicell.abc.abc_context.insert_metadata_db", side_effect=RuntimeError("db locked")), \
         patch("uq_physicell.abc.abc_context.insert_models_db"), \
         caplog.at_level(logging.WARNING):
        result = run_abc_calibration(calib_context=ctx)

    assert result is fake_history
    mock_run_calibration.assert_called_once()
    assert any("Could not persist calibration metadata" in r.message for r in caplog.records)


def test_run_abc_calibration_final_persist_failure_logs_warning(tmp_path, caplog):
    """A failure in the final adaptive-weights/model-probabilities persist block
    (after run_calibration) must be caught and logged, not raised.
    """
    from uq_physicell.abc import CalibrationContext, run_abc_calibration

    adaptive_file = tmp_path / "adaptive_log.json"
    adaptive_file.write_text("{}")

    ctx = CalibrationContext(db_path="dummy.db", **_minimal_kwargs(
        abc_options={'adaptive_distance': True, 'adaptive_distance_file': str(adaptive_file)}
    ))

    fake_history = MagicMock()
    fake_history.n_populations = 1
    fake_history.total_nr_simulations = 1
    fake_abc_smc = MagicMock()
    fake_abc_smc.history = fake_history

    with patch.object(ctx, "setup_sampler", return_value=MagicMock()), \
         patch.object(ctx, "setup_population_strategy", return_value=100), \
         patch.object(ctx, "setup_distance_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_transition_function", return_value=None), \
         patch.object(ctx, "setup_epsilon_function", return_value=MagicMock()), \
         patch.object(ctx, "setup_abc_smc", return_value=fake_abc_smc), \
         patch.object(ctx, "load_or_create_database", return_value=(False, 0, 0)), \
         patch.object(ctx, "run_calibration"), \
         patch.object(ctx, "_instantiate_model", return_value=MagicMock()), \
         patch("uq_physicell.abc.abc_context.insert_metadata_db"), \
         patch("uq_physicell.abc.abc_context.insert_models_db"), \
         patch("uq_physicell.abc.abc_context.insert_adaptive_weights_db", side_effect=RuntimeError("disk full")), \
         patch("uq_physicell.abc.abc_context.load_dict_from_json", return_value={}), \
         caplog.at_level(logging.WARNING):
        result = run_abc_calibration(calib_context=ctx)

    assert result is fake_history
    assert any("Could not persist calibration metadata" in r.message for r in caplog.records)


def test_patch_pyabc_nan_particle_weight_guards_degenerate_weight():
    """A proposed parameter landing far enough into a transition kernel's tail
    can make both prior and transition density underflow to exactly 0.0,
    producing a 0.0/0.0 = nan particle weight that (unpatched) crashes pyABC's
    own population-weight-normalization assertion only after every simulation
    in that population has already finished. The patch (applied automatically
    on `import uq_physicell.abc`) must substitute 0.0 for any non-finite
    weight instead, and leave normal weight computation untouched.
    """
    import numpy as np
    import uq_physicell.abc  # noqa: F401 -- applies the patch as an import side effect
    from pyabc.inference_util.inference_util import create_weight_function, _weight_function

    assert getattr(_weight_function, "_uq_physicell_nan_guard", False)

    # numpy-float 0/0 -> nan silently, the actual real-world failure mode
    wf_nan = create_weight_function(prior_pdf=lambda m, t: np.float64(0.0), transition_pdf=lambda m, t: np.float64(0.0))
    assert wf_nan(0, {"x": 1.0}, acceptance_weight=1.0) == 0.0

    # plain-Python-float 0/0 raises ZeroDivisionError instead -- same guard applies
    wf_zde = create_weight_function(prior_pdf=lambda m, t: 0.0, transition_pdf=lambda m, t: 0.0)
    assert wf_zde(0, {"x": 1.0}, acceptance_weight=1.0) == 0.0

    # nonzero / 0 -> inf, also guarded
    wf_inf = create_weight_function(prior_pdf=lambda m, t: np.float64(2.0), transition_pdf=lambda m, t: np.float64(0.0))
    assert wf_inf(0, {"x": 1.0}, acceptance_weight=1.0) == 0.0

    # Non-degenerate weight computation is unaffected
    wf_ok = create_weight_function(prior_pdf=lambda m, t: 2.0, transition_pdf=lambda m, t: 4.0)
    assert wf_ok(0, {"x": 1.0}, acceptance_weight=1.0) == 0.5


def main():
    """Run all tests."""
    print("🧪 Testing ABC CalibrationContext")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_initialization),
        ("Validation Test", test_configuration_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append(result)
        print(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! CalibrationContext is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)