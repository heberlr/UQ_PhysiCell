"""Integration test for run_simulations with inter-process (concurrent.futures) execution.

Mirrors the ex2/ex3 workflow: User-defined sampler, set_samples(), context.run().
PhysiCell_Model and the replicate runner are mocked so no real PhysiCell binary is needed.
"""

import pickle
import pytest
from unittest.mock import patch, call

from uq_physicell.model_analysis.ma_context import ModelAnalysisContext, run_simulations


# ─── mocks ──────────────────────────────────────────────────────────────────

NUM_REPLICATES = 2


class MockPhysiCellModel:
    """Minimal stand-in for PhysiCell_Model — no file I/O required."""
    def __init__(self, ini_filePath, strucName):
        self.configFilePath = ini_filePath
        self.keyModel = strucName
        self.numReplicates = NUM_REPLICATES
        self.XML_parameters_variable = {"xml_p1": "p1", "xml_p2": "p2"}
        self.parameters_rules_variable = {}
        self.output_folder = "/tmp/uq_test_output"

    def info(self):
        return "MockPhysiCellModel info"


def mock_run_replicate_serializable(
    PhysiCellModel_conf, sample_id, replicate_id,
    ParametersXML, ParametersRules,
    qoi_functions=None, qoi_def={},
    return_binary_output=True, drop_columns=None,
    custom_summary_function=None,
):
    return sample_id, replicate_id, pickle.dumps({"out1": 1.0})


# ─── shared patch set (applied per test function) ───────────────────────────

def mock_run_replicate(
    PhysiCell_Model=None, sample_id=None, replicate_id=None,
    ParametersXML=None, ParametersRules=None,
    qoi_functions=None, qoi_def={},
    return_binary_output=True, drop_columns=None,
    custom_summary_function=None,
):
    return sample_id, replicate_id, pickle.dumps({"out1": 1.0})


PATCHES = [
    patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model",           new=MockPhysiCellModel),
    patch("uq_physicell.model_analysis.ma_context.run_replicate_serializable", new=mock_run_replicate_serializable),
    patch("uq_physicell.model_analysis.ma_context.run_replicate",              new=mock_run_replicate),
    patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
]


def _apply_patches(fn):
    """Decorator: apply shared patches to a single test function."""
    for p in reversed(PATCHES):
        fn = p(fn)
    return fn


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def model_config():
    return {"ini_path": "test.ini", "struc_name": "test_model"}


@pytest.fixture
def params_info():
    return {"p1": {"ref_value": 1.0}, "p2": {"ref_value": 2.0}}


@pytest.fixture
def samples():
    return {
        0: {"p1": 1.0, "p2": 2.0},
        1: {"p1": 1.5, "p2": 2.5},
        2: {"p1": 0.5, "p2": 1.5},
    }


def _make_ctx(db_path, model_config, params_info, qois_info=None, **kw):
    return ModelAnalysisContext(
        db_path, model_config, "User-defined",
        params_info, qois_info or {}, **kw,
    )


# ─── tests ───────────────────────────────────────────────────────────────────

@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_db_structure_created(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,                           # _disable_wal_mode
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info, num_workers=2)
    ctx.set_samples(samples)
    run_simulations(ctx)

    mock_create.assert_called_once_with(db_path)
    mock_meta.assert_called_once()
    mock_params.assert_called_once()
    mock_qois.assert_called_once()
    mock_samples.assert_called_once()


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_output_inserted_for_each_sample_replicate(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info, num_workers=2)
    ctx.set_samples(samples)
    run_simulations(ctx)

    expected = len(samples) * NUM_REPLICATES   # 3 samples × 2 replicates = 6
    assert mock_output.call_count == expected


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_context_run_alias_equivalent(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info, num_workers=2)
    ctx.set_samples(samples)
    ctx.run()   # alias — same result as run_simulations(ctx)

    mock_create.assert_called_once()
    assert mock_output.call_count == len(samples) * NUM_REPLICATES


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_serial_execution_also_works(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info,
                    parallel_method="serial", num_workers=1)
    ctx.set_samples(samples)
    ctx.run()

    mock_create.assert_called_once()
    assert mock_output.call_count == len(samples) * NUM_REPLICATES


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
       return_value=(True, [], [], []))   # DB exists, no missing simulations
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_existing_db_skips_structure_creation(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info, num_workers=1)
    ctx.set_samples(samples)
    run_simulations(ctx)

    mock_create.assert_not_called()
    mock_meta.assert_not_called()
    mock_output.assert_not_called()


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_qoi_funcs_stored_in_db(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    qoi_funcs = {"live": lambda df_cell: len(df_cell[df_cell["dead"] == False])}
    ctx = _make_ctx(db_path, model_config, params_info, qois_info=qoi_funcs, num_workers=2)
    ctx.set_samples(samples)
    ctx.run()

    mock_qois.assert_called_once()
    stored_qois = mock_qois.call_args[0][1]
    assert "live" in stored_qois


@_apply_patches
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
def test_cancellation_before_run_inserts_nothing(
    mock_output, mock_samples, mock_qois, mock_params,
    mock_meta, mock_create, mock_check,
    mock_wal,
    db_path, model_config, params_info, samples,
):
    ctx = _make_ctx(db_path, model_config, params_info, num_workers=2)
    ctx.set_samples(samples)
    ctx._cancellation_requested = True
    run_simulations(ctx)

    mock_output.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
