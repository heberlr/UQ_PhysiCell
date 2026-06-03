"""Integration test for run_simulations with inter-node (MPI) execution.

Skipped automatically when mpi4py is not installed.
PhysiCell_Model and replicate runner are mocked — no real PhysiCell binary needed.
"""

import pickle
import pytest
from unittest.mock import patch

try:
    from mpi4py import MPI
    mpi_available = True
except (ImportError, RuntimeError):
    mpi_available = False
    MPI = None

from uq_physicell.model_analysis.ma_context import ModelAnalysisContext, run_simulations


pytestmark = pytest.mark.skipif(not mpi_available, reason="mpi4py not installed")


# ─── mocks ──────────────────────────────────────────────────────────────────

class MockPhysiCellModel:
    def __init__(self, ini_filePath, strucName):
        self.configFilePath = ini_filePath
        self.keyModel = strucName
        self.numReplicates = 2
        self.XML_parameters_variable = {"xml_p1": "p1", "xml_p2": "p2"}
        self.parameters_rules_variable = {}
        self.output_folder = "/tmp/uq_mpi_test_output"

    def info(self):
        return "MockPhysiCellModel info"

    def RunModel(self, *args, **kwargs):
        return {"out1": 1.0}


def mock_run_replicate(
    PhysiCellModel, sample_id, replicate_id,
    ParametersXML, ParametersRules,
    qoi_functions=None, qoi_def={},
    return_binary_output=True, drop_columns=None,
    custom_summary_function=None,
):
    return sample_id, replicate_id, pickle.dumps({"out1": 1.0})


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_mpi.db")


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


# ─── tests ───────────────────────────────────────────────────────────────────

@patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", new=MockPhysiCellModel)
@patch("uq_physicell.model_analysis.ma_context.run_replicate", new=mock_run_replicate)
@patch("uq_physicell.model_analysis.ma_context.check_simulations_db", return_value=(False, [], [], []))
@patch("uq_physicell.model_analysis.ma_context.create_structure")
@patch("uq_physicell.model_analysis.ma_context.insert_metadata")
@patch("uq_physicell.model_analysis.ma_context.insert_param_space")
@patch("uq_physicell.model_analysis.ma_context.insert_qois")
@patch("uq_physicell.model_analysis.ma_context.insert_samples")
@patch("uq_physicell.model_analysis.ma_context.insert_output")
@patch("uq_physicell.model_analysis.ma_context._disable_wal_mode")
class TestRunSimulationsMPI:

    def test_rank0_creates_db_structure(
        self, mock_wal, mock_output, mock_samples, mock_qois,
        mock_params, mock_meta, mock_create, mock_check,
        db_path, model_config, params_info, samples,
    ):
        ctx = ModelAnalysisContext(
            db_path, model_config, "User-defined", params_info, {},
            parallel_method="inter-node",
        )
        ctx.set_samples(samples)
        run_simulations(ctx)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            mock_create.assert_called_once_with(db_path)
            mock_meta.assert_called_once()
            mock_params.assert_called_once()
            mock_qois.assert_called_once()
            mock_samples.assert_called_once()

    def test_output_inserted_across_ranks(
        self, mock_wal, mock_output, mock_samples, mock_qois,
        mock_params, mock_meta, mock_create, mock_check,
        db_path, model_config, params_info, samples,
    ):
        ctx = ModelAnalysisContext(
            db_path, model_config, "User-defined", params_info, {},
            parallel_method="inter-node",
        )
        ctx.set_samples(samples)
        run_simulations(ctx)

        # Each rank inserts its share; total across all ranks = samples × replicates
        comm = MPI.COMM_WORLD
        local_count = mock_output.call_count
        total_count = comm.allreduce(local_count, op=MPI.SUM)
        expected = len(samples) * MockPhysiCellModel("", "").numReplicates
        assert total_count == expected

    def test_context_run_alias(
        self, mock_wal, mock_output, mock_samples, mock_qois,
        mock_params, mock_meta, mock_create, mock_check,
        db_path, model_config, params_info, samples,
    ):
        ctx = ModelAnalysisContext(
            db_path, model_config, "User-defined", params_info, {},
            parallel_method="inter-node",
        )
        ctx.set_samples(samples)
        ctx.run()   # alias — equivalent to run_simulations(ctx)

        rank = MPI.COMM_WORLD.Get_rank()
        if rank == 0:
            mock_create.assert_called_once()


if __name__ == "__main__":
    # Run with: mpiexec -n 4 python tests/test_ma_mpi_run_simulations.py
    pytest.main([__file__, "-v"])
