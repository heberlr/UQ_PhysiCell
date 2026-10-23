"""Unit tests for ModelAnalysisContext — API surface and sampler validation.

Covers:
- model_config normalization (dict / tuple / list)
- set_samples() with dict and list inputs
- run() as an alias for run_simulations()
- QoI lambda serialization validation at context creation
- _validate_params_for_sampler() for all sampler families
"""

import logging
import pickle
import signal
import pytest
from unittest.mock import patch, MagicMock

import uq_physicell.model_analysis.ma_context as ma_context_module
from uq_physicell.model_analysis.ma_context import ModelAnalysisContext, run_simulations


# ─── helpers ────────────────────────────────────────────────────────────────

def _make_context(tmp_path, sampler, params_info, qois_info=None, **kw):
    return ModelAnalysisContext(
        str(tmp_path / "test.db"),
        {"ini_path": "test.ini", "struc_name": "model"},
        sampler,
        params_info,
        qois_info if qois_info is not None else {},
        **kw,
    )


# ─── model_config normalization ─────────────────────────────────────────────

class TestModelConfigNormalization:
    def test_dict_form(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        assert ctx.dic_metadata["IniFilePath"] == "test.ini"
        assert ctx.dic_metadata["StrucName"] == "model"

    def test_tuple_form(self, tmp_path):
        ctx = ModelAnalysisContext(
            str(tmp_path / "test.db"),
            ("test.ini", "model"),
            "User-defined", {}, {},
        )
        assert ctx.dic_metadata["IniFilePath"] == "test.ini"
        assert ctx.dic_metadata["StrucName"] == "model"

    def test_list_form(self, tmp_path):
        ctx = ModelAnalysisContext(
            str(tmp_path / "test.db"),
            ["test.ini", "model"],
            "User-defined", {}, {},
        )
        assert ctx.dic_metadata["IniFilePath"] == "test.ini"
        assert ctx.dic_metadata["StrucName"] == "model"


# ─── set_samples ────────────────────────────────────────────────────────────

class TestSetSamples:
    @pytest.fixture
    def ctx(self, tmp_path):
        return _make_context(tmp_path, "User-defined", {})

    def test_dict_input_stored_as_is(self, ctx):
        samples = {
            0: {"p1": 1.0, "p2": 2.0},
            1: {"p1": 1.5, "p2": 2.5},
        }
        ctx.set_samples(samples)
        assert ctx.dic_samples == samples

    def test_list_input_auto_assigns_ids(self, ctx):
        samples = [{"p1": 1.0, "p2": 2.0}, {"p1": 1.5, "p2": 2.5}]
        ctx.set_samples(samples)
        assert ctx.dic_samples[0] == {"p1": 1.0, "p2": 2.0}
        assert ctx.dic_samples[1] == {"p1": 1.5, "p2": 2.5}

    def test_single_sample_list(self, ctx):
        ctx.set_samples([{"p1": 0.125}])
        assert len(ctx.dic_samples) == 1
        assert 0 in ctx.dic_samples

    def test_overwrite_existing_samples(self, ctx):
        ctx.set_samples({0: {"p1": 1.0}})
        ctx.set_samples({0: {"p1": 9.9}, 1: {"p1": 8.8}})
        assert len(ctx.dic_samples) == 2
        assert ctx.dic_samples[0]["p1"] == 9.9


# ─── run() alias ────────────────────────────────────────────────────────────

class TestRunAlias:
    def test_run_delegates_to_run_simulations(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        with patch("uq_physicell.model_analysis.ma_context.run_simulations") as mock_rs:
            ctx.run()
            mock_rs.assert_called_once_with(ctx)


# ─── QoI serialization validation ───────────────────────────────────────────

class TestQoISerializationValidation:
    def test_valid_df_cell_lambda_accepted(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {},
            qois_info={"live": lambda df_cell: len(df_cell[df_cell["dead"] == False])})
        assert "live" in ctx.qois_dict

    def test_valid_df_subs_lambda_accepted(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {},
            qois_info={"ifn": lambda df_subs: df_subs["interferon"].mean()})
        assert "ifn" in ctx.qois_dict

    def test_closure_over_local_variable_raises(self, tmp_path):
        threshold = 100
        with pytest.raises(ValueError, match="threshold"):
            _make_context(tmp_path, "User-defined", {},
                qois_info={"q": lambda df_cell: len(df_cell[df_cell["count"] > threshold])})

    def test_qoi_def_allows_external_helper(self, tmp_path):
        def count_dead(df):
            return len(df[df["dead"] == True])

        ctx = _make_context(tmp_path, "User-defined", {},
            qois_info={"dead": lambda df_cell: count_dead(df_cell)},
            qoi_def={"count_dead": count_dead})
        assert "dead" in ctx.qois_dict

    def test_empty_qois_info_accepted(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {}, qois_info={})
        assert ctx.qois_dict == {}

    def test_none_qois_info_accepted(self, tmp_path):
        """Regression test: qois_info=None must not crash __init__ (as it did with
        AttributeError: 'NoneType' object has no attribute 'items').

        Calls ModelAnalysisContext directly rather than through _make_context,
        whose own qois_info if qois_info is not None else {} normalization would
        silently avoid ever exercising the real None path (as happened before
        this fix, when the tab2 GUI passed None straight through and crashed).
        None means "no QoI processing, store the raw mcds list" -- distinct from
        {} ("compute zero QoIs per snapshot") -- so it must be preserved, not
        coerced to {}.
        """
        ctx = ModelAnalysisContext(
            str(tmp_path / "test.db"),
            {"ini_path": "test.ini", "struc_name": "model"},
            "User-defined",
            {},
            None,
        )
        assert ctx.qois_dict is None

    def test_string_qoi_function_accepted_directly(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {},
            qois_info={"live": "lambda df_cell: len(df_cell[df_cell['dead'] == False])"})
        assert "live" in ctx.qois_dict


# ─── Sampler validation ──────────────────────────────────────────────────────

GLOBAL_SAMPLERS = ["Sobol", "Latin hypercube sampling (LHS)", "Fast"]


class TestGlobalSamplerValidation:
    @pytest.mark.parametrize("sampler", GLOBAL_SAMPLERS)
    def test_valid_bounds_passes(self, tmp_path, sampler):
        ctx = _make_context(tmp_path, sampler,
            {"p1": {"lower_bound": 0.0, "upper_bound": 1.0}})
        ctx._validate_params_for_sampler()  # must not raise

    @pytest.mark.parametrize("sampler", GLOBAL_SAMPLERS)
    def test_ref_value_optional(self, tmp_path, sampler):
        # ref_value is metadata — not required for sampling
        ctx = _make_context(tmp_path, sampler,
            {"p1": {"lower_bound": 0.0, "upper_bound": 1.0, "ref_value": 0.5}})
        ctx._validate_params_for_sampler()  # must not raise

    @pytest.mark.parametrize("sampler", ["Sobol", "Latin hypercube sampling (LHS)"])
    def test_missing_lower_bound_raises(self, tmp_path, sampler):
        ctx = _make_context(tmp_path, sampler,
            {"p1": {"upper_bound": 1.0}})
        with pytest.raises(ValueError, match="lower_bound"):
            ctx._validate_params_for_sampler()

    @pytest.mark.parametrize("sampler", ["Sobol", "Latin hypercube sampling (LHS)"])
    def test_missing_upper_bound_raises(self, tmp_path, sampler):
        ctx = _make_context(tmp_path, sampler,
            {"p1": {"lower_bound": 0.0}})
        with pytest.raises(ValueError, match="upper_bound"):
            ctx._validate_params_for_sampler()

    @pytest.mark.parametrize("sampler", ["Sobol", "Latin hypercube sampling (LHS)"])
    def test_perturbation_present_warns(self, tmp_path, sampler, caplog):
        ctx = _make_context(tmp_path, sampler,
            {"p1": {"lower_bound": 0.0, "upper_bound": 1.0, "perturbation": 50.0}})
        with caplog.at_level(logging.WARNING):
            ctx._validate_params_for_sampler()
        assert "perturbation" in caplog.text
        assert "ignored" in caplog.text

    @pytest.mark.parametrize("sampler", ["Sobol", "Latin hypercube sampling (LHS)"])
    def test_multiple_params_all_validated(self, tmp_path, sampler):
        # First param OK, second missing upper_bound → error names the offending param
        ctx = _make_context(tmp_path, sampler, {
            "p1": {"lower_bound": 0.0, "upper_bound": 1.0},
            "p2": {"lower_bound": 0.0},   # missing upper_bound
        })
        with pytest.raises(ValueError, match="p2"):
            ctx._validate_params_for_sampler()


class TestOATSamplerValidation:
    def test_valid_params_passes(self, tmp_path):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"ref_value": 1.0, "perturbation": [1.0, 5.0, 10.0]}})
        ctx._validate_params_for_sampler()

    def test_missing_ref_value_raises(self, tmp_path):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"perturbation": [5.0]}})
        with pytest.raises(ValueError, match="ref_value"):
            ctx._validate_params_for_sampler()

    def test_missing_perturbation_raises(self, tmp_path):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"ref_value": 1.0}})
        with pytest.raises(ValueError, match="perturbation"):
            ctx._validate_params_for_sampler()

    def test_perturbation_none_raises_for_continuous(self, tmp_path):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"ref_value": 1.0, "perturbation": None}})
        with pytest.raises(ValueError, match="perturbation"):
            ctx._validate_params_for_sampler()

    def test_bool_type_allows_none_perturbation(self, tmp_path):
        ctx = _make_context(tmp_path, "OAT",
            {"flag": {"ref_value": 1, "perturbation": None, "type": "bool"}})
        ctx._validate_params_for_sampler()  # must not raise

    def test_lower_bound_present_warns(self, tmp_path, caplog):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"ref_value": 1.0, "perturbation": [5.0], "lower_bound": 0.0}})
        with caplog.at_level(logging.WARNING):
            ctx._validate_params_for_sampler()
        assert "lower_bound" in caplog.text

    def test_upper_bound_present_warns(self, tmp_path, caplog):
        ctx = _make_context(tmp_path, "OAT",
            {"p1": {"ref_value": 1.0, "perturbation": [5.0], "upper_bound": 2.0}})
        with caplog.at_level(logging.WARNING):
            ctx._validate_params_for_sampler()
        assert "upper_bound" in caplog.text


class TestUserDefinedSamplerValidation:
    def test_empty_params_info_passes(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        ctx._validate_params_for_sampler()

    def test_ref_value_only_passes(self, tmp_path):
        # ref_value is optional metadata — valid even without bounds
        ctx = _make_context(tmp_path, "User-defined",
            {"p1": {"ref_value": 0.125}, "p2": {"ref_value": 1.0}})
        ctx._validate_params_for_sampler()

    def test_no_bounds_no_perturbation_no_error(self, tmp_path):
        # User-defined ignores all fields — no validation errors or warnings
        ctx = _make_context(tmp_path, "User-defined",
            {"p1": {"ref_value": 1.0, "lower_bound": 0.0, "upper_bound": 2.0}})
        ctx._validate_params_for_sampler()  # no warning expected for User-defined


# ─── QoI recreation failure at init ──────────────────────────────────────────

class TestQoIRecreationFailureAtInit:
    def test_syntactically_invalid_qoi_string_raises(self, tmp_path):
        with pytest.raises(ValueError, match="cannot be serialized"):
            _make_context(tmp_path, "User-defined", {},
                qois_info={"bad": "lambda df_cell: ("})


# ─── Parallelization method validation at init ───────────────────────────────

class TestParallelMethodValidation:
    def test_inter_node_without_mpi_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ma_context_module, "mpi_available", False)
        monkeypatch.setattr(ma_context_module, "mpi_import_error", RuntimeError("no mpi lib"))
        with pytest.raises(ImportError, match="MPI support is unavailable"):
            _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")

    def test_inter_process_without_futures_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ma_context_module, "futures_available", False)
        with pytest.raises(ImportError, match="concurrent.futures is not available"):
            _make_context(tmp_path, "User-defined", {}, parallel_method="inter-process")

    def test_invalid_parallel_method_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid parallel_method"):
            _make_context(tmp_path, "User-defined", {}, parallel_method="bogus")

    def test_serial_forces_single_worker(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", num_workers=8)
        assert ctx.num_workers == 1


# ─── generate_samples dispatch ───────────────────────────────────────────────

class TestGenerateSamplesDispatch:
    def test_oat_calls_run_local_sampler(self, tmp_path):
        params = {"p1": {"ref_value": 1.0, "perturbation": [5.0]}}
        ctx = _make_context(tmp_path, "OAT", params)
        fake_samples = {0: {"p1": 1.0}, 1: {"p1": 1.05}}
        with patch("uq_physicell.model_analysis.ma_context.run_local_sampler",
                   return_value=fake_samples) as mock_local:
            ctx.generate_samples()
        mock_local.assert_called_once_with(params, "OAT")
        assert ctx.dic_samples == fake_samples

    def test_global_sampler_calls_run_global_sampler(self, tmp_path):
        params = {"p1": {"lower_bound": 0.0, "upper_bound": 1.0}}
        ctx = _make_context(tmp_path, "Sobol", params)
        fake_samples = {0: {"p1": 0.5}}
        with patch("uq_physicell.model_analysis.ma_context.run_global_sampler",
                   return_value=fake_samples) as mock_global:
            ctx.generate_samples(N=8, M=4, seed=123)
        mock_global.assert_called_once_with(params, "Sobol", 8, 4, 123)
        assert ctx.dic_samples == fake_samples

    def test_user_defined_does_not_generate_samples(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        with patch("uq_physicell.model_analysis.ma_context.run_local_sampler") as mock_local, \
             patch("uq_physicell.model_analysis.ma_context.run_global_sampler") as mock_global:
            ctx.generate_samples()
        mock_local.assert_not_called()
        mock_global.assert_not_called()
        assert not hasattr(ctx, "dic_samples")

    def test_generate_samples_validates_before_sampling(self, tmp_path):
        params = {"p1": {"upper_bound": 1.0}}  # missing lower_bound
        ctx = _make_context(tmp_path, "Sobol", params)
        with patch("uq_physicell.model_analysis.ma_context.run_global_sampler") as mock_global:
            with pytest.raises(ValueError, match="lower_bound"):
                ctx.generate_samples()
        mock_global.assert_not_called()


# ─── cancelled() / request_cancellation() ────────────────────────────────────

class TestCancelled:
    def test_default_false(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        assert ctx.cancelled() is False

    def test_true_after_request(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        ctx.request_cancellation()
        assert ctx.cancelled() is True


class TestRequestCancellation:
    def test_sets_flag_and_returns_true(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        result = ctx.request_cancellation()
        assert result is True
        assert ctx._cancellation_requested is True

    def test_terminates_model_processes(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        mock_model = MagicMock()
        mock_model.terminate_all_simulations.return_value = {1: 0, 2: -9}
        ctx.model = mock_model
        ctx.request_cancellation()
        mock_model.terminate_all_simulations.assert_called_once()

    def test_no_model_no_error(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {})
        assert ctx.model is None
        ctx.request_cancellation()  # must not raise

    def test_cancels_pending_futures_inter_process(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-process")
        pending = MagicMock()
        pending.done.return_value = False
        pending.cancelled.return_value = False
        done_future = MagicMock()
        done_future.done.return_value = True
        already_cancelled = MagicMock()
        already_cancelled.done.return_value = False
        already_cancelled.cancelled.return_value = True
        ctx.futures = [pending, done_future, already_cancelled]
        ctx.request_cancellation()
        pending.cancel.assert_called_once()
        done_future.cancel.assert_not_called()
        already_cancelled.cancel.assert_not_called()

    def test_does_not_cancel_futures_for_serial(self, tmp_path):
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial")
        pending = MagicMock()
        pending.done.return_value = False
        pending.cancelled.return_value = False
        ctx.futures = [pending]
        ctx.request_cancellation()
        pending.cancel.assert_not_called()


# ─── run_simulations: serial path, signal handling, and mocked MPI ──────────

def _make_mock_model(tmp_path, num_replicates=2):
    model = MagicMock()
    model.output_folder = str(tmp_path / "nonexistent_output")
    model.build_effective_config_fingerprint.return_value = {
        "ini_file_hash": "h1", "xml_file_hash": "h2", "rules_file_hash": "h3",
        "structure_config_hash": "h4", "effective_run_hash": "h5",
    }
    model.XML_parameters_variable = {}
    model.parameters_rules_variable = {}
    model.numReplicates = num_replicates
    model.info.return_value = None
    return model


class TestRunSimulationsSerial:
    def test_new_db_serial_execution(self, tmp_path):
        model = _make_mock_model(tmp_path)
        model.info.side_effect = lambda: print("Model Info")
        # An existing output folder must be removed before generating new simulations.
        output_folder = tmp_path / "existing_output"
        output_folder.mkdir()
        model.output_folder = str(output_folder)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial")
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure") as mock_create,
            patch("uq_physicell.model_analysis.ma_context.insert_metadata") as mock_ins_meta,
            patch("uq_physicell.model_analysis.ma_context.insert_param_space") as mock_ins_params,
            patch("uq_physicell.model_analysis.ma_context.insert_qois") as mock_ins_qois,
            patch("uq_physicell.model_analysis.ma_context.insert_samples") as mock_ins_samples,
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, {"live": 5}, 42)) as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode") as mock_disable_wal,
        ):
            run_simulations(ctx)

        mock_create.assert_called_once_with(ctx.db_path)
        mock_ins_meta.assert_called_once()
        mock_ins_params.assert_called_once_with(ctx.db_path, ctx.params_dict)
        mock_ins_qois.assert_called_once_with(ctx.db_path, ctx.qois_dict)
        mock_ins_samples.assert_called_once_with(ctx.db_path, ctx.dic_samples)
        assert mock_run_replicate.call_count == 2  # 1 sample x numReplicates=2
        assert mock_insert_output.call_count == 2
        mock_disable_wal.assert_called_once_with(ctx.db_path)
        assert ctx.model is model
        assert not output_folder.exists()

    def test_create_structure_failure_reraises(self, tmp_path):
        model = _make_mock_model(tmp_path)
        mock_logger = MagicMock()
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", logger=mock_logger)
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure",
                  side_effect=RuntimeError("create boom")),
        ):
            with pytest.raises(RuntimeError, match="create boom"):
                run_simulations(ctx)
        assert mock_logger.error.called

    def test_final_insert_output_failure_reraises(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        mock_logger = MagicMock()
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", logger=mock_logger)
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, b"raw", 1)),
            patch("uq_physicell.model_analysis.ma_context.insert_output",
                  side_effect=RuntimeError("write boom")),
        ):
            with pytest.raises(RuntimeError, match="write boom"):
                run_simulations(ctx)
        assert mock_logger.error.called

    def test_existing_db_skips_structure_creation(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial")
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(True, [{"p1": 1.0}], [0], [0])),
            patch("uq_physicell.model_analysis.ma_context.create_structure") as mock_create,
            patch("uq_physicell.model_analysis.ma_context.insert_samples") as mock_ins_samples,
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, b"raw", 7)) as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        mock_create.assert_not_called()
        mock_ins_samples.assert_not_called()
        mock_run_replicate.assert_called_once()
        mock_insert_output.assert_called_once_with(ctx.db_path, 0, 0, pickle.dumps(b"raw"), seed=7)

    def test_custom_summary_function_used_when_provided(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        model.RunModel.return_value = {"qoi": 1}
        model._last_random_seed = 999
        summary_fn = lambda *a, **kw: None
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial",
                             summary_function=summary_fn)
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        model.RunModel.assert_called_once()
        _, run_kwargs = model.RunModel.call_args
        assert run_kwargs["SummaryFunction"] is summary_fn
        mock_insert_output.assert_called_once()
        out_args, out_kwargs = mock_insert_output.call_args
        assert out_args[0] == ctx.db_path
        assert out_args[1] == 0
        assert out_args[2] == 0
        assert out_kwargs["seed"] == 999

    def test_cancellation_before_loop_skips_execution(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial")
        ctx.set_samples({0: {"p1": 1.0}})
        ctx.request_cancellation()

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate") as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode") as mock_disable_wal,
        ):
            run_simulations(ctx)

        mock_run_replicate.assert_not_called()
        mock_insert_output.assert_not_called()
        mock_disable_wal.assert_called_once()

    def test_model_init_failure_logs_and_reraises(self, tmp_path):
        mock_logger = MagicMock()
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", logger=mock_logger)
        with patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model",
                   side_effect=RuntimeError("init boom")):
            with pytest.raises(RuntimeError, match="init boom"):
                run_simulations(ctx)
        mock_logger.error.assert_called_once()

    def test_check_simulations_db_failure_reraises(self, tmp_path):
        model = _make_mock_model(tmp_path)
        mock_logger = MagicMock()
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", logger=mock_logger)
        ctx.set_samples({0: {"p1": 1.0}})
        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  side_effect=RuntimeError("check boom")),
        ):
            with pytest.raises(RuntimeError, match="check boom"):
                run_simulations(ctx)
        assert mock_logger.error.called

    def test_insert_metadata_failure_reraises(self, tmp_path):
        model = _make_mock_model(tmp_path)
        mock_logger = MagicMock()
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial", logger=mock_logger)
        ctx.set_samples({0: {"p1": 1.0}})

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata",
                  side_effect=RuntimeError("insert boom")),
        ):
            with pytest.raises(RuntimeError, match="insert boom"):
                run_simulations(ctx)
        assert mock_logger.error.called


class TestRunSimulationsSignalHandler:
    def test_signal_handler_registered_and_invoked(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="serial")
        ctx.set_samples({0: {"p1": 1.0}})
        captured = {}

        def fake_signal(sig, handler):
            captured[sig] = handler
            return None  # never touches the real OS-level handler

        with (
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, b"x", 1)),
            patch("uq_physicell.model_analysis.ma_context.insert_output"),
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
            patch("signal.signal", side_effect=fake_signal),
        ):
            run_simulations(ctx)

        assert signal.SIGINT in captured
        assert signal.SIGTERM in captured

        # SIGINT with a non-'inter-process' context exits the process.
        with pytest.raises(SystemExit):
            captured[signal.SIGINT](signal.SIGINT, None)
        assert ctx.cancelled() is True

        # SIGTERM never calls sys.exit directly -- just requests cancellation.
        captured[signal.SIGTERM](signal.SIGTERM, None)


class TestRunSimulationsMPIMocked:
    """Exercises the inter-node branch with a fully mocked mpi4py.MPI —
    no real MPI runtime required, unlike test_ma_mpi_run_simulations.py
    (which needs `mpiexec -n N` and is otherwise skipped)."""

    def test_rank0_creates_db_and_runs_its_share(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")
        ctx.set_samples({0: {"p1": 1.0}, 1: {"p1": 2.0}})

        fake_comm = MagicMock()
        fake_comm.Get_rank.return_value = 0
        fake_comm.Get_size.return_value = 1
        fake_comm.bcast.side_effect = lambda value, root=0: value
        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD = fake_comm

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure") as mock_create,
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, b"raw", 1)) as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        mock_create.assert_called_once_with(ctx.db_path)
        assert mock_run_replicate.call_count == 2  # 2 samples x numReplicates=1, size=1
        assert mock_insert_output.call_count == 2
        fake_comm.Barrier.assert_called_once()
        fake_mpi.Finalize.assert_called_once()

    def test_nonzero_rank_uses_broadcasted_data_not_local_db_check(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=2)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")
        ctx.set_samples({0: {"p1": 1.0}})  # 1 sample x numReplicates=2 -> All_Samples length 2

        fake_comm = MagicMock()
        fake_comm.Get_rank.return_value = 1
        fake_comm.Get_size.return_value = 2
        # Broadcast call order: exist_db, All_Samples, All_Replicates, All_Parameters, All_Seeds
        fake_comm.bcast.side_effect = [False, [], [], [], [111, 222]]
        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD = fake_comm

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db") as mock_check,
            patch("uq_physicell.model_analysis.ma_context.create_structure") as mock_create,
            patch("uq_physicell.model_analysis.ma_context.insert_metadata") as mock_ins_meta,
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 1, b"raw", 2)) as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        # Non-zero ranks never touch the database-existence check or structure setup directly.
        mock_check.assert_not_called()
        mock_create.assert_not_called()
        mock_ins_meta.assert_not_called()
        # np.array_split(arange(2), 2) -> rank 1 gets index [1] -> one simulation.
        mock_run_replicate.assert_called_once()
        mock_insert_output.assert_called_once()
        fake_comm.Barrier.assert_called_once()
        fake_mpi.Finalize.assert_called_once()

    def _rank0_fake_mpi(self):
        fake_comm = MagicMock()
        fake_comm.Get_rank.return_value = 0
        fake_comm.Get_size.return_value = 1
        fake_comm.bcast.side_effect = lambda value, root=0: value
        fake_mpi = MagicMock()
        fake_mpi.COMM_WORLD = fake_comm
        return fake_mpi, fake_comm

    def test_cancellation_mid_loop_breaks(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")
        ctx.set_samples({0: {"p1": 1.0}, 1: {"p1": 2.0}})
        ctx._cancellation_requested = True
        fake_mpi, fake_comm = self._rank0_fake_mpi()

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate") as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        mock_run_replicate.assert_not_called()
        mock_insert_output.assert_not_called()
        fake_comm.Barrier.assert_called_once()
        fake_mpi.Finalize.assert_called_once()

    def test_summary_function_branch_used(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        model.RunModel.return_value = {"qoi": 1}
        model._last_random_seed = 555
        summary_fn = lambda *a, **kw: None
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node",
                             summary_function=summary_fn)
        ctx.set_samples({0: {"p1": 1.0}})
        fake_mpi, fake_comm = self._rank0_fake_mpi()

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        model.RunModel.assert_called_once()
        mock_insert_output.assert_called_once()
        _, out_kwargs = mock_insert_output.call_args
        assert out_kwargs["seed"] == 555

    def test_per_simulation_exception_continues(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")
        ctx.set_samples({0: {"p1": 1.0}, 1: {"p1": 2.0}})
        mock_logger = MagicMock()
        ctx.logger = mock_logger
        fake_mpi, fake_comm = self._rank0_fake_mpi()

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  side_effect=[RuntimeError("sim boom"), (0, 1, b"ok", 3)]) as mock_run_replicate,
            patch("uq_physicell.model_analysis.ma_context.insert_output") as mock_insert_output,
            patch("uq_physicell.model_analysis.ma_context._disable_wal_mode"),
        ):
            run_simulations(ctx)

        assert mock_run_replicate.call_count == 2
        mock_insert_output.assert_called_once()
        assert mock_logger.error.called

    def test_insert_output_failure_raises(self, tmp_path):
        model = _make_mock_model(tmp_path, num_replicates=1)
        ctx = _make_context(tmp_path, "User-defined", {}, parallel_method="inter-node")
        ctx.set_samples({0: {"p1": 1.0}})
        mock_logger = MagicMock()
        ctx.logger = mock_logger
        fake_mpi, fake_comm = self._rank0_fake_mpi()

        with (
            patch("uq_physicell.model_analysis.ma_context.MPI", fake_mpi),
            patch("uq_physicell.model_analysis.ma_context.PhysiCell_Model", return_value=model),
            patch("uq_physicell.model_analysis.ma_context.check_simulations_db",
                  return_value=(False, [], [], [])),
            patch("uq_physicell.model_analysis.ma_context.create_structure"),
            patch("uq_physicell.model_analysis.ma_context.insert_metadata"),
            patch("uq_physicell.model_analysis.ma_context.insert_param_space"),
            patch("uq_physicell.model_analysis.ma_context.insert_qois"),
            patch("uq_physicell.model_analysis.ma_context.insert_samples"),
            patch("uq_physicell.model_analysis.ma_context.run_replicate",
                  return_value=(0, 0, b"raw", 1)),
            patch("uq_physicell.model_analysis.ma_context.insert_output",
                  side_effect=RuntimeError("mpi write boom")),
        ):
            with pytest.raises(RuntimeError, match="mpi write boom"):
                run_simulations(ctx)
        assert mock_logger.error.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
