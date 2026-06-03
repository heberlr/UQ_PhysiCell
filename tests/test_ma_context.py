"""Unit tests for ModelAnalysisContext — API surface and sampler validation.

Covers:
- model_config normalization (dict / tuple / list)
- set_samples() with dict and list inputs
- run() as an alias for run_simulations()
- QoI lambda serialization validation at context creation
- _validate_params_for_sampler() for all sampler families
"""

import logging
import pytest
from unittest.mock import patch

from uq_physicell.model_analysis.ma_context import ModelAnalysisContext


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
