"""Unit tests for uq_physicell.utils.sumstats.

Focuses on pure data-transformation / dispatch logic: QoI function wrapping,
lambda-source extraction, recreation from strings, summary aggregation, and
retry/error-handling helpers. PhysiCell/pcdl/MCDS boundaries are mocked so
no real simulation output or files are required.
"""

import errno
import importlib.util
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from uq_physicell.utils.sumstats import (
    _check_functions_need_microenv,
    _convert_qoi_function_to_string,
    _create_named_function_from_string,
    _create_wrapper_for_qoi_function,
    _extract_lambda_source,
    _safe_rmtree,
    qoi_func_persistent_homology,
    qoi_func_radial_density_summary,
    qoi_func_relational_ph,
    recreate_qoi_functions,
    safe_call_qoi_function,
    summ_func_FinalPopLiveDead,
    summ_func_TimeSeriesPopLiveDead,
    summary_function,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def _fake_cell_df(n_live=3, n_dead=2):
    rows = [{"dead": False, "ID": i} for i in range(n_live)]
    rows += [{"dead": True, "ID": i} for i in range(n_live, n_live + n_dead)]
    return pd.DataFrame(rows)


def _fake_mcds(time=0.0, runtime=1.5, cell_df=None, conc_df=None):
    mcds = MagicMock()
    mcds.get_time.return_value = time
    mcds.get_runtime.return_value = runtime
    mcds.get_cell_df.return_value = cell_df if cell_df is not None else _fake_cell_df()
    mcds.get_conc_df.return_value = conc_df if conc_df is not None else pd.DataFrame({"substrate": [1.0, 2.0]})
    return mcds


# ─── summ_func_FinalPopLiveDead ──────────────────────────────────────────────

class TestSummFuncFinalPopLiveDead:
    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        mcds = _fake_mcds(time=100.0, runtime=42.0, cell_df=_fake_cell_df(3, 2))
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeStep", return_value=mcds),
            patch("uq_physicell.utils.sumstats.rmtree") as mock_rmtree,
        ):
            df = summ_func_FinalPopLiveDead(str(outdir), None, {"p1": 1.0}, 0, 1)

        assert df is not None
        assert df.iloc[0]["live_cells"] == 3
        assert df.iloc[0]["dead_cells"] == 2
        assert df.iloc[0]["sampleID"] == 0
        assert df.iloc[0]["replicateID"] == 1
        assert df.iloc[0]["p1"] == 1.0
        assert df.iloc[0]["run_time_sec"] == 42.0
        mock_rmtree.assert_called_once_with(str(outdir))

    def test_writes_to_file_and_returns_none(self, tmp_path):
        mcds = _fake_mcds()
        outdir = tmp_path / "output"
        outdir.mkdir()
        summary_file = tmp_path / "summary.tsv"
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeStep", return_value=mcds),
            patch("uq_physicell.utils.sumstats.rmtree"),
        ):
            result = summ_func_FinalPopLiveDead(str(outdir), str(summary_file), {}, 0, 0)

        assert result is None
        assert summary_file.exists()

    def test_all_dead_cells(self, tmp_path):
        mcds = _fake_mcds(cell_df=_fake_cell_df(0, 5))
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeStep", return_value=mcds),
            patch("uq_physicell.utils.sumstats.rmtree"),
        ):
            df = summ_func_FinalPopLiveDead(str(outdir), None, {}, 0, 0)
        assert df.iloc[0]["live_cells"] == 0
        assert df.iloc[0]["dead_cells"] == 5


# ─── summ_func_TimeSeriesPopLiveDead ─────────────────────────────────────────

class TestSummFuncTimeSeriesPopLiveDead:
    def test_aggregates_across_snapshots(self, tmp_path):
        mcds_list = [
            _fake_mcds(time=0.0, cell_df=_fake_cell_df(5, 0)),
            _fake_mcds(time=60.0, cell_df=_fake_cell_df(4, 1)),
            _fake_mcds(time=120.0, cell_df=_fake_cell_df(2, 3)),
        ]
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = mcds_list
        outdir = tmp_path / "output"
        outdir.mkdir()

        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats.rmtree") as mock_rmtree,
        ):
            df = summ_func_TimeSeriesPopLiveDead(str(outdir), None, {"p1": 2.0}, 3, 4)

        assert len(df) == 3
        assert list(df["time"]) == [0.0, 60.0, 120.0]
        assert list(df["live_cells"]) == [5, 4, 2]
        assert list(df["dead_cells"]) == [0, 1, 3]
        assert (df["sampleID"] == 3).all()
        assert (df["replicateID"] == 4).all()
        assert (df["p1"] == 2.0).all()
        mock_rmtree.assert_called_once_with(str(outdir))

    def test_single_snapshot_writes_file(self, tmp_path):
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds(time=0.0)]
        outdir = tmp_path / "output"
        outdir.mkdir()
        summary_file = tmp_path / "ts.tsv"
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats.rmtree"),
        ):
            result = summ_func_TimeSeriesPopLiveDead(str(outdir), str(summary_file), {}, 0, 0)
        assert result is None
        assert summary_file.exists()


# ─── _check_functions_need_microenv ──────────────────────────────────────────

class TestCheckFunctionsNeedMicroenv:
    def test_empty_dict_defaults_to_true(self):
        assert _check_functions_need_microenv({}) is True

    def test_none_defaults_to_true(self):
        assert _check_functions_need_microenv(None) is True

    def test_df_subs_param_needs_microenv(self):
        func = _create_wrapper_for_qoi_function(lambda df_subs: 1, "df_subs", "q")
        assert _check_functions_need_microenv({"q": func}) is True

    def test_conc_named_param_needs_microenv(self):
        func = _create_wrapper_for_qoi_function(lambda conc_data: 1, "conc_data", "q")
        assert _check_functions_need_microenv({"q": func}) is True

    def test_micro_named_param_needs_microenv(self):
        func = _create_wrapper_for_qoi_function(lambda microenv: 1, "microenv", "q")
        assert _check_functions_need_microenv({"q": func}) is True

    def test_only_cell_functions_do_not_need_microenv(self):
        func = _create_wrapper_for_qoi_function(lambda df_cell: 1, "df_cell", "q")
        assert _check_functions_need_microenv({"q": func}) is False

    def test_mixed_functions_needs_microenv_if_any_does(self):
        func_cell = _create_wrapper_for_qoi_function(lambda df_cell: 1, "df_cell", "q1")
        func_subs = _create_wrapper_for_qoi_function(lambda df_subs: 1, "df_subs", "q2")
        assert _check_functions_need_microenv({"q1": func_cell, "q2": func_subs}) is True


# ─── safe_call_qoi_function ──────────────────────────────────────────────────

class TestSafeCallQoiFunction:
    def test_unrecognized_param_name_without_metadata_raises(self):
        def bare(foo):
            return 1
        with pytest.raises(ValueError, match="Could not dispatch QoI function"):
            safe_call_qoi_function(bare, mcds=_fake_mcds())

    def test_bare_function_dispatches_by_its_own_param_name(self):
        # No wrapper/__param_name__ needed when the parameter is already named after a
        # recognized input (df, df_cell, df_subs, df_conc, adata, sdata, domain, mcds, mcds_ts).
        def bare(df_cell):
            return len(df_cell)
        mcds = _fake_mcds(cell_df=_fake_cell_df(2, 1))
        assert safe_call_qoi_function(bare, mcds=mcds) == 3

    def test_df_cell_dispatches_cell_dataframe(self):
        captured = {}
        def collect(df_cell):
            captured["df"] = df_cell
            return len(df_cell)
        func = _create_wrapper_for_qoi_function(collect, "df_cell", "q")
        mcds = _fake_mcds(cell_df=_fake_cell_df(2, 1))
        result = safe_call_qoi_function(func, mcds=mcds)
        assert result == 3
        assert captured["df"] is mcds.get_cell_df.return_value

    def test_df_alias_dispatches_cell_dataframe(self):
        func = _create_wrapper_for_qoi_function(lambda df: len(df), "df", "q")
        mcds = _fake_mcds(cell_df=_fake_cell_df(1, 1))
        assert safe_call_qoi_function(func, mcds=mcds) == 2

    def test_df_subs_dispatches_conc_dataframe(self):
        func = _create_wrapper_for_qoi_function(lambda df_subs: df_subs["substrate"].mean(), "df_subs", "q")
        mcds = _fake_mcds(conc_df=pd.DataFrame({"substrate": [1.0, 3.0]}))
        assert safe_call_qoi_function(func, mcds=mcds) == 2.0

    def test_sdata_dispatches_spatialdata_and_caches(self):
        func = _create_wrapper_for_qoi_function(lambda sdata: sdata, "sdata", "q")
        mcds = _fake_mcds()
        data_cache = {}
        assert safe_call_qoi_function(func, mcds=mcds, data_cache=data_cache) is mcds.get_spatialdata.return_value
        assert safe_call_qoi_function(func, mcds=mcds, data_cache=data_cache) is mcds.get_spatialdata.return_value
        mcds.get_spatialdata.assert_called_once()

    def test_domain_dispatches_muspan_and_caches(self):
        func = _create_wrapper_for_qoi_function(lambda domain: domain, "domain", "q")
        mcds = _fake_mcds()
        data_cache = {}
        assert safe_call_qoi_function(func, mcds=mcds, data_cache=data_cache) is mcds.get_muspan.return_value
        assert safe_call_qoi_function(func, mcds=mcds, data_cache=data_cache) is mcds.get_muspan.return_value
        mcds.get_muspan.assert_called_once()

    @pytest.mark.parametrize("param_name", ["sdata", "domain"])
    def test_sdata_domain_bare_function_dispatches(self, param_name):
        namespace = {}
        exec(f"def bare({param_name}):\n    return {param_name}", namespace)
        mcds = _fake_mcds()
        expected = mcds.get_spatialdata.return_value if param_name == "sdata" else mcds.get_muspan.return_value
        assert safe_call_qoi_function(namespace["bare"], mcds=mcds) is expected

    @pytest.mark.parametrize("param_name", ["sdata", "domain"])
    def test_sdata_domain_raise_when_mcds_is_none(self, param_name):
        func = _create_wrapper_for_qoi_function(lambda x: x, param_name, "q")
        with pytest.raises(ValueError, match="mcds is None"):
            safe_call_qoi_function(func, mcds=None)

    def test_mcds_param_passes_mcds_object(self):
        func = _create_wrapper_for_qoi_function(lambda mcds: mcds.get_time(), "mcds", "q")
        mcds = _fake_mcds(time=42.0)
        assert safe_call_qoi_function(func, mcds=mcds) == 42.0

    def test_mcds_ts_only_computed_on_last_snapshot(self):
        func = _create_wrapper_for_qoi_function(lambda mcds_ts: len(mcds_ts), "mcds_ts", "q")
        mcds_list = [_fake_mcds(time=0.0), _fake_mcds(time=1.0)]
        assert safe_call_qoi_function(func, mcds=mcds_list[0], list_mcds=mcds_list) is None
        assert safe_call_qoi_function(func, mcds=mcds_list[-1], list_mcds=mcds_list) == 2

    def test_unresolvable_dispatch_raises(self):
        # df_cell param but mcds is explicitly None
        func = _create_wrapper_for_qoi_function(lambda df_cell: 1, "df_cell", "q")
        with pytest.raises(ValueError, match="mcds is None"):
            safe_call_qoi_function(func, mcds=None)


# ─── _create_wrapper_for_qoi_function ────────────────────────────────────────

class TestCreateWrapper:
    def test_wrapper_preserves_metadata_and_delegates(self):
        def original(x):
            return x * 2
        wrapper = _create_wrapper_for_qoi_function(original, "df_cell", "double")
        assert wrapper.__param_name__ == "df_cell"
        assert wrapper.__name__ == "double"
        assert wrapper(21) == 42


# ─── _extract_lambda_source ──────────────────────────────────────────────────

class TestExtractLambdaSource:
    def test_simple_lambda_assignment(self):
        f = lambda df_cell: len(df_cell)
        source = _extract_lambda_source(f)
        assert source.startswith("lambda df_cell:")

    def test_lambda_inside_dict_literal_with_trailing_comma(self):
        d = {
            "q": lambda df_cell: len(df_cell),
        }
        source = _extract_lambda_source(d["q"])
        assert "lambda df_cell" in source
        assert source.strip().endswith(")") or "len(df_cell)" in source

    def test_no_lambda_raises(self):
        def regular_function(x):
            return x
        with pytest.raises(ValueError, match="No lambda expression found"):
            _extract_lambda_source(regular_function)

    def test_lambda_followed_by_trailing_call_syntax_requires_shrinking(self, tmp_path, monkeypatch):
        # inspect.getsource() on a lambda passed as a call argument returns the
        # whole call-site line; the trailing ", extra_arg=1)" makes the naive
        # "wrap the whole thing in parens" parse invalid, forcing the
        # progressively-shorter-suffix loop to actually shrink and retry.
        mod_file = tmp_path / "lambda_case_mod.py"
        mod_file.write_text(
            "def some_call(f, extra_arg=None):\n"
            "    return f\n"
            "\n"
            "captured = some_call(lambda df_cell: len(df_cell), extra_arg=1)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        import importlib
        import sys
        sys.modules.pop("lambda_case_mod", None)
        try:
            mod = importlib.import_module("lambda_case_mod")
            source = _extract_lambda_source(mod.captured)
            assert source == "lambda df_cell: len(df_cell)"
        finally:
            sys.modules.pop("lambda_case_mod", None)


# ─── _convert_qoi_function_to_string ──────────────────────────────────────────

class TestConvertQoiFunctionToString:
    def test_callable_converts_to_source_string(self):
        f = lambda df_cell: len(df_cell)
        result = _convert_qoi_function_to_string(f, "q")
        assert "lambda df_cell" in result

    def test_non_callable_raises(self):
        with pytest.raises(ValueError, match="not callable"):
            _convert_qoi_function_to_string("not a function", "q")


# ─── _create_named_function_from_string ──────────────────────────────────────

class TestCreateNamedFunctionFromString:
    def test_extracts_arg_name_from_lambda(self):
        wrapper = _create_named_function_from_string("live", "lambda df_cell: len(df_cell)")
        assert wrapper.__param_name__ == "df_cell"
        assert wrapper.__name__ == "live"
        assert wrapper(_fake_cell_df(2, 1)) == 3

    def test_defaults_to_mcds_when_no_lambda_arg_match(self):
        # A func_str without a "lambda <name>:" pattern falls back to param_name='mcds'
        def helper(mcds):
            return mcds.get_time()
        wrapper = _create_named_function_from_string("q", "helper", qoi_def={"helper": helper})
        assert wrapper.__param_name__ == "mcds"
        assert wrapper(_fake_mcds(time=7.0)) == 7.0

    def test_qoi_def_namespace_allows_external_helper(self):
        def helper(df):
            return len(df)
        wrapper = _create_named_function_from_string(
            "q", "lambda df_cell: helper(df_cell)", qoi_def={"helper": helper})
        assert wrapper(_fake_cell_df(1, 1)) == 2

    def test_invalid_syntax_raises_value_error(self):
        with pytest.raises(ValueError, match="Error evaluating QoI function string"):
            _create_named_function_from_string("bad", "lambda df_cell: (")

    def test_restricted_namespace_blocks_arbitrary_builtins(self):
        # Lambda creation is lazy -- the body only executes (and fails) when called.
        wrapper = _create_named_function_from_string("bad", "lambda df_cell: __import__('os')")
        with pytest.raises(NameError):
            wrapper(_fake_cell_df())


# ─── recreate_qoi_functions ───────────────────────────────────────────────────

class TestRecreateQoiFunctions:
    def test_recreates_from_string(self):
        funcs = recreate_qoi_functions({"live": "lambda df_cell: len(df_cell)"})
        assert funcs["live"](_fake_cell_df(2, 0)) == 2

    def test_recreates_from_callable(self):
        funcs = recreate_qoi_functions({"live": lambda df_cell: len(df_cell)})
        assert funcs["live"].__param_name__ == "df_cell"
        assert funcs["live"](_fake_cell_df(3, 0)) == 3

    def test_none_placeholder_preserved(self):
        funcs = recreate_qoi_functions({"custom": None})
        assert funcs["custom"] is None

    def test_mixed_types(self):
        funcs = recreate_qoi_functions({
            "a": "lambda df_cell: len(df_cell)",
            "b": lambda df_subs: df_subs.shape[0],
            "c": None,
        })
        assert set(funcs.keys()) == {"a", "b", "c"}
        assert funcs["c"] is None

    def test_error_wraps_with_qoi_name(self):
        with pytest.raises(ValueError, match="Error recreating QoI function 'broken'"):
            recreate_qoi_functions({"broken": "lambda df_cell: ("})

    def test_empty_dict_returns_empty(self):
        assert recreate_qoi_functions({}) == {}


# ─── _safe_rmtree ─────────────────────────────────────────────────────────────

class TestSafeRmtree:
    def test_successful_removal_no_retry(self, tmp_path):
        target = tmp_path / "out"
        target.mkdir()
        with patch("uq_physicell.utils.sumstats.rmtree") as mock_rmtree:
            _safe_rmtree(str(target))
        mock_rmtree.assert_called_once_with(str(target))

    def test_retries_on_enotempty_then_succeeds(self, tmp_path):
        target = str(tmp_path / "out")
        enotempty_err = OSError(errno.ENOTEMPTY, "Directory not empty")
        with (
            patch("uq_physicell.utils.sumstats.rmtree", side_effect=[enotempty_err, None]) as mock_rmtree,
            patch("uq_physicell.utils.sumstats.time.sleep") as mock_sleep,
        ):
            _safe_rmtree(target, retries=5, delay=0.01)
        assert mock_rmtree.call_count == 2
        mock_sleep.assert_called_once_with(0.01)

    def test_non_enotempty_error_breaks_immediately(self, tmp_path):
        target = str(tmp_path / "out")
        other_err = OSError(errno.EACCES, "Permission denied")
        with (
            patch("uq_physicell.utils.sumstats.rmtree", side_effect=[other_err, None]) as mock_rmtree,
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            _safe_rmtree(target, retries=5, delay=0.01)
        # First call raised EACCES (not retried further), then the final best-effort call.
        assert mock_rmtree.call_count == 2
        assert any("Could not fully remove" in str(w.message) for w in caught)

    def test_exhausts_retries_and_warns(self, tmp_path):
        target = str(tmp_path / "out")
        enotempty_err = OSError(errno.ENOTEMPTY, "Directory not empty")
        with (
            patch("uq_physicell.utils.sumstats.rmtree", side_effect=[enotempty_err] * 3 + [None]) as mock_rmtree,
            patch("uq_physicell.utils.sumstats.time.sleep"),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            _safe_rmtree(target, retries=3, delay=0.01)
        assert mock_rmtree.call_count == 4  # 3 retries + 1 final ignore_errors call
        assert any("Could not fully remove" in str(w.message) for w in caught)


# ─── summary_function ─────────────────────────────────────────────────────────

class TestSummaryFunction:
    def test_no_qoi_functions_returns_raw_mcds_list(self, tmp_path):
        mcds_list = [_fake_mcds(time=0.0), _fake_mcds(time=1.0)]
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = mcds_list
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats.rmtree") as mock_rmtree,
        ):
            result = summary_function(str(outdir), None, {}, 0, 0, qoi_functions=None)
        assert result is mcds_list
        mock_rmtree.assert_called_once_with(str(outdir))

    def test_no_qoi_functions_with_drop_columns_returns_dataframes(self, tmp_path):
        cell_df = pd.DataFrame({"position_x": [1.0], "extra_col": ["drop_me"]})
        mcds_list = [_fake_mcds(cell_df=cell_df)]
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = mcds_list
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats.rmtree"),
        ):
            result = summary_function(str(outdir), None, {}, 0, 0,
                                       qoi_functions=None, drop_columns=["extra_col"])
        assert isinstance(result, list)
        assert "extra_col" not in result[0].columns
        assert "position_x" in result[0].columns

    def test_drop_columns_ignores_missing_column(self, tmp_path):
        cell_df = pd.DataFrame({"position_x": [1.0]})
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds(cell_df=cell_df)]
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats.rmtree"),
        ):
            result = summary_function(str(outdir), None, {}, 0, 0,
                                       qoi_functions=None, drop_columns=["does_not_exist"])
        assert "position_x" in result[0].columns

    def test_computes_qois_across_snapshots(self, tmp_path):
        mcds_list = [
            _fake_mcds(time=0.0, cell_df=_fake_cell_df(5, 0)),
            _fake_mcds(time=60.0, cell_df=_fake_cell_df(3, 2)),
        ]
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = mcds_list
        qoi_functions = {
            "live": _create_wrapper_for_qoi_function(
                lambda df_cell: len(df_cell[df_cell["dead"] == False]), "df_cell", "live"),
        }
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats._safe_rmtree"),
        ):
            df = summary_function(str(outdir), None, {"p1": 9.0}, 1, 2, qoi_functions=qoi_functions)

        assert list(df["live"]) == [5, 3]
        assert list(df["time"]) == [0.0, 60.0]
        assert (df["sampleID"] == 1).all()
        assert (df["replicateID"] == 2).all()
        assert (df["p1"] == 9.0).all()

    def test_skips_snapshot_with_no_qoi_data(self, tmp_path):
        # mcds_ts param only computes on the last snapshot -- earlier snapshots
        # produce no QoI data at all and must be skipped entirely.
        mcds_list = [_fake_mcds(time=0.0), _fake_mcds(time=1.0)]
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = mcds_list
        qoi_functions = {
            "count": _create_wrapper_for_qoi_function(lambda mcds_ts: len(mcds_ts), "mcds_ts", "count"),
        }
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats._safe_rmtree"),
        ):
            df = summary_function(str(outdir), None, {}, 0, 0, qoi_functions=qoi_functions)
        assert len(df) == 1
        assert df.iloc[0]["time"] == 1.0
        assert df.iloc[0]["count"] == 2

    def test_file_not_found_wrapped_as_runtime_error(self, tmp_path):
        outdir = tmp_path / "missing_output"
        with patch("uq_physicell.utils.sumstats.pcdl.TimeSeries",
                   side_effect=FileNotFoundError("no such file")):
            with pytest.raises(RuntimeError, match="Required file not found"):
                summary_function(str(outdir), None, {}, 0, 0, qoi_functions=None)

    def test_generic_exception_wrapped_as_runtime_error(self, tmp_path):
        outdir = tmp_path / "output"
        with patch("uq_physicell.utils.sumstats.pcdl.TimeSeries",
                   side_effect=ValueError("weird pcdl error")):
            with pytest.raises(RuntimeError, match="An error occurred while processing QoIs"):
                summary_function(str(outdir), None, {}, 0, 0, qoi_functions=None)

    def test_error_computing_qoi_raises_runtime_error(self, tmp_path):
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds()]

        def boom(df_cell):
            raise KeyError("missing_col")
        qoi_functions = {"bad": _create_wrapper_for_qoi_function(boom, "df_cell", "bad")}
        outdir = tmp_path / "output"
        outdir.mkdir()
        with patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts):
            with pytest.raises(RuntimeError, match="Error computing QoIs"):
                summary_function(str(outdir), None, {}, 0, 0, qoi_functions=qoi_functions)

    def test_remove_folder_false_skips_cleanup(self, tmp_path):
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds()]
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats._safe_rmtree") as mock_rmtree,
        ):
            summary_function(str(outdir), None, {}, 0, 0, qoi_functions={}, RemoveFolder=False)
        mock_rmtree.assert_not_called()

    def test_writes_to_summary_file_and_returns_none(self, tmp_path):
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds()]
        outdir = tmp_path / "output"
        outdir.mkdir()
        summary_file = tmp_path / "summary.tsv"
        qoi_functions = {
            "live": _create_wrapper_for_qoi_function(
                lambda df_cell: len(df_cell[df_cell["dead"] == False]), "df_cell", "live"),
        }
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts),
            patch("uq_physicell.utils.sumstats._safe_rmtree"),
        ):
            result = summary_function(str(outdir), str(summary_file), {}, 0, 0, qoi_functions=qoi_functions)
        assert result is None
        assert summary_file.exists()

    def test_no_microenv_loaded_when_only_cell_qois(self, tmp_path):
        mcds_ts = MagicMock()
        mcds_ts.get_mcds_list.return_value = [_fake_mcds()]
        qoi_functions = {
            "live": _create_wrapper_for_qoi_function(lambda df_cell: len(df_cell), "df_cell", "live"),
        }
        outdir = tmp_path / "output"
        outdir.mkdir()
        with (
            patch("uq_physicell.utils.sumstats.pcdl.TimeSeries", return_value=mcds_ts) as mock_ts,
            patch("uq_physicell.utils.sumstats._safe_rmtree"),
        ):
            summary_function(str(outdir), None, {}, 0, 0, qoi_functions=qoi_functions)
        _, kwargs = mock_ts.call_args
        assert kwargs["microenv"] is False


# ─── qoi_func_radial_density_summary ─────────────────────────────────────────

class TestQoiFuncRadialDensitySummary:
    def test_computes_expected_statistics_keys(self):
        df = pd.DataFrame({
            "position_x": [0.0, 1.0, 2.0, 3.0],
            "position_y": [0.0, 0.0, 0.0, 0.0],
            "position_z": [0.0, 0.0, 0.0, 0.0],
        })
        result = qoi_func_radial_density_summary(df)
        assert set(result.keys()) == {
            "center_of_mass", "median", "spread", "iqr", "skewness", "kurtosis",
        }
        assert result["center_of_mass"] == pytest.approx(1.5)
        assert result["median"] == pytest.approx(1.5)

    def test_custom_center_shifts_distances(self):
        df = pd.DataFrame({
            "position_x": [5.0, 5.0],
            "position_y": [0.0, 0.0],
            "position_z": [0.0, 0.0],
        })
        result = qoi_func_radial_density_summary(df, center=[5, 0, 0])
        assert result["center_of_mass"] == pytest.approx(0.0)
        assert result["spread"] == pytest.approx(0.0)

    def test_single_point_zero_spread(self):
        df = pd.DataFrame({"position_x": [1.0], "position_y": [1.0], "position_z": [1.0]})
        result = qoi_func_radial_density_summary(df)
        assert result["spread"] == pytest.approx(0.0)
        assert result["iqr"] == pytest.approx(0.0)


# ─── qoi_func_relational_ph (early-return / guard branches only) ────────────

# muspan is not on PyPI (see pyproject.toml), so it is absent on CI runners.
requires_muspan = pytest.mark.skipif(
    importlib.util.find_spec("muspan") is None or importlib.util.find_spec("gudhi") is None,
    reason="muspan and gudhi are required",
)


@requires_muspan
class TestQoiFuncRelationalPhGuards:
    def test_missing_landmark_type_returns_empty_series(self):
        df = pd.DataFrame({
            "position_x": [0.0, 1.0],
            "position_y": [0.0, 1.0],
            "position_z": [0.0, 0.0],
            "radius": [5.0, 5.0],
            "cell_type": ["B", "B"],
        })
        vec, diag = qoi_func_relational_ph(df, landmark_type="A", witness_type="B")
        assert isinstance(vec, pd.Series)
        assert vec.empty
        assert diag is None

    def test_missing_witness_type_returns_empty_series(self):
        df = pd.DataFrame({
            "position_x": [0.0, 1.0],
            "position_y": [0.0, 1.0],
            "position_z": [0.0, 0.0],
            "radius": [5.0, 5.0],
            "cell_type": ["A", "A"],
        })
        vec, diag = qoi_func_relational_ph(df, landmark_type="A", witness_type="B")
        assert vec.empty
        assert diag is None

    def test_fewer_than_three_landmarks_returns_empty_series(self):
        df = pd.DataFrame({
            "position_x": [0.0, 1.0, 0.5, 0.6],
            "position_y": [0.0, 0.0, 1.0, 1.0],
            "position_z": [0.0, 0.0, 0.0, 0.0],
            "radius": [5.0, 5.0, 5.0, 5.0],
            "cell_type": ["A", "A", "B", "B"],
        })
        vec, diag = qoi_func_relational_ph(df, landmark_type="A", witness_type="B")
        assert vec.empty
        assert diag is None


@requires_muspan
class TestQoiFuncRelationalPhComputation:
    """These exercise the real gudhi/muspan computation path (both are
    installed in this environment) rather than mocking them, since the
    early-return guards above are cheap but don't reach the actual
    filtration/vectorisation logic."""

    def _make_df(self, seed=1, n_per_type=6):
        rng = np.random.default_rng(seed)
        n = 2 * n_per_type
        return pd.DataFrame({
            "position_x": list(rng.random(n_per_type)) + list(rng.random(n_per_type)),
            "position_y": list(rng.random(n_per_type)) + list(rng.random(n_per_type)),
            "position_z": [0.0] * n,
            "radius": [0.02] * n,
            "cell_type": ["A"] * n_per_type + ["B"] * n_per_type,
        })

    def test_distance_mode_returns_populated_series(self):
        df = self._make_df()
        vec, diag = qoi_func_relational_ph(df, landmark_type="A", witness_type="B", mode="distance")
        assert isinstance(vec, pd.Series)
        assert not vec.empty
        assert isinstance(diag, list)

    def test_count_mode_runs_without_error(self):
        # "count"-mode filtrations on small random point clouds can degenerate
        # (e.g. all-zero filtration values) — vectorise_persistence then raises
        # internally and the function falls back to an empty Series. Either
        # outcome is valid; the key behavior under test is that it never
        # propagates the exception.
        df = self._make_df(seed=2)
        vec, diag = qoi_func_relational_ph(df, landmark_type="A", witness_type="B", mode="count")
        assert isinstance(vec, pd.Series)
        assert isinstance(diag, list)


class TestQoiFuncPersistentHomology:
    """Real muspan/gudhi computation (both installed here) for the happy
    path, plus the ImportError guard when muspan genuinely is unavailable."""

    @requires_muspan
    def test_returns_series_with_persistence_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "position_x": rng.random(15),
            "position_y": rng.random(15),
            "cell_type": ["A"] * 8 + ["B"] * 7,
        })
        result, fig = qoi_func_persistent_homology(df)
        assert isinstance(result, pd.Series)
        assert not result.empty
        assert fig is None

    def test_import_error_when_muspan_missing(self):
        df = pd.DataFrame({"position_x": [0.0], "position_y": [0.0], "cell_type": ["A"]})
        with patch.dict("sys.modules", {"muspan": None}):
            with pytest.raises(ImportError, match="muspan library is required"):
                qoi_func_persistent_homology(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
