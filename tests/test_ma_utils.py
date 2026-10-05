import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from uq_physicell.model_analysis.utils import (
    _reshape_sa_expanded_data,
    mcds_list_to_qoi_df_for_sa,
    mcds_list_to_qoi_df_long,
    mcds_list_to_qoi_df_for_calib,
    get_qoi_from_db_file,
    calculate_qoi_from_db_file,
    get_mean_std_qois,
    get_relative_mcse_qois,
    get_summary_statistics_qois,
    calculate_qoi_statistics,
    apply_pca_to_qois,
    apply_autoencoder_to_qois,
    regression_accuracy_parameters,
    align_params_to_qois,
    find_optimal_qoi_set,
    _regression_accuracy_with_weights,
    recursive_feature_elimination,
)
from uq_physicell.utils.sumstats import recreate_qoi_functions, _create_wrapper_for_qoi_function


class FakeMCDS:
    """Minimal stand-in for pcdl.TimeStep -- only get_time() is exercised
    by the QoI dispatch paths used in these tests."""

    def __init__(self, time):
        self._time = time

    def get_time(self):
        return self._time


# ─── _reshape_sa_expanded_data ───────────────────────────────────────────────

class TestReshapeSaExpandedData(unittest.TestCase):
    def test_basic_reshape_creates_wide_columns(self):
        data = pd.DataFrame({
            'SampleID': [0, 0, 1, 1],
            'ReplicateID': [0, 0, 0, 0],
            'time': [0, 1, 0, 1],
            'cell_count': [100, 150, 80, 120],
        })
        result = _reshape_sa_expanded_data(data, ['cell_count'])

        self.assertIn('cell_count_0', result.columns)
        self.assertIn('cell_count_1', result.columns)
        self.assertIn('time_0', result.columns)
        self.assertIn('time_1', result.columns)
        row0 = result[result['SampleID'] == 0].iloc[0]
        self.assertEqual(row0['cell_count_0'], 100)
        self.assertEqual(row0['cell_count_1'], 150)

    def test_non_numeric_qoi_values_coerced_to_nan(self):
        # A second sample provides a valid value at time_id=1 so pivot_table's
        # dropna default does not drop the whole cell_count_1 column.
        data = pd.DataFrame({
            'SampleID': [0, 0, 1, 1],
            'ReplicateID': [0, 0, 0, 0],
            'time': [0, 1, 0, 1],
            'cell_count': [100, 'not-a-number', 50, 60],
        })
        result = _reshape_sa_expanded_data(data, ['cell_count'])
        row0 = result[result['SampleID'] == 0].iloc[0]
        self.assertTrue(np.isnan(row0['cell_count_1']))

    def test_missing_grouping_columns_raises_value_error(self):
        data = pd.DataFrame({'time': [0, 1], 'cell_count': [1, 2]})
        with self.assertRaises(ValueError) as ctx:
            _reshape_sa_expanded_data(data, ['cell_count'])
        self.assertIn('Error reshaping expanded data', str(ctx.exception))


# ─── get_qoi_from_db_file ────────────────────────────────────────────────────

class TestGetQoiFromDbFile(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_flattens_and_filters_requested_qois(self, mock_load_output):
        inner_0_0 = pd.DataFrame({'time': [0, 1], 'live': [10, 12], 'dead': [1, 2]})
        inner_1_0 = pd.DataFrame({'time': [0, 1], 'live': [8, 9], 'dead': [0, 1]})
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0, 1],
            'ReplicateID': [0, 0],
            'Data': [inner_0_0, inner_1_0],
        })

        result = get_qoi_from_db_file('dummy.db', ['live', 'not_present'])

        self.assertListEqual(list(result.columns), ['SampleID', 'time', 'ReplicateID', 'live'])
        self.assertEqual(len(result), 4)
        # sorted by SampleID, time, ReplicateID
        self.assertTrue((result['SampleID'].values == sorted(result['SampleID'].values)).all())
        sample0 = result[result['SampleID'] == 0].sort_values('time')
        self.assertListEqual(list(sample0['live']), [10, 12])


# ─── mcds_list_to_qoi_df_for_sa / long / for_calib ──────────────────────────

def _qoi_funcs(**name_to_lambda):
    return recreate_qoi_functions(qoi_functions=name_to_lambda)


class TestMcdsListToQoiDfForSA(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_wide_format_with_uneven_time_series(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0, 1],
            'ReplicateID': [0, 0],
            'Data': [
                [FakeMCDS(0.0), FakeMCDS(1.0)],
                [FakeMCDS(0.0)],
            ],
        })
        recreated = _qoi_funcs(metric=lambda mcds: mcds.get_time() * 10)

        result = mcds_list_to_qoi_df_for_sa(recreated, [0, 1], chunk_size=10, db_file='dummy.db')

        self.assertEqual(len(result), 2)
        row0 = result[result['SampleID'] == 0].iloc[0]
        self.assertEqual(row0['metric_0'], 0.0)
        self.assertEqual(row0['metric_1'], 10.0)
        row1 = result[result['SampleID'] == 1].iloc[0]
        self.assertEqual(row1['metric_0'], 0.0)
        # sample 1 only had one snapshot -> metric_1 column exists (from sample0) but is NaN here
        self.assertTrue(pd.isna(row1['metric_1']))


class TestMcdsListToQoiDfForCalib(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_long_format_one_row_per_snapshot(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })
        recreated = _qoi_funcs(metric=lambda mcds: mcds.get_time() + 1)

        result = mcds_list_to_qoi_df_for_calib(recreated, [0], chunk_size=10, db_file='dummy.db')

        self.assertEqual(len(result), 2)
        self.assertListEqual(sorted(result['time'].tolist()), [0.0, 1.0])
        self.assertListEqual(sorted(result['metric'].tolist()), [1.0, 2.0])

    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_snapshot_with_no_qoi_values_is_skipped(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })
        # returns None for time==1.0, so that snapshot contributes no row
        recreated = _qoi_funcs(metric=lambda mcds: None if mcds.get_time() == 1.0 else 99.0)

        result = mcds_list_to_qoi_df_for_calib(recreated, [0], chunk_size=10, db_file='dummy.db')

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['time'], 0.0)


class TestMcdsListToQoiDfLong(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_single_valued_qoi_long_format(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })
        recreated = _qoi_funcs(metric=lambda mcds: mcds.get_time())

        result = mcds_list_to_qoi_df_long(recreated, [0], chunk_size=10, db_file='dummy.db')

        self.assertListEqual(list(result.columns), ['SampleID', 'time', 'ReplicateID', 'metric'])
        self.assertListEqual(result['metric'].tolist(), [0.0, 1.0])

    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_dict_valued_qoi_expands_columns_and_pads_missing_keys(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })

        def multi_qoi(mcds):
            if mcds.get_time() == 0.0:
                return {'a': 1, 'b': 2}
            return {}

        wrapped = _create_wrapper_for_qoi_function(func=multi_qoi, param_name='mcds', qoi_name='multi')
        result = mcds_list_to_qoi_df_long({'multi': wrapped}, [0], chunk_size=10, db_file='dummy.db')

        self.assertIn('multi_a', result.columns)
        self.assertIn('multi_b', result.columns)
        row_t0 = result[result['time'] == 0.0].iloc[0]
        row_t1 = result[result['time'] == 1.0].iloc[0]
        self.assertEqual(row_t0['multi_a'], 1)
        self.assertEqual(row_t0['multi_b'], 2)
        self.assertTrue(pd.isna(row_t1['multi_a']))
        self.assertTrue(pd.isna(row_t1['multi_b']))

    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_wrapped_timeseries_dataframe_expands_by_time(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })

        def many_per_timestep(mcds_ts):
            return pd.DataFrame({
                'time': [1.0, 0.0],
                'a': [11.0, 10.0],
                'b': [21.0, 20.0],
            })

        wrapped = _create_wrapper_for_qoi_function(
            func=many_per_timestep,
            param_name='mcds_ts',
            qoi_name='perplexity',
        )
        result = mcds_list_to_qoi_df_long(
            {'perplexity': wrapped},
            [0],
            chunk_size=10,
            db_file='dummy.db',
        )

        self.assertNotIn('perplexity', result.columns)
        self.assertIn('perplexity_a', result.columns)
        self.assertIn('perplexity_b', result.columns)
        self.assertListEqual(result['perplexity_a'].tolist(), [10.0, 11.0])
        self.assertListEqual(result['perplexity_b'].tolist(), [20.0, 21.0])

    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_wrapped_timeseries_scalar_and_scalar_mapping_broadcast(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })
        recreated = _qoi_funcs(
            series_count=lambda mcds_ts: len(mcds_ts),
            summary=lambda mcds_ts: {'auc': 12.0, 'peak': 7.0},
        )

        result = mcds_list_to_qoi_df_long(recreated, [0], chunk_size=10, db_file='dummy.db')

        self.assertListEqual(result['series_count'].tolist(), [2, 2])
        self.assertListEqual(result['summary_auc'].tolist(), [12.0, 12.0])
        self.assertListEqual(result['summary_peak'].tolist(), [7.0, 7.0])

    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_wrapped_timeseries_mapping_sequences_use_mapping_keys(self, mock_load_output):
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0],
            'ReplicateID': [0],
            'Data': [[FakeMCDS(0.0), FakeMCDS(1.0)]],
        })
        recreated = _qoi_funcs(
            trajectory=lambda mcds_ts: {'live': [1, 2], 'dead': [0, 1]},
        )

        result = mcds_list_to_qoi_df_long(recreated, [0], chunk_size=10, db_file='dummy.db')

        self.assertIn('trajectory_live', result.columns)
        self.assertIn('trajectory_dead', result.columns)
        self.assertNotIn('trajectory_key', result.columns)
        self.assertListEqual(result['trajectory_live'].tolist(), [1, 2])
        self.assertListEqual(result['trajectory_dead'].tolist(), [0, 1])


class TestCalculateQoiFromDbFile(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.load_output')
    @patch('uq_physicell.model_analysis.utils.load_samples')
    def test_mode_dispatch_sa_long_calib(self, mock_load_samples, mock_load_output):
        mock_load_samples.return_value = {0: {}, 1: {}}
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0, 1],
            'ReplicateID': [0, 0],
            'Data': [[FakeMCDS(0.0)], [FakeMCDS(0.0)]],
        })
        qoi_functions = {'metric': 'lambda mcds: mcds.get_time()'}

        df_sa = calculate_qoi_from_db_file('dummy.db', qoi_functions, mode='sa')
        df_long = calculate_qoi_from_db_file('dummy.db', qoi_functions, mode='long')
        df_calib = calculate_qoi_from_db_file('dummy.db', qoi_functions, mode='calib')

        self.assertIn('metric_0', df_sa.columns)
        self.assertIn('metric', df_long.columns)
        self.assertIn('metric', df_calib.columns)

    @patch('uq_physicell.model_analysis.utils.load_output')
    @patch('uq_physicell.model_analysis.utils.load_samples')
    def test_unknown_mode_raises_value_error(self, mock_load_samples, mock_load_output):
        mock_load_samples.return_value = {0: {}}
        mock_load_output.return_value = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [[FakeMCDS(0.0)]],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_from_db_file('dummy.db', {'metric': 'lambda mcds: 1'}, mode='bogus')
        self.assertIn('Unknown mode', str(ctx.exception))


# ─── get_mean_std_qois / get_relative_mcse_qois / get_summary_statistics_qois

class TestGetMeanStdQois(unittest.TestCase):
    def test_groups_by_sample_and_drops_replicate_id(self):
        df_qois = pd.DataFrame({
            'SampleID': [0, 0, 1, 1],
            'ReplicateID': [0, 1, 0, 1],
            'qoi': [10.0, 20.0, 100.0, 200.0],
        })
        df_mean, df_std = get_mean_std_qois(df_qois)

        self.assertNotIn('ReplicateID', df_mean.columns)
        self.assertNotIn('ReplicateID', df_std.columns)
        self.assertAlmostEqual(df_mean.loc[0, 'qoi'], 15.0)
        self.assertAlmostEqual(df_mean.loc[1, 'qoi'], 150.0)
        self.assertAlmostEqual(df_std.loc[0, 'qoi'], np.std([10.0, 20.0], ddof=1))

    def test_filter_columns_adds_extra_grouping_key(self):
        df_qois = pd.DataFrame({
            'SampleID': [0, 0, 0, 0],
            'ReplicateID': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'qoi': [10.0, 20.0, 12.0, 22.0],
        })
        df_mean, df_std = get_mean_std_qois(df_qois, filter_columns=['time'])
        self.assertAlmostEqual(df_mean.loc[(0, 0), 'qoi'], 11.0)
        self.assertAlmostEqual(df_mean.loc[(0, 1), 'qoi'], 21.0)


class TestGetRelativeMcseQois(unittest.TestCase):
    def test_relative_mcse_formula(self):
        df_mean = pd.DataFrame({'qoi': [10.0, 20.0]})
        df_std = pd.DataFrame({'qoi': [1.0, 2.0]})
        result = get_relative_mcse_qois(df_mean, df_std, num_replicates=4, time_columns=[])

        epsilon = max(0.01 * np.nanmedian(np.abs(df_mean.to_numpy()).flatten()), 1e-12)
        expected_0 = (1.0 / np.sqrt(4)) / (10.0 + epsilon)
        expected_1 = (2.0 / np.sqrt(4)) / (20.0 + epsilon)
        self.assertAlmostEqual(result['qoi'].iloc[0], expected_0)
        self.assertAlmostEqual(result['qoi'].iloc[1], expected_1)

    def test_time_columns_are_restored_from_mean(self):
        df_mean = pd.DataFrame({'time_0': [0.0, 0.0], 'qoi_0': [10.0, 20.0]})
        df_std = pd.DataFrame({'time_0': [0.0, 0.0], 'qoi_0': [1.0, 2.0]})
        result = get_relative_mcse_qois(df_mean, df_std, num_replicates=2, time_columns=['time_0'])
        self.assertListEqual(list(result['time_0']), [0.0, 0.0])


class TestGetSummaryStatisticsQois(unittest.TestCase):
    def test_raises_when_no_time_information_present(self):
        df_qois = pd.DataFrame({
            'SampleID': [0, 0], 'ReplicateID': [0, 1], 'qoi': [1.0, 2.0],
        })
        with self.assertRaises(ValueError) as ctx:
            get_summary_statistics_qois(df_qois)
        self.assertIn('No time columns found', str(ctx.exception))

    def test_long_format_with_time_column(self):
        df_qois = pd.DataFrame({
            'SampleID': [0, 0, 0, 0],
            'ReplicateID': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'qoi': [10.0, 20.0, 12.0, 22.0],
        })
        df_mean, df_std, df_mcse = get_summary_statistics_qois(df_qois)
        self.assertAlmostEqual(df_mean.loc[(0, 0), 'qoi'], 11.0)
        self.assertAlmostEqual(df_mean.loc[(0, 1), 'qoi'], 21.0)
        self.assertFalse(df_std.isna().all().all())

    def test_wide_format_with_time_prefixed_columns(self):
        df_qois = pd.DataFrame({
            'SampleID': [0, 0],
            'ReplicateID': [0, 1],
            'time_0': [0.0, 0.0],
            'qoi_0': [10.0, 12.0],
        })
        df_mean, df_std, df_mcse = get_summary_statistics_qois(df_qois)
        # time columns should be copied straight from df_mean, not divided
        self.assertListEqual(list(df_mcse['time_0']), list(df_mean['time_0']))


# ─── calculate_qoi_statistics ────────────────────────────────────────────────

class TestCalculateQoiStatistics(unittest.TestCase):
    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    def test_dataframe_without_data_column_used_directly(self, mock_check):
        df_qois_data = pd.DataFrame({
            'SampleID': [0, 0, 0, 0],
            'ReplicateID': [0, 0, 1, 1],
            'time': [0, 1, 0, 1],
            'qoi': [10.0, 20.0, 12.0, 22.0],
        })
        df_mean, df_std, df_mcse = calculate_qoi_statistics(
            'dummy.db', qoi_funcs={}, df_qois_data=df_qois_data,
        )
        self.assertAlmostEqual(df_mean.loc[(0, 0), 'qoi'], 11.0)

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=False)
    def test_db_inconsistency_raises_by_default(self, mock_check):
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'time': [0], 'qoi': [1.0],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={}, df_qois_data=df_qois_data)
        self.assertIn('consistency check failed', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.load_output', side_effect=RuntimeError('disk error'))
    def test_load_output_failure_wrapped_in_value_error(self, mock_load_output):
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={})
        self.assertIn('Error loading output data from database', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    @patch('uq_physicell.model_analysis.utils.load_output')
    def test_no_data_provided_and_metadata_only_frame_fails_downstream(self, mock_load_output, mock_check):
        # load_data=False path -> no 'Data' column and no 'time' column either,
        # so get_summary_statistics_qois raises and gets wrapped.
        mock_load_output.return_value = pd.DataFrame({'SampleID': [0], 'ReplicateID': [0]})
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={})
        mock_load_output.assert_called_once_with('dummy.db', load_data=False)
        self.assertIn('Error taking the mean', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    def test_data_column_present_but_no_qoi_funcs_raises(self, mock_check):
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [pd.DataFrame({'time': [0]})],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={}, df_qois_data=df_qois_data)
        self.assertIn('No QoI functions defined', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    @patch('uq_physicell.model_analysis.utils.get_qoi_from_db_file')
    def test_dataframe_mode_a_data_extracted_via_helper(self, mock_get_qoi, mock_check):
        mock_get_qoi.return_value = pd.DataFrame({
            'SampleID': [0, 0, 0, 0],
            'time': [0, 1, 0, 1],
            'ReplicateID': [0, 0, 1, 1],
            'qoi': [10.0, 20.0, 12.0, 22.0],
        })
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [pd.DataFrame({'time': [0], 'qoi': [10.0]})],
        })
        df_mean, df_std, df_mcse = calculate_qoi_statistics(
            'dummy.db', qoi_funcs={'qoi': None}, df_qois_data=df_qois_data,
        )
        mock_get_qoi.assert_called_once_with('dummy.db', ['qoi'])
        self.assertAlmostEqual(df_mean.loc[(0, 0), 'qoi'], 11.0)

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    @patch('uq_physicell.model_analysis.utils.calculate_qoi_from_db_file')
    def test_list_mode_empty_result_raises(self, mock_calc, mock_check):
        mock_calc.return_value = pd.DataFrame()
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [[object()]],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={'qoi': None}, df_qois_data=df_qois_data)
        self.assertIn('df_qois is empty', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    @patch('uq_physicell.model_analysis.utils.calculate_qoi_from_db_file', side_effect=RuntimeError('boom'))
    def test_list_mode_exception_wrapped(self, mock_calc, mock_check):
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [[object()]],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={'qoi': None}, df_qois_data=df_qois_data)
        self.assertIn('Error calculating QoIs from mcds list', str(ctx.exception))

    @patch('uq_physicell.model_analysis.utils.check_db_consistency', return_value=True)
    def test_data_column_of_unsupported_type_raises(self, mock_check):
        df_qois_data = pd.DataFrame({
            'SampleID': [0], 'ReplicateID': [0], 'Data': [{'not': 'a dataframe or list'}],
        })
        with self.assertRaises(ValueError) as ctx:
            calculate_qoi_statistics('dummy.db', qoi_funcs={'qoi': None}, df_qois_data=df_qois_data)
        self.assertIn('neither Dataframe nor List', str(ctx.exception))


# ─── apply_pca_to_qois ────────────────────────────────────────────────────────

class TestApplyPcaToQois(unittest.TestCase):
    def test_shapes_and_keys(self):
        rng = np.random.default_rng(0)
        df_mean = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f'q{i}' for i in range(5)])
        result = apply_pca_to_qois(df_mean, latent_dim=3, seed=0)

        self.assertEqual(result['method'], 'pca')
        self.assertEqual(result['encoder_output'].shape, (20, 3))
        self.assertEqual(result['reconstruction'].shape, (20, 5))

    def test_latent_dim_clamped_to_num_features(self):
        df_mean = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [3.0, 2.0, 1.0]})
        result = apply_pca_to_qois(df_mean, latent_dim=5, seed=0)
        self.assertEqual(result['encoder_output'].shape[1], 2)


# ─── apply_autoencoder_to_qois ────────────────────────────────────────────────

class TestApplyAutoencoderToQois(unittest.TestCase):
    def _small_df(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame(rng.normal(size=(6, 3)), columns=['a', 'b', 'c'])

    def test_torch_path_reproducible_with_seed(self):
        df_mean = self._small_df()
        kwargs = dict(latent_dim=1, epochs=2, batch_size=4, seed=42)
        result1 = apply_autoencoder_to_qois(df_mean, **kwargs)
        result2 = apply_autoencoder_to_qois(df_mean, **kwargs)

        if result1 is None:
            self.skipTest('torch not available in this environment')

        self.assertEqual(result1['method'], 'torch')
        self.assertEqual(result1['encoder_output'].shape, (6, 1))
        self.assertEqual(result1['reconstruction'].shape, (6, 3))
        np.testing.assert_allclose(result1['encoder_output'], result2['encoder_output'])

    def test_returns_none_when_torch_unavailable(self):
        df_mean = self._small_df()
        with patch.dict(sys.modules, {'torch': None}):
            result = apply_autoencoder_to_qois(df_mean, latent_dim=1, epochs=1, batch_size=4)
        self.assertIsNone(result)


# ─── regression_accuracy_parameters ──────────────────────────────────────────

class TestRegressionAccuracyParameters(unittest.TestCase):
    def test_returns_r2_per_parameter(self):
        rng = np.random.default_rng(2)
        encoder_output = rng.normal(size=(15, 2))
        df_parameters = pd.DataFrame({
            'p1': encoder_output[:, 0] * 2.0,
            'p2': rng.normal(size=15),
        })
        result = regression_accuracy_parameters(df_parameters, encoder_output)

        self.assertListEqual(list(result.index), ['p1', 'p2'])
        self.assertListEqual(list(result.columns), ['R2_CV'])
        self.assertTrue(np.isfinite(result['R2_CV']).all())


# ─── align_params_to_qois ─────────────────────────────────────────────────────

class TestAlignParamsToQois(unittest.TestCase):
    def test_flat_index_encodes_categorical_columns(self):
        df_params = pd.DataFrame({'cat': ['low', 'high', 'low'], 'num': [1.0, 2.0, 3.0]},
                                  index=pd.Index([0, 1, 2], name='SampleID'))
        df_qois = pd.DataFrame({'qoi': [1, 2, 3]}, index=pd.Index([0, 1, 2], name='SampleID'))

        result = align_params_to_qois(df_params, df_qois)

        self.assertTrue(np.issubdtype(result['cat'].dtype, np.integer))
        self.assertListEqual(list(result['num']), [1.0, 2.0, 3.0])

    def test_multiindex_time_expands_params_per_timepoint(self):
        df_params = pd.DataFrame({'p1': [10.0, 20.0]}, index=pd.Index([0, 1], name='SampleID'))
        idx = pd.MultiIndex.from_tuples([(0, 0.0), (0, 1.0), (1, 0.0)], names=['SampleID', 'time'])
        df_qois = pd.DataFrame({'qoi': [1, 2, 3]}, index=idx)

        result = align_params_to_qois(df_params, df_qois)

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[(0, 0.0), 'p1'], 10.0)
        self.assertEqual(result.loc[(0, 1.0), 'p1'], 10.0)
        self.assertEqual(result.loc[(1, 0.0), 'p1'], 20.0)

    def test_sample_id_missing_from_params_is_skipped(self):
        df_params = pd.DataFrame({'p1': [10.0]}, index=pd.Index([0], name='SampleID'))
        idx = pd.MultiIndex.from_tuples([(0, 0.0), (1, 0.0)], names=['SampleID', 'time'])
        df_qois = pd.DataFrame({'qoi': [1, 2]}, index=idx)

        result = align_params_to_qois(df_params, df_qois)
        self.assertEqual(len(result), 1)
        self.assertIn((0, 0.0), result.index)


# ─── find_optimal_qoi_set (RFECV mocked -- heavy fit isolated) ──────────────

class TestFindOptimalQoiSet(unittest.TestCase):
    @patch('sklearn.feature_selection.RFECV')
    def test_delegates_to_rfecv_and_returns_selector(self, mock_rfecv_cls):
        mock_selector = MagicMock()
        mock_selector.support_ = np.array([True, False])
        mock_rfecv_cls.return_value = mock_selector

        df_qois = pd.DataFrame({'q1': [1.0, 2.0, 3.0], 'q2': [3.0, 2.0, 1.0]})
        df_params = pd.DataFrame({'p1': [1.0, 2.0, 3.0]})

        result = find_optimal_qoi_set(df_qois, df_params)

        self.assertIs(result, mock_selector)
        mock_selector.fit.assert_called_once()
        fit_args = mock_selector.fit.call_args[0]
        np.testing.assert_array_equal(fit_args[0], df_qois.sort_index().to_numpy())
        np.testing.assert_array_equal(fit_args[1], df_params.sort_index().to_numpy())


# ─── _regression_accuracy_with_weights ───────────────────────────────────────

class TestRegressionAccuracyWithWeights(unittest.TestCase):
    def test_uniform_and_weighted_paths_both_run(self):
        rng = np.random.default_rng(3)
        encoder_output = rng.normal(size=(12, 2))
        df_parameters = pd.DataFrame({'p1': rng.normal(size=12)})

        result_uniform = _regression_accuracy_with_weights(df_parameters, encoder_output, mcse_weights=None)
        weights = np.ones(12)
        result_weighted = _regression_accuracy_with_weights(df_parameters, encoder_output, mcse_weights=weights)

        self.assertListEqual(list(result_uniform.columns), ['R2_CV'])
        self.assertListEqual(list(result_weighted.columns), ['R2_CV'])


# ─── recursive_feature_elimination (AE + RFECV mocked) ───────────────────────

class TestRecursiveFeatureElimination(unittest.TestCase):
    def _fake_ae(self, df, **kwargs):
        return {
            'method': 'fake',
            'encoder_output': df.to_numpy(dtype=float),
            'reconstruction': df.to_numpy(dtype=float),
            'model': None,
            'scaler': None,
        }

    def test_stage1_and_stage2_filtering_then_rfecv_selection(self):
        n = 8
        df_qois_mean = pd.DataFrame({
            'qoiA': [0, 1, 0, 1, 0, 1, 0, 1],  # uncorrelated with qoiC/qoiD
            'qoiB': np.linspace(0, 1, n),
            'qoiC': np.linspace(10, 20, n),
            'qoiD': np.linspace(10, 20, n),  # perfectly correlated with qoiC
        }, index=pd.Index(range(n), name='SampleID'))
        df_qois_mcse = pd.DataFrame({
            'qoiA': [0.01] * n,
            'qoiB': [0.2] * n,   # extreme noise -> removed in stage 1
            'qoiC': [0.02] * n,  # lower median -> kept over qoiD
            'qoiD': [0.05] * n,
        }, index=df_qois_mean.index)
        df_params = pd.DataFrame({'p1': np.linspace(0, 1, n)}, index=df_qois_mean.index)

        mock_selector = MagicMock()
        mock_selector.support_ = np.array([True, False])  # keep qoiA, drop qoiC

        with patch('uq_physicell.model_analysis.utils.apply_autoencoder_to_qois', side_effect=self._fake_ae), \
             patch('sklearn.feature_selection.RFECV', return_value=mock_selector):
            result = recursive_feature_elimination(
                df_qois_mean, df_qois_mcse, df_params,
                autoencoder_params={'latent_dim': 1, 'epochs': 1, 'batch_size': 4, 'seed': 0},
                mcse_threshold=0.10, correlation_threshold=0.95, verbose=False,
            )

        self.assertListEqual(result['removed_extreme_noise'], ['qoiB'])
        self.assertListEqual(result['removed_redundant'], ['qoiD'])
        self.assertListEqual(result['cleaned_qois'], ['qoiA', 'qoiC'])
        self.assertListEqual(result['final_qois'], ['qoiA'])
        self.assertIn('R2_CV', result['full_regression_r2'].columns)
        self.assertIn('R2_CV', result['reduced_regression_r2'].columns)
        self.assertListEqual(
            list(result['synthetic_recovery_test'].columns),
            ['Full_R2', 'Reduced_R2', 'R2_Drop'],
        )

    def test_trajectory_nan_filled_with_last_value_per_sample(self):
        # 3 samples x 2 timepoints = 6 rows, enough for the function's cv=5 folds.
        idx = pd.MultiIndex.from_tuples(
            [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0)],
            names=['SampleID', 'time'],
        )
        df_qois_mean = pd.DataFrame({
            'traj': [np.nan, 7.0, np.nan, 9.0, np.nan, 11.0],
            'stable': [3.0, 9.0, 1.0, 4.0, 5.0, 2.0],
        }, index=idx)
        df_qois_mcse = pd.DataFrame({
            'traj': [np.nan, 0.01, np.nan, 0.01, np.nan, 0.01],
            'stable': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        }, index=idx)
        df_params = pd.DataFrame({'p1': [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], name='SampleID'))

        mock_selector = MagicMock()
        mock_selector.support_ = np.array([True, True])

        with patch('uq_physicell.model_analysis.utils.apply_autoencoder_to_qois', side_effect=self._fake_ae), \
             patch('sklearn.feature_selection.RFECV', return_value=mock_selector):
            result = recursive_feature_elimination(
                df_qois_mean, df_qois_mcse, df_params,
                autoencoder_params={'latent_dim': 2, 'epochs': 1, 'batch_size': 4, 'seed': 0},
                mcse_threshold=0.10, correlation_threshold=0.95, verbose=False,
            )

        self.assertListEqual(result['removed_extreme_noise'], [])
        self.assertListEqual(result['removed_redundant'], [])
        filled_traj = result['full_autoencoder_results']['encoder_output'][:, result['cleaned_qois'].index('traj')]
        np.testing.assert_allclose(sorted(filled_traj), [7.0, 7.0, 9.0, 9.0, 11.0, 11.0])

    def test_all_nan_rows_raise_value_error(self):
        idx = pd.Index(range(3), name='SampleID')
        df_qois_mean = pd.DataFrame({'qoi': [np.nan, np.nan, np.nan]}, index=idx)
        df_qois_mcse = pd.DataFrame({'qoi': [0.01, 0.01, 0.01]}, index=idx)
        df_params = pd.DataFrame({'p1': [1.0, 2.0, 3.0]}, index=idx)

        with self.assertRaises(ValueError) as ctx:
            recursive_feature_elimination(df_qois_mean, df_qois_mcse, df_params)
        self.assertIn('No valid QoI data remaining', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
