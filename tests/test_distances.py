import unittest
import numpy as np
import pandas as pd

# Import the distance functions to test
from uq_physicell.utils import SumSquaredDifferences, Manhattan, Chebyshev, relative_rmse, chi2_distance


class TestDistanceFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing distance functions
        self.dic_model_data = {
            "time": np.array([0, 1, 2, 3, 4]),
            "value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        self.dic_obs_data = {
            "time": np.array([0, 1, 2, 3, 4]),
            "value": np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        }
        
        # Data with partial overlap in time points
        self.dic_model_data_partial = {
            "time": np.array([0, 1, 2, 5, 6]),
            "value": np.array([1.0, 2.0, 3.0, 6.0, 7.0])
        }
        
        self.dic_obs_data_partial = {
            "time": np.array([0, 1, 2, 3, 4]),
            "value": np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        }
        
        # Data with no overlap in time points
        self.dic_model_data_no_overlap = {
            "time": np.array([5, 6, 7, 8, 9]),
            "value": np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        }

    def test_none_model_data_returns_inf(self):
        """A failed simulation (CalibrationContext._run_physicell_model returns
        None) must reject the particle via np.inf, not crash on None["time"]."""
        self.assertEqual(SumSquaredDifferences(None, self.dic_obs_data), np.inf)
        self.assertEqual(Manhattan(None, self.dic_obs_data), np.inf)
        self.assertEqual(Chebyshev(None, self.dic_obs_data), np.inf)

    def test_sum_squared_differences_perfect_match(self):
        """Test SumSquaredDifferences with identical time points."""
        result = SumSquaredDifferences(self.dic_model_data, self.dic_obs_data)
        
        # Calculate expected result
        # differences: [1.0-1.1, 2.0-1.9, 3.0-3.2, 4.0-3.8, 5.0-5.1] = [-0.1, 0.1, -0.2, 0.2, -0.1]
        # squared: [0.01, 0.01, 0.04, 0.04, 0.01]
        # sum: 0.11
        expected = 0.11
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_sum_squared_differences_partial_overlap(self):
        """Test SumSquaredDifferences with partial overlap in time points."""
        result = SumSquaredDifferences(self.dic_model_data_partial, self.dic_obs_data_partial)
        
        # Common time points: [0, 1, 2]
        # Model values at [0, 1, 2]: [1.0, 2.0, 3.0]
        # Obs values at [0, 1, 2]: [1.1, 1.9, 3.2]
        # differences: [-0.1, 0.1, -0.2]
        # squared: [0.01, 0.01, 0.04]
        # sum: 0.06
        expected = 0.06
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_sum_squared_differences_no_overlap(self):
        """Test SumSquaredDifferences with no overlap in time points."""
        with self.assertRaises(ValueError) as context:
            SumSquaredDifferences(self.dic_model_data_no_overlap, self.dic_obs_data)
        
        self.assertIn("No matching time points found", str(context.exception))

    def test_manhattan_distance_perfect_match(self):
        """Test Manhattan distance with identical time points."""
        result = Manhattan(self.dic_model_data, self.dic_obs_data)
        
        # Calculate expected result
        # differences: [-0.1, 0.1, -0.2, 0.2, -0.1]
        # absolute values: [0.1, 0.1, 0.2, 0.2, 0.1]
        # sum: 0.7
        expected = 0.7
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_manhattan_distance_partial_overlap(self):
        """Test Manhattan distance with partial overlap in time points."""
        result = Manhattan(self.dic_model_data_partial, self.dic_obs_data_partial)
        
        # Common time points: [0, 1, 2]
        # differences: [-0.1, 0.1, -0.2]
        # absolute values: [0.1, 0.1, 0.2]
        # sum: 0.4
        expected = 0.4
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_manhattan_distance_no_overlap(self):
        """Test Manhattan distance with no overlap in time points."""
        with self.assertRaises(ValueError) as context:
            Manhattan(self.dic_model_data_no_overlap, self.dic_obs_data)
        
        self.assertIn("No matching time points found", str(context.exception))

    def test_chebyshev_distance_perfect_match(self):
        """Test Chebyshev distance with identical time points."""
        result = Chebyshev(self.dic_model_data, self.dic_obs_data)
        
        # Calculate expected result
        # differences: [-0.1, 0.1, -0.2, 0.2, -0.1]
        # absolute values: [0.1, 0.1, 0.2, 0.2, 0.1]
        # max: 0.2
        expected = 0.2
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_chebyshev_distance_partial_overlap(self):
        """Test Chebyshev distance with partial overlap in time points."""
        result = Chebyshev(self.dic_model_data_partial, self.dic_obs_data_partial)
        
        # Common time points: [0, 1, 2]
        # differences: [-0.1, 0.1, -0.2]
        # absolute values: [0.1, 0.1, 0.2]
        # max: 0.2
        expected = 0.2
        
        self.assertAlmostEqual(result, expected, places=10)

    def test_chebyshev_distance_no_overlap(self):
        """Test Chebyshev distance with no overlap in time points."""
        with self.assertRaises(ValueError) as context:
            Chebyshev(self.dic_model_data_no_overlap, self.dic_obs_data)
        
        self.assertIn("No matching time points found", str(context.exception))

    def test_identical_data(self):
        """Test all distance functions with identical model and observed data."""
        identical_obs_data = {
            "time": self.dic_model_data["time"].copy(),
            "value": self.dic_model_data["value"].copy()
        }
        
        # All distances should be zero for identical data
        self.assertEqual(SumSquaredDifferences(self.dic_model_data, identical_obs_data), 0.0)
        self.assertEqual(Manhattan(self.dic_model_data, identical_obs_data), 0.0)
        self.assertEqual(Chebyshev(self.dic_model_data, identical_obs_data), 0.0)

    def test_single_time_point(self):
        """Test distance functions with single time point."""
        single_model = {
            "time": np.array([1]),
            "value": np.array([2.0])
        }
        
        single_obs = {
            "time": np.array([1]),
            "value": np.array([2.5])
        }
        
        # Expected difference: 2.0 - 2.5 = -0.5
        # SSD: 0.25, Manhattan: 0.5, Chebyshev: 0.5
        self.assertEqual(SumSquaredDifferences(single_model, single_obs), 0.25)
        self.assertEqual(Manhattan(single_model, single_obs), 0.5)
        self.assertEqual(Chebyshev(single_model, single_obs), 0.5)

    def test_empty_overlap_handling(self):
        """Test that functions correctly handle case with no common time points."""
        # Data with no overlapping time points - should raise ValueError
        no_overlap_model = {
            "time": np.array([10, 11, 12]),
            "value": np.array([1.0, 2.0, 3.0])
        }
        
        with self.assertRaises(ValueError):
            SumSquaredDifferences(no_overlap_model, self.dic_obs_data)
        
        with self.assertRaises(ValueError):
            Manhattan(no_overlap_model, self.dic_obs_data)
        
        with self.assertRaises(ValueError):
            Chebyshev(no_overlap_model, self.dic_obs_data)

    def test_float_time_points(self):
        """Test distance functions with non-integer time points."""
        float_model = {
            "time": np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            "value": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        }
        
        float_obs = {
            "time": np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
            "value": np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        }
        
        # Should work the same as integer time points
        result_ssd = SumSquaredDifferences(float_model, float_obs)
        result_manhattan = Manhattan(float_model, float_obs)
        result_chebyshev = Chebyshev(float_model, float_obs)
        
        # Values should be computed correctly
        self.assertAlmostEqual(result_ssd, 0.11, places=10)
        self.assertAlmostEqual(result_manhattan, 0.7, places=10)
        self.assertAlmostEqual(result_chebyshev, 0.2, places=10)

    def test_large_differences(self):
        """Test distance functions with large value differences."""
        large_diff_obs = {
            "time": self.dic_model_data["time"].copy(),
            "value": np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        }
        
        # Model: [1, 2, 3, 4, 5], Obs: [10, 20, 30, 40, 50]
        # Differences: [-9, -18, -27, -36, -45]
        
        result_ssd = SumSquaredDifferences(self.dic_model_data, large_diff_obs)
        result_manhattan = Manhattan(self.dic_model_data, large_diff_obs)
        result_chebyshev = Chebyshev(self.dic_model_data, large_diff_obs)
        
        # SSD: 81 + 324 + 729 + 1296 + 2025 = 4455
        # Manhattan: 9 + 18 + 27 + 36 + 45 = 135
        # Chebyshev: max(9, 18, 27, 36, 45) = 45
        
        self.assertEqual(result_ssd, 4455)
        self.assertEqual(result_manhattan, 135)
        self.assertEqual(result_chebyshev, 45)

    def test_negative_values(self):
        """Test distance functions with negative values."""
        negative_model = {
            "time": np.array([0, 1, 2]),
            "value": np.array([-1.0, -2.0, -3.0])
        }
        
        negative_obs = {
            "time": np.array([0, 1, 2]),
            "value": np.array([-1.5, -1.8, -3.2])
        }
        
        # Differences: [-1.0-(-1.5), -2.0-(-1.8), -3.0-(-3.2)] = [0.5, -0.2, 0.2]
        
        result_ssd = SumSquaredDifferences(negative_model, negative_obs)
        result_manhattan = Manhattan(negative_model, negative_obs)
        result_chebyshev = Chebyshev(negative_model, negative_obs)
        
        # SSD: 0.25 + 0.04 + 0.04 = 0.33
        # Manhattan: 0.5 + 0.2 + 0.2 = 0.9
        # Chebyshev: max(0.5, 0.2, 0.2) = 0.5
        
        self.assertAlmostEqual(result_ssd, 0.33, places=10)
        self.assertAlmostEqual(result_manhattan, 0.9, places=10)
        self.assertAlmostEqual(result_chebyshev, 0.5, places=10)


class TestRelativeRMSE(unittest.TestCase):

    def setUp(self):
        self.obs = {"QoI1": np.array([1.0, 2.0, 3.0, 4.0])}

    def test_none_sim_returns_inf(self):
        """A failed simulation (sim=None) must be rejected via np.inf, matching
        the convention used by SumSquaredDifferences/Manhattan/Chebyshev."""
        self.assertEqual(relative_rmse(None, self.obs, "QoI1"), np.inf)

    def test_plain_array_matches_hand_computed_value(self):
        sim = {"QoI1": np.array([1.5, 2.0, 2.0, 5.0])}
        # diffs: -0.5, 0, 1, -1; denom = max(|obs|,1) = [1,2,3,4]
        # terms: -0.5, 0, 0.333..., -0.25 -> mean(sq) = 0.10590277...
        result = relative_rmse(sim, self.obs, "QoI1")
        self.assertAlmostEqual(result, 0.3254270698294439, places=10)

    def test_dataframe_reconstructed_from_storage_matches_plain_array(self):
        """pyABC reconstructs historical particles from storage as a DataFrame
        with ['time', key] columns rather than the plain 1-D array a live
        simulation produces for the same key; both shapes must give the same
        result once the key column is pulled out."""
        plain_sim = {"QoI1": np.array([1.5, 2.0, 2.0, 5.0])}
        df_sim = {"QoI1": pd.DataFrame({"time": [0, 1, 2, 3], "QoI1": [1.5, 2.0, 2.0, 5.0]})}

        result_plain = relative_rmse(plain_sim, self.obs, "QoI1")
        result_df = relative_rmse(df_sim, self.obs, "QoI1")

        self.assertEqual(result_plain, result_df)
        self.assertAlmostEqual(result_df, 0.3254270698294439, places=10)

    def test_missing_key_in_sim_returns_inf(self):
        sim = {"other_key": np.array([1.0, 2.0, 3.0, 4.0])}
        self.assertEqual(relative_rmse(sim, self.obs, "QoI1"), np.inf)

    def test_missing_key_in_obs_returns_inf(self):
        sim = {"QoI1": np.array([1.0, 2.0, 3.0, 4.0])}
        self.assertEqual(relative_rmse(sim, {"other_key": np.array([1.0])}, "QoI1"), np.inf)

    def test_shape_mismatch_returns_inf(self):
        """A run that stopped early / produced malformed output has a different
        length than the observed series -- must reject, not raise or broadcast."""
        sim = {"QoI1": np.array([1.0, 2.0, 3.0])}
        self.assertEqual(relative_rmse(sim, self.obs, "QoI1"), np.inf)

    def test_all_nan_returns_inf(self):
        obs_all_nan = {"QoI1": np.array([np.nan, np.nan, np.nan, np.nan])}
        sim = {"QoI1": np.array([1.0, 2.0, 3.0, 4.0])}
        self.assertEqual(relative_rmse(sim, obs_all_nan, "QoI1"), np.inf)

    def test_non_overlapping_finite_values_returns_inf(self):
        """Both series have finite values, but never at the same index -- the
        overlap mask is empty, so there is nothing to compute an RMSE over."""
        obs_disjoint = {"QoI1": np.array([1.0, np.nan, 3.0, np.nan])}
        sim_disjoint = {"QoI1": np.array([np.nan, 2.0, np.nan, 4.0])}
        self.assertEqual(relative_rmse(sim_disjoint, obs_disjoint, "QoI1"), np.inf)

    def test_endpoint_only_obs_restricts_to_finite_overlap(self):
        """A cumulative/endpoint QoI (e.g. a death count) is NaN everywhere
        except the last time point; the RMSE must be computed only over the
        finite overlap, not corrupted by (or rejected because of) the NaNs."""
        obs = {"QoI1": np.array([np.nan, np.nan, 10.0])}
        sim = {"QoI1": np.array([1.0, 2.0, 12.0])}
        # only the last point is finite in both: diff=-2, denom=max(10,1)=10 -> 0.2
        result = relative_rmse(sim, obs, "QoI1")
        self.assertAlmostEqual(result, 0.2, places=10)


class TestChi2Distance(unittest.TestCase):
    """chi2_distance(dic_obs_data, dic_model_data, rel_noise): observed data FIRST
    (the BO calling convention), sigma = rel_noise * |obs|."""

    def _d(self, t, v):
        return {"time": np.asarray(t, dtype=float), "value": np.asarray(v, dtype=float)}

    def test_exported_from_utils(self):
        import uq_physicell.utils as utils
        self.assertIn("chi2_distance", utils.__all__)
        self.assertIs(utils.chi2_distance, chi2_distance)

    def test_none_model_returns_inf(self):
        """A failed simulation must reject the particle via np.inf."""
        self.assertEqual(chi2_distance(self._d([0, 1], [1.0, 2.0]), None), np.inf)

    def test_perfect_match_is_zero(self):
        d = self._d([0, 1, 2], [100.0, 200.0, 300.0])
        self.assertEqual(chi2_distance(d, d), 0.0)

    def test_hand_computed_value(self):
        # obs = [100, 200], model = [105, 190], rel_noise = 0.05
        # sigma = [5, 10]; z = [5/5, -10/10] = [1, -1]; mean(z^2) = 1
        obs = self._d([0, 1], [100.0, 200.0])
        model = self._d([0, 1], [105.0, 190.0])
        self.assertAlmostEqual(chi2_distance(obs, model), 1.0, places=12)

    def test_rel_noise_parameter_scales_distance(self):
        obs = self._d([0], [100.0])
        model = self._d([0], [105.0])
        self.assertAlmostEqual(chi2_distance(obs, model, rel_noise=0.05), 1.0, places=12)
        # doubling sigma divides the distance by 4
        self.assertAlmostEqual(chi2_distance(obs, model, rel_noise=0.10), 0.25, places=12)

    def test_argument_order_is_observed_then_model(self):
        """Not symmetric: sigma comes from the FIRST argument (the observation).
        Swapping the arguments (the pyABC order) silently gives a different value."""
        obs = self._d([0], [100.0])
        model = self._d([0], [105.0])
        self.assertAlmostEqual(chi2_distance(obs, model), 1.0, places=12)               # sigma = 5.00
        self.assertAlmostEqual(chi2_distance(model, obs), (5.0 / 5.25) ** 2, places=12)  # sigma = 5.25
        self.assertNotAlmostEqual(chi2_distance(obs, model), chi2_distance(model, obs), places=6)

    def test_distance_independent_of_number_of_points(self):
        """Mean (not sum): a long series and a single end-point with the same relative
        error give the same distance -- what makes time-series and endpoint QoIs comparable."""
        n = 31
        obs_series = self._d(np.arange(n), np.full(n, 1000.0))
        model_series = self._d(np.arange(n), np.full(n, 1050.0))
        obs_point = self._d([7200], [1000.0])
        model_point = self._d([7200], [1050.0])
        # diff = 50, sigma = 0.05 * 1000 = 50 -> z = 1 -> distance = 1 for both
        self.assertAlmostEqual(chi2_distance(obs_series, model_series), 1.0, places=12)
        self.assertAlmostEqual(chi2_distance(obs_point, model_point), 1.0, places=12)

    def test_partial_time_overlap_uses_common_points_only(self):
        obs = self._d([0, 1, 2, 3], [100.0, 200.0, 300.0, 400.0])
        model = self._d([0, 1, 2, 5], [105.0, 190.0, 300.0, 9999.0])
        # common t = {0, 1, 2}: z^2 = [1, 1, 0] -> mean = 2/3
        self.assertAlmostEqual(chi2_distance(obs, model), 2.0 / 3.0, places=12)

    def test_no_overlap_raises(self):
        with self.assertRaises(ValueError) as ctx:
            chi2_distance(self._d([0, 1], [1.0, 2.0]), self._d([5, 6], [1.0, 2.0]))
        self.assertIn("No matching time points found", str(ctx.exception))

    def test_zero_observation_is_finite_not_nan(self):
        """sigma is floored at 1e-8, so an observed 0 matched exactly gives 0, not 0/0."""
        d = self._d([0, 1], [0.0, 10.0])
        self.assertEqual(chi2_distance(d, d), 0.0)
        self.assertTrue(np.isfinite(chi2_distance(d, self._d([0, 1], [1.0, 10.0]))))

    def test_expected_value_is_one_at_truth_under_injected_noise(self):
        """With 5% relative Gaussian noise injected exactly as in GenerateData.ipynb and
        the model equal to the noise-free truth, the reduced chi-squared is ~1 -- the
        property that puts fitness = exp(-distance) on a common scale across QoIs."""
        rng = np.random.default_rng(0)
        t = np.arange(4000)
        truth = np.linspace(500.0, 2000.0, t.size)
        obs_noisy = truth + truth * 0.05 * rng.standard_normal(t.size)
        self.assertAlmostEqual(chi2_distance(self._d(t, obs_noisy), self._d(t, truth)), 1.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()