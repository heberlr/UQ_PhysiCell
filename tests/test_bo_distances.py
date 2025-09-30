import unittest
import numpy as np

# Import the distance functions to test
from uq_physicell.bo.distances import SumSquaredDifferences, Manhattan, Chebyshev


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


if __name__ == '__main__':
    unittest.main()