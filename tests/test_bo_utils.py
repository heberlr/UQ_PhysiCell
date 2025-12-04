import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock, patch, Mock

# Import the functions to test
from uq_physicell.bo.utils import (
    normalize_params_df,
    extract_best_parameters,
    extract_best_parameters_db,
    extract_all_pareto_points,
    analyze_pareto_results,
    get_observed_qoi,
    normalize_params,
    unnormalize_params,
    tensor_to_param_dict,
    param_dict_to_tensor
)


class TestBOUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data for testing
        self.df_params = pd.DataFrame({
            'SampleID': [1, 1, 2, 2],
            'ParamName': ['param1', 'param2', 'param1', 'param2'],
            'ParamValue': [0.5, 1.5, 0.8, 1.2]
        })
        
        self.df_search_space = pd.DataFrame({
            'ParamName': ['param1', 'param2'],
            'lower_bound': [0.0, 1.0],
            'upper_bound': [1.0, 2.0]
        })
        
        self.search_space = {
            'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
            'param2': {'type': 'real', 'lower_bound': 1.0, 'upper_bound': 2.0},
            'param3': {'type': 'integer', 'lower_bound': 1, 'upper_bound': 10}
        }
        
        # Sample GP models and samples for testing extract_best_parameters
        self.df_gp_models = pd.DataFrame({
            'IterationID': [1, 2, 3],
            'Hypervolume': [0.1, 0.3, 0.2]
        })
        
        self.df_samples = pd.DataFrame({
            'IterationID': [1, 1, 2, 2, 3, 3],
            'SampleID': [1, 1, 2, 2, 3, 3],
            'ParamName': ['param1', 'param2', 'param1', 'param2', 'param1', 'param2'],
            'ParamValue': [0.1, 1.1, 0.5, 1.5, 0.3, 1.3]
        })
        
        # Sample data for Pareto analysis
        self.df_qois = pd.DataFrame({
            'QoI_Name': ['obj1', 'obj2']
        })
        
        self.df_output = pd.DataFrame({
            'SampleID': [1, 2, 3],
            'ObjFunc': [
                {'obj1': 0.1, 'obj2': 0.8},
                {'obj1': 0.9, 'obj2': 0.2},
                {'obj1': 0.5, 'obj2': 0.5}
            ]
        })

    def test_normalize_params_df(self):
        """Test normalize_params_df function."""
        result = normalize_params_df(self.df_params, self.df_search_space)
        
        # Check that the result has the correct columns
        expected_columns = ['SampleID', 'ParamName', 'ParamValue']
        self.assertListEqual(list(result.columns), expected_columns)
        
        # Check normalization for specific values
        # param1: (0.5 - 0.0) / (1.0 - 0.0) = 0.5
        # param2: (1.5 - 1.0) / (2.0 - 1.0) = 0.5
        normalized_values = result[result['SampleID'] == 1]['ParamValue'].values
        expected_values = [0.5, 0.5]
        np.testing.assert_array_almost_equal(normalized_values, expected_values)

    def test_extract_best_parameters(self):
        """Test extract_best_parameters function."""
        best_params, best_sample_id = extract_best_parameters(self.df_gp_models, self.df_samples)
        
        # Best hypervolume is 0.3 at IterationID=2, SampleID=2
        expected_params = {'param1': 0.5, 'param2': 1.5}
        expected_sample_id = 2
        
        self.assertEqual(best_params, expected_params)
        self.assertEqual(best_sample_id, expected_sample_id)

    @patch('uq_physicell.bo.utils.load_structure')
    def test_extract_best_parameters_db(self, mock_load_structure):
        """Test extract_best_parameters_db function."""
        # Mock the load_structure return values
        mock_load_structure.return_value = (
            None, None, None, self.df_gp_models, self.df_samples, None
        )
        
        best_params, best_sample_id = extract_best_parameters_db('/fake/path.db')
        
        # Should call load_structure and return same result as extract_best_parameters
        mock_load_structure.assert_called_once_with('/fake/path.db')
        expected_params = {'param1': 0.5, 'param2': 1.5}
        expected_sample_id = 2
        
        self.assertEqual(best_params, expected_params)
        self.assertEqual(best_sample_id, expected_sample_id)

    @patch('botorch.utils.multi_objective.pareto.is_non_dominated')
    def test_extract_all_pareto_points(self, mock_is_non_dominated):
        """Test extract_all_pareto_points function."""
        # Mock is_non_dominated to return a mask indicating first and second points are Pareto optimal
        mock_is_non_dominated.return_value = torch.tensor([True, True, False])
        
        result = extract_all_pareto_points(self.df_qois, self.df_samples, self.df_output)
        
        # Check structure of result
        self.assertIn('pareto_front', result)
        pareto_front = result['pareto_front']
        
        self.assertIn('fitness_values', pareto_front)
        self.assertIn('sample_ids', pareto_front)
        self.assertIn('parameters', pareto_front)
        self.assertIn('n_points', pareto_front)
        
        # Check that we get 2 Pareto points
        self.assertEqual(pareto_front['n_points'], 2)
        self.assertEqual(len(pareto_front['sample_ids']), 2)

    def test_get_observed_qoi(self):
        """Test get_observed_qoi function."""
        # Create a temporary CSV file for testing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Obj1_Col,Obj2_Col\n")
            f.write("0,0.1,0.8\n")
            f.write("1,0.2,0.7\n")
            f.write("2,0.3,0.6\n")
            temp_file = f.name
        
        try:
            # Create df_qois with column mappings
            df_qois_test = pd.DataFrame({
                'QoI_Name': ['obj1', 'obj2'],
                'ObsData_Column': ['Obj1_Col', 'Obj2_Col']
            })
            
            result = get_observed_qoi(temp_file, df_qois_test)
            
            # Check that columns are renamed correctly
            expected_columns = ['time', 'obj1', 'obj2']
            self.assertListEqual(list(result.columns), expected_columns)
            
            # Check data values
            self.assertEqual(len(result), 3)
            self.assertEqual(result.iloc[0]['time'], 0)
            self.assertEqual(result.iloc[0]['obj1'], 0.1)
            
        finally:
            os.unlink(temp_file)

    def test_normalize_params_tensor(self):
        """Test normalize_params function with tensor input."""
        params = torch.tensor([0.5, 1.5, 5.0])  # [param1, param2, param3]
        
        result = normalize_params(params, self.search_space)
        
        # Expected: [(0.5-0.0)/(1.0-0.0), (1.5-1.0)/(2.0-1.0), (5.0-1)/(10-1)]
        expected = torch.tensor([0.5, 0.5, 4.0/9.0], dtype=torch.double)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_unnormalize_params_1d(self):
        """Test unnormalize_params function with 1D tensor."""
        normalized_params = torch.tensor([0.5, 0.5, 0.5])
        
        result = unnormalize_params(normalized_params, self.search_space)
        
        # Expected: [0.0+0.5*1.0, 1.0+0.5*1.0, 1+0.5*9] = [0.5, 1.5, 5.5->6]
        expected = torch.tensor([0.5, 1.5, 6.0], dtype=torch.double)  # integer param gets rounded
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_unnormalize_params_2d(self):
        """Test unnormalize_params function with 2D tensor (batch)."""
        normalized_params = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        result = unnormalize_params(normalized_params, self.search_space)
        
        # Expected: 
        # First row: [0.0, 1.0, 1.0]
        # Second row: [1.0, 2.0, 10.0]
        expected = torch.tensor([
            [0.0, 1.0, 1.0],
            [1.0, 2.0, 10.0]
        ], dtype=torch.double)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_tensor_to_param_dict_1d(self):
        """Test tensor_to_param_dict function with 1D tensor."""
        params_tensor = torch.tensor([0.5, 1.5, 5.0])
        
        result = tensor_to_param_dict(params_tensor, self.search_space)
        
        expected = {'param1': 0.5, 'param2': 1.5, 'param3': 5.0}
        self.assertEqual(result, expected)

    def test_tensor_to_param_dict_2d(self):
        """Test tensor_to_param_dict function with 2D tensor."""
        params_tensor = torch.tensor([
            [0.1, 1.1, 2.0],
            [0.9, 1.9, 8.0]
        ])
        
        result = tensor_to_param_dict(params_tensor, self.search_space)
        
        expected = [
            {'param1': 0.1, 'param2': 1.1, 'param3': 2.0},
            {'param1': 0.9, 'param2': 1.9, 'param3': 8.0}
        ]
        # Use approximate equality for floating point comparisons
        self.assertEqual(len(result), len(expected))
        for i, (actual, expect) in enumerate(zip(result, expected)):
            self.assertEqual(set(actual.keys()), set(expect.keys()))
            for key in actual.keys():
                self.assertAlmostEqual(actual[key], expect[key], places=5)

    def test_param_dict_to_tensor(self):
        """Test param_dict_to_tensor function."""
        params_dict = {'param1': 0.5, 'param2': 1.5, 'param3': 5.0}
        
        result = param_dict_to_tensor(params_dict, self.search_space)
        
        expected = torch.tensor([0.5, 1.5, 5.0], dtype=torch.double)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_invalid_tensor_dimensions(self):
        """Test error handling for invalid tensor dimensions."""
        # Test 3D tensor (should raise ValueError)
        invalid_tensor = torch.tensor([[[1, 2, 3]]])
        
        with self.assertRaises(ValueError):
            unnormalize_params(invalid_tensor, self.search_space)
        
        with self.assertRaises(ValueError):
            tensor_to_param_dict(invalid_tensor, self.search_space)

    def test_unknown_parameter_type(self):
        """Test error handling for unknown parameter type."""
        bad_search_space = {
            'param1': {'type': 'unknown', 'lower_bound': 0.0, 'upper_bound': 1.0}
        }
        
        params = torch.tensor([0.5])
        
        with self.assertRaises(ValueError):
            normalize_params(params, bad_search_space)
        
        with self.assertRaises(ValueError):
            unnormalize_params(params, bad_search_space)

    @patch('uq_physicell.bo.utils.extract_all_pareto_points')
    def test_analyze_pareto_results(self, mock_extract_pareto):
        """Test analyze_pareto_results function."""
        # Mock the extract_all_pareto_points return value
        mock_pareto_data = {
            'pareto_front': {
                'fitness_values': np.array([[0.1, 0.8], [0.9, 0.2]]),
                'sample_ids': [1, 2],
                'parameters': [{'param1': 0.1}, {'param1': 0.9}],
                'n_points': 2
            }
        }
        mock_extract_pareto.return_value = mock_pareto_data
        
        # Capture print output by mocking print
        with patch('builtins.print') as mock_print:
            result = analyze_pareto_results(self.df_qois, self.df_samples, self.df_output)
        
        # Check that function returns the expected data
        self.assertEqual(result, mock_pareto_data)
        
        # Verify that print was called (indicating output was generated)
        self.assertTrue(mock_print.called)


if __name__ == '__main__':
    unittest.main()