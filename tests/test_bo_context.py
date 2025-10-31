import unittest
import os
import tempfile
import logging
import pandas as pd
import torch
import numpy as np
from unittest.mock import MagicMock, patch, Mock, mock_open

# Import the classes and functions to test
from uq_physicell.bo.bo_context import CalibrationContext, run_bayesian_optimization
from uq_physicell.utils import SumSquaredDifferences


class TestCalibrationContext(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for observed data
        self.test_data = pd.DataFrame({
            'Time': [0, 1, 2, 3],
            'Obj1_Column': [0.1, 0.2, 0.3, 0.4],
            'Obj2_Column': [0.8, 0.7, 0.6, 0.5]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        # Basic configuration dictionaries
        self.obsData_columns = {
            'obj1': 'Obj1_Column',
            'obj2': 'Obj2_Column'
        }
        
        self.model_config = {
            'ini_path': '/fake/path/config.ini',
            'struc_name': 'test_structure',
            'numReplicates': 3
        }
        
        self.qoi_functions = {
            'obj1': 'lambda df: df["metric1"].sum()',
            'obj2': 'lambda df: df["metric2"].mean()'
        }
        
        self.distance_functions = {
            'obj1': {'function': SumSquaredDifferences, 'weight': 1e-5},
            'obj2': {'function': SumSquaredDifferences, 'weight': 1e-4}
        }
        
        self.search_space = {
            'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
            'param2': {'type': 'real', 'lower_bound': 0.5, 'upper_bound': 2.0}
        }
        
        self.bo_options = {
            'num_initial_samples': 10,
            'num_iterations': 5,
            'max_workers': 2,
            'batch_size_per_iteration': 1
        }
        
        # Create a logger for testing with higher level to suppress error messages
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.INFO)

    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass

    def test_init_with_file_path(self):
        """Test CalibrationContext initialization with file path for observed data."""
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # Check that observed data was loaded correctly
        self.assertEqual(context.obsData_path, self.temp_file.name)
        self.assertIn('obj1', context.dic_obsData)
        self.assertIn('obj2', context.dic_obsData)
        
        # Check that data transformation worked
        np.testing.assert_array_equal(context.dic_obsData['obj1'], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(context.dic_obsData['obj2'], [0.8, 0.7, 0.6, 0.5])

    def test_init_with_dict_data(self):
        """Test CalibrationContext initialization with dictionary data."""
        dict_data = {
            'obj1': np.array([0.1, 0.2, 0.3]),
            'obj2': np.array([0.8, 0.7, 0.6])
        }
        
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=dict_data,
            obsData_columns=self.obsData_columns,  # This shouldn't be used for dict input
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # Check that observed data was set directly
        self.assertIsNone(context.obsData_path)
        self.assertEqual(context.dic_obsData, dict_data)

    def test_invalid_observed_data_file(self):
        """Test error handling for invalid observed data file."""
        with self.assertRaises(Exception):
            CalibrationContext(
                db_path='/fake/db.db',
                obsData='/nonexistent/file.csv',
                obsData_columns=self.obsData_columns,
                model_config=self.model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=self.search_space,
                bo_options=self.bo_options,
                logger=self.logger
            )

    def test_invalid_acquisition_strategy(self):
        """Test error handling for invalid acquisition strategy."""
        bo_options_bad = self.bo_options.copy()
        bo_options_bad['acq_func_strategy'] = 'invalid_strategy'
        
        with self.assertRaises(Exception):
            CalibrationContext(
                db_path='/fake/db.db',
                obsData=self.temp_file.name,
                obsData_columns=self.obsData_columns,
                model_config=self.model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=self.search_space,
                bo_options=bo_options_bad,
                logger=self.logger
            )

    def test_valid_acquisition_strategies(self):
        """Test that all valid acquisition strategies are accepted."""
        valid_strategies = ["diversity_bonus", "uncertainty_weighting", "soft_constraints", 
                          "adaptive_scaling", "combined", "none"]
        
        for strategy in valid_strategies:
            bo_options_test = self.bo_options.copy()
            bo_options_test['acq_func_strategy'] = strategy
            
            # This should not raise an exception
            context = CalibrationContext(
                db_path='/fake/db.db',
                obsData=self.temp_file.name,
                obsData_columns=self.obsData_columns,
                model_config=self.model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=self.search_space,
                bo_options=bo_options_test,
                logger=self.logger
            )
            
            # If we got here without exception, the strategy was accepted
            self.assertIsInstance(context, CalibrationContext)

    def test_worker_configuration(self):
        """Test worker configuration calculation."""
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # Check worker configuration
        self.assertEqual(context.num_replicates, 3)
        self.assertEqual(context.max_workers, 2)
        self.assertEqual(context.workers_inner, min(2, 3))  # min(max_workers, num_replicates)
        self.assertEqual(context.workers_out, max(1, 2 // context.workers_inner))

    def test_qoi_details_initialization(self):
        """Test QoI details initialization."""
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # Check QoI details structure
        self.assertIn('QOI_Name', context.qoi_details)
        self.assertIn('QOI_Function', context.qoi_details)
        self.assertIn('ObsData_Column', context.qoi_details)
        self.assertIn('QoI_distanceFunction', context.qoi_details)
        self.assertIn('QoI_distanceWeight', context.qoi_details)
        
        # Check QoI details content
        self.assertEqual(context.qoi_details['QOI_Name'], ['obj1', 'obj2'])
        self.assertEqual(context.qoi_details['ObsData_Column'], ['Obj1_Column', 'Obj2_Column'])
        self.assertEqual(context.qoi_details['QoI_distanceWeight'], [1e-5, 1e-4])

    def test_metadata_initialization(self):
        """Test metadata initialization."""
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # Check metadata structure
        self.assertIn('BO_Method', context.dic_metadata)
        self.assertIn('ObsData_Path', context.dic_metadata)
        self.assertIn('Ini_File_Path', context.dic_metadata)
        self.assertIn('StructureName', context.dic_metadata)
        
        # Check specific values
        self.assertEqual(context.dic_metadata['ObsData_Path'], self.temp_file.name)
        self.assertEqual(context.dic_metadata['Ini_File_Path'], self.model_config['ini_path'])
        self.assertEqual(context.dic_metadata['StructureName'], self.model_config['struc_name'])
        
        # For multi-objective (2 QoIs)
        self.assertTrue('Multi-objective' in context.dic_metadata['BO_Method'])

    def test_single_objective_metadata(self):
        """Test metadata for single objective case."""
        single_qoi_functions = {'obj1': 'lambda df: df["metric1"].sum()'}
        single_distance_functions = {'obj1': {'function': SumSquaredDifferences, 'weight': 1e-5}}
        
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns={'obj1': 'Obj1_Column'},
            model_config=self.model_config,
            qoi_functions=single_qoi_functions,
            distance_functions=single_distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        
        # For single objective
        self.assertTrue('Single-objective' in context.dic_metadata['BO_Method'])

    def test_optional_parameters(self):
        """Test optional parameters in bo_options."""
        bo_options_with_optional = self.bo_options.copy()
        bo_options_with_optional.update({
            'fixed_params': {'param3': 0.5},
            'summary_function': 'custom_summary',
            'custom_run_single_replicate_func': lambda: None,
            'custom_aggregation_func': lambda: None,
            'acq_func_strategy': 'combined',
            'diversity_weight': 0.1,
            'uncertainty_weight': 0.2
        })
        
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=bo_options_with_optional,
            logger=self.logger
        )
        
        # Check optional parameters
        self.assertEqual(context.fixed_params, {'param3': 0.5})
        self.assertEqual(context.summary_function, 'custom_summary')
        self.assertIsNotNone(context.custom_run_single_replicate_func)
        self.assertIsNotNone(context.custom_aggregation_func)
        
        # Check metadata includes strategy information
        self.assertIn('Enhancement_Strategy', context.dic_metadata)
        self.assertIn('Diversity_Weight', context.dic_metadata)
        self.assertIn('Uncertainty_Weight', context.dic_metadata)

    def test_missing_column_error(self):
        """Test error when observed data column is missing."""
        bad_obsData_columns = {
            'obj1': 'NonExistent_Column',
            'obj2': 'Obj2_Column'
        }
        
        with self.assertRaises(ValueError):
            CalibrationContext(
                db_path='/fake/db.db',
                obsData=self.temp_file.name,
                obsData_columns=bad_obsData_columns,
                model_config=self.model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=self.search_space,
                bo_options=self.bo_options,
                logger=self.logger
            )


class TestRunBayesianOptimization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for run_bayesian_optimization tests."""
        # Create a logger with CRITICAL level to suppress error messages during testing
        self.logger = logging.getLogger('test_logger')
        self.logger.setLevel(logging.CRITICAL)  # Changed from INFO to CRITICAL
        
        # Create a mock CalibrationContext
        self.mock_context = MagicMock()
        self.mock_context.db_path = '/fake/path.db'
        self.mock_context.logger = self.logger
        self.mock_context.qoi_details = {'QOI_Name': ['obj1', 'obj2']}

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    @patch('uq_physicell.bo.bo_context.create_structure')
    @patch('uq_physicell.bo.bo_context.insert_metadata')
    @patch('uq_physicell.bo.bo_context.insert_param_space')
    @patch('uq_physicell.bo.bo_context.insert_qois')
    @patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization')
    def test_fresh_optimization_multi_objective(self, mock_multi_obj, mock_insert_qois, 
                                               mock_insert_params, mock_insert_meta, 
                                               mock_create_struct, mock_exists):
        """Test fresh optimization for multi-objective case."""
        # Database doesn't exist
        mock_exists.return_value = False
        
        # Mock the generate_and_evaluate_samples method
        self.mock_context.num_initial_samples = 10
        self.mock_context.generate_and_evaluate_samples.return_value = (
            torch.randn(10, 2), torch.randn(10, 2), torch.randn(10, 2), torch.randn(10, 2)
        )
        
        run_bayesian_optimization(self.mock_context)
        
        # Verify database setup calls
        mock_create_struct.assert_called_once_with('/fake/path.db')
        mock_insert_meta.assert_called_once()
        mock_insert_params.assert_called_once()
        mock_insert_qois.assert_called_once()
        
        # Verify multi-objective optimization was called
        mock_multi_obj.assert_called_once()
        
        # Verify sample generation was called
        self.mock_context.generate_and_evaluate_samples.assert_called_once_with(10, start_sample_id=0, iteration_id=0)

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    @patch('uq_physicell.bo.bo_context.single_objective_bayesian_optimization')
    def test_fresh_optimization_single_objective(self, mock_single_obj, mock_exists):
        """Test fresh optimization for single objective case."""
        # Database doesn't exist
        mock_exists.return_value = False
        
        # Single QoI
        self.mock_context.qoi_details = {'QOI_Name': ['obj1']}
        
        # Mock required methods
        self.mock_context.num_initial_samples = 10
        self.mock_context.generate_and_evaluate_samples.return_value = (
            torch.randn(10, 2), torch.randn(10, 1), torch.randn(10, 1), torch.randn(10, 1)
        )
        
        with patch('uq_physicell.bo.bo_context.create_structure'), \
             patch('uq_physicell.bo.bo_context.insert_metadata'), \
             patch('uq_physicell.bo.bo_context.insert_param_space'), \
             patch('uq_physicell.bo.bo_context.insert_qois'):
            
            run_bayesian_optimization(self.mock_context)
            
            # Verify single-objective optimization was called
            mock_single_obj.assert_called_once()

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    @patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization')
    def test_resume_optimization(self, mock_multi_obj, mock_exists):
        """Test resuming optimization from existing database."""
        # Database exists
        mock_exists.return_value = True
        
        # Mock the load_existing_data method
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2), 
            torch.randn(15, 2), 2, 0.5  # latest_iteration, latest_hypervolume
        )
        
        run_bayesian_optimization(self.mock_context)
        
        # Verify load_existing_data was called
        self.mock_context.load_existing_data.assert_called_once()
        
        # Verify multi-objective optimization was called with start_iteration=3
        mock_multi_obj.assert_called_once()
        args = mock_multi_obj.call_args[0]
        start_iteration = args[5]  # 6th argument (0-indexed)
        self.assertEqual(start_iteration, 3)

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    def test_additional_iterations(self, mock_exists):
        """Test additional iterations parameter."""
        # Database exists
        mock_exists.return_value = True
        
        # Mock the methods
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2), 
            torch.randn(15, 2), 2, 0.5
        )
        
        with patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization'):
            run_bayesian_optimization(self.mock_context, additional_iterations=5)
            
            # Verify update_bo_iterations was called with additional_iterations
            self.mock_context.update_bo_iterations.assert_called_once_with(5)

    def test_exception_handling(self):
        """Test exception handling during optimization."""
        # Temporarily set logger to CRITICAL level to suppress the error message during this test
        original_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL)
        
        try:
            with patch('uq_physicell.bo.bo_context.os.path.exists', return_value=False):
                # Make generate_and_evaluate_samples raise an exception
                self.mock_context.generate_and_evaluate_samples.side_effect = Exception("Test error")
                
                with patch('uq_physicell.bo.bo_context.create_structure'), \
                     patch('uq_physicell.bo.bo_context.insert_metadata'), \
                     patch('uq_physicell.bo.bo_context.insert_param_space'), \
                     patch('uq_physicell.bo.bo_context.insert_qois'):
                    
                    with self.assertRaises(Exception):
                        run_bayesian_optimization(self.mock_context)
        finally:
            # Restore original logger level
            self.logger.setLevel(original_level)


if __name__ == '__main__':
    unittest.main()