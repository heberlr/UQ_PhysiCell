import unittest
import os
import json
import tempfile
import logging
import pandas as pd
import torch
import numpy as np
from unittest.mock import MagicMock, patch, Mock, mock_open
import pytest

# Require heavy optional dependencies for these tests: botorch and gpytorch.
# When running under pytest, this will skip the module with an informative reason if missing.
pytest.importorskip('botorch', reason="This test requires 'botorch'. Install it via 'pip install botorch'.")
pytest.importorskip('gpytorch', reason="This test requires 'gpytorch'. Install it via 'pip install gpytorch'.")

# Import the classes and functions to test
from uq_physicell.bo.bo_context import (
    CalibrationContext,
    run_bayesian_optimization,
    single_objective_bayesian_optimization,
    multi_objective_bayesian_optimization,
    _fit_gp_models,
    _optimize_acquisition_function,
    _extract_pareto_and_hypervolume_from_acqf,
    _convergence_status_to_json,
)
from uq_physicell.utils import SumSquaredDifferences
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP


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

        # BO_Options is stored as a JSON string with the resolved BO configuration
        self.assertIn('BO_Options', context.dic_metadata)
        bo_options_record = json.loads(context.dic_metadata['BO_Options'])
        self.assertEqual(bo_options_record['num_initial_samples'], 10)
        self.assertEqual(bo_options_record['num_iterations'], 5)
        self.assertEqual(bo_options_record['batch_size_per_iteration'], 1)
        self.assertEqual(bo_options_record['use_exponential_fitness'], True)
        self.assertEqual(bo_options_record['use_correlated_gp'], False)
        self.assertEqual(bo_options_record['ref_point'], [0.0, 0.0])
        self.assertIsNone(bo_options_record['custom_aggregation_func'])

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

    def test_default_logger_created_when_none_provided(self):
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=None
        )
        self.assertIsInstance(context.logger, logging.Logger)
        self.assertFalse(context.logger.propagate)
        self.assertTrue(len(context.logger.handlers) >= 1)

    def test_num_replicates_read_from_ini_file(self):
        ini_file = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
        ini_file.write("[test_structure]\nnumReplicates = 4\n")
        ini_file.close()
        try:
            model_config = {'ini_path': ini_file.name, 'struc_name': 'test_structure'}
            context = CalibrationContext(
                db_path='/fake/db.db',
                obsData=self.temp_file.name,
                obsData_columns=self.obsData_columns,
                model_config=model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=self.search_space,
                bo_options=self.bo_options,
                logger=self.logger
            )
            self.assertEqual(context.num_replicates, 4)
        finally:
            os.unlink(ini_file.name)

    def test_db_path_initial_samples_loads_search_space_from_db(self):
        df_param_space = pd.DataFrame({
            'ParamName': ['param1', 'param2'],
            'lower_bound': [0.0, 0.5],
            'upper_bound': [1.0, 2.0],
        })
        bo_options = self.bo_options.copy()
        bo_options['db_path_initial_samples'] = '/fake/initial.db'
        with patch('uq_physicell.bo.bo_context.load_ma_parameter_samples', return_value=df_param_space):
            context = CalibrationContext(
                db_path='/fake/db.db',
                obsData=self.temp_file.name,
                obsData_columns=self.obsData_columns,
                model_config=self.model_config,
                qoi_functions=self.qoi_functions,
                distance_functions=self.distance_functions,
                search_space=None,
                bo_options=bo_options,
                logger=self.logger
            )
        self.assertEqual(context.num_initial_samples, 0)
        self.assertEqual(context.search_space['param1'], {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0})
        self.assertEqual(context.search_space['param2'], {'type': 'real', 'lower_bound': 0.5, 'upper_bound': 2.0})

    def test_default_distance_functions_used_when_none_provided(self):
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=None,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        self.assertEqual(context.qoi_details['QoI_distanceFunction'], ['SumSquaredDifferences', 'SumSquaredDifferences'])
        # Weights are auto-estimated from the observed data range, never zero/None
        for weight in context.qoi_details['QoI_distanceWeight']:
            self.assertIsNotNone(weight)
            self.assertGreater(weight, 0.0)

    def test_estimate_weights_from_obsdata_auto_formula(self):
        context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.temp_file.name,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions={
                'obj1': {'function': SumSquaredDifferences},
                'obj2': {'function': SumSquaredDifferences, 'weight': 1e-4},
            },
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger
        )
        updated = context._estimate_weights_from_obsdata(self.obsData_columns)
        expected_weight = 1.0 / (0.3 * 4 + 1e-10)  # range=0.4-0.1, 4 observations
        self.assertAlmostEqual(updated['obj1']['weight'], expected_weight, places=6)
        self.assertEqual(updated['obj2']['weight'], 1e-4)


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
        self.mock_context.db_path_initial_samples = None
        
        # Patch PhysiCell_Model in the bo_context module so tests don't instantiate the real model
        # This keeps changes confined to tests and avoids touching core code.
        self._physicell_patcher = patch('uq_physicell.bo.bo_context.PhysiCell_Model')
        self.mock_physicell_class = self._physicell_patcher.start()
        # Ensure the instance has a remove_io_folders method that does nothing
        mock_physicell_instance = MagicMock()
        mock_physicell_instance.remove_io_folders.return_value = None
        self.mock_physicell_class.return_value = mock_physicell_instance

    def tearDown(self):
        """Stop any active patchers from setUp."""
        try:
            self._physicell_patcher.stop()
        except Exception:
            pass

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
            torch.randn(10, 2), torch.randn(10, 2), torch.randn(10, 2)
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
        self.mock_context.generate_and_evaluate_samples.assert_called_once_with(start_sample_id=0, iteration_id=0)

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
            torch.randn(10, 2), torch.randn(10, 1), torch.randn(10, 1)
        )
        
        with patch('uq_physicell.bo.bo_context.create_structure'), \
             patch('uq_physicell.bo.bo_context.insert_metadata'), \
             patch('uq_physicell.bo.bo_context.insert_param_space'), \
             patch('uq_physicell.bo.bo_context.insert_qois'):
            
            run_bayesian_optimization(self.mock_context)
            
            # Verify single-objective optimization was called
            mock_single_obj.assert_called_once()

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    @patch('uq_physicell.bo.bo_context.create_structure')
    @patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization')
    def test_resume_optimization(self, mock_multi_obj, mock_create_struct, mock_exists):
        """Test resuming optimization from existing database."""
        # Database exists
        mock_exists.return_value = True
        
        # Mock the load_existing_data method
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2),
            2, 0.5  # latest_iteration, latest_hypervolume
        )
        
        run_bayesian_optimization(self.mock_context)
        
        # Verify load_existing_data was called
        self.mock_context.load_existing_data.assert_called_once()
        
        # Verify multi-objective optimization was called with start_iteration=3
        mock_multi_obj.assert_called_once()
        args = mock_multi_obj.call_args[0]
        start_iteration = args[4]  # 5th argument (0-indexed)
        self.assertEqual(start_iteration, 3)

    @patch('uq_physicell.bo.bo_context.os.path.exists')
    @patch('uq_physicell.bo.bo_context.create_structure')
    def test_additional_iterations(self, mock_create_struct, mock_exists):
        """Test additional iterations parameter."""
        # Database exists
        mock_exists.return_value = True
        
        # Mock the methods
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2),
            2, 0.5
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

    def test_fresh_optimization_uses_db_path_initial_samples(self):
        self.mock_context.qoi_details = {'QOI_Name': ['obj1']}
        self.mock_context.db_path_initial_samples = '/fake/initial.db'
        self.mock_context.generate_initial_samples_from_db.return_value = (
            torch.randn(5, 2), torch.randn(5, 1), torch.randn(5, 1)
        )
        with patch('uq_physicell.bo.bo_context.os.path.exists', return_value=False), \
             patch('uq_physicell.bo.bo_context.create_structure'), \
             patch('uq_physicell.bo.bo_context.insert_metadata'), \
             patch('uq_physicell.bo.bo_context.insert_param_space'), \
             patch('uq_physicell.bo.bo_context.insert_qois'), \
             patch('uq_physicell.bo.bo_context.single_objective_bayesian_optimization'):
            run_bayesian_optimization(self.mock_context)

        self.mock_context.generate_initial_samples_from_db.assert_called_once_with(start_sample_id=0, iteration_id=0)
        self.mock_context.generate_and_evaluate_samples.assert_not_called()

    def test_resume_loads_stored_weights_from_database(self):
        self.mock_context.qoi_details = {'QOI_Name': ['obj1', 'obj2']}
        self.mock_context.distance_functions = {'obj1': {'weight': 1e-5}, 'obj2': {'weight': 1e-4}}
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2), 2, 0.5
        )
        df_qois = pd.DataFrame({'QoI_Name': ['obj1', 'obj2'], 'QoI_distanceWeight': [9.9e-3, 8.8e-3]})
        with patch('uq_physicell.bo.bo_context.os.path.exists', return_value=True), \
             patch('uq_physicell.bo.bo_context.create_structure'), \
             patch('uq_physicell.bo.bo_context.load_qois', return_value=df_qois), \
             patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization'):
            run_bayesian_optimization(self.mock_context)

        self.assertEqual(self.mock_context.distance_functions['obj1']['weight'], 9.9e-3)
        self.assertEqual(self.mock_context.distance_functions['obj2']['weight'], 8.8e-3)

    def test_resume_weight_load_failure_is_non_fatal(self):
        self.mock_context.load_existing_data.return_value = (
            torch.randn(15, 2), torch.randn(15, 2), torch.randn(15, 2), 2, 0.5
        )
        with patch('uq_physicell.bo.bo_context.os.path.exists', return_value=True), \
             patch('uq_physicell.bo.bo_context.create_structure'), \
             patch('uq_physicell.bo.bo_context.load_qois', side_effect=RuntimeError('db locked')), \
             patch('uq_physicell.bo.bo_context.multi_objective_bayesian_optimization') as mock_multi:
            run_bayesian_optimization(self.mock_context)

        mock_multi.assert_called_once()

    def test_resume_exception_still_raises(self):
        original_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL)
        try:
            self.mock_context.load_existing_data.side_effect = Exception("resume boom")
            with patch('uq_physicell.bo.bo_context.os.path.exists', return_value=True), \
                 patch('uq_physicell.bo.bo_context.create_structure'):
                with self.assertRaises(Exception):
                    run_bayesian_optimization(self.mock_context)
        finally:
            self.logger.setLevel(original_level)


class TestCalibrationContextBusinessLogic(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('test_logger_business')
        self.logger.setLevel(logging.CRITICAL)

        self.obsData = {
            'time': np.array([0., 1., 2., 3.]),
            'obj1': np.array([0.1, 0.2, 0.3, 0.4]),
            'obj2': np.array([0.8, 0.7, 0.6, 0.5]),
        }
        self.obsData_columns = {'obj1': 'Obj1_Column', 'obj2': 'Obj2_Column'}
        self.model_config = {'ini_path': '/fake/path/config.ini', 'struc_name': 'test_structure', 'numReplicates': 2}
        self.qoi_functions = {'obj1': 'lambda df: df["metric1"].sum()', 'obj2': 'lambda df: df["metric2"].mean()'}
        self.distance_functions = {
            'obj1': {'function': SumSquaredDifferences, 'weight': 1e-5},
            'obj2': {'function': SumSquaredDifferences, 'weight': 1e-4},
        }
        self.search_space = {
            'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
            'param2': {'type': 'real', 'lower_bound': 0.5, 'upper_bound': 2.0},
        }
        self.bo_options = {'num_initial_samples': 3, 'num_iterations': 5, 'max_workers': 2, 'batch_size_per_iteration': 1}

        self.context = CalibrationContext(
            db_path='/fake/db.db',
            obsData=self.obsData,
            obsData_columns=self.obsData_columns,
            model_config=self.model_config,
            qoi_functions=self.qoi_functions,
            distance_functions=self.distance_functions,
            search_space=self.search_space,
            bo_options=self.bo_options,
            logger=self.logger,
        )

        self.replicate_results = [
            {'time': np.array([0., 1., 2., 3.]), 'obj1': np.array([0.1, 0.2, 0.3, 0.4]), 'obj2': np.array([0.8, 0.7, 0.6, 0.5])},
            {'time': np.array([0., 1., 2., 3.]), 'obj1': np.array([0.11, 0.19, 0.31, 0.39]), 'obj2': np.array([0.79, 0.71, 0.59, 0.51])},
        ]

    # -- default_aggregation_func --------------------------------------------------

    def test_default_aggregation_func_exponential_fitness(self):
        objectives, obj_noise, dic_results = self.context.default_aggregation_func(self.replicate_results, sample_id=0)
        self.assertAlmostEqual(objectives['obj1'], 1.0, places=3)
        self.assertAlmostEqual(objectives['obj2'], 1.0, places=3)
        self.assertIn(0, dic_results)
        self.assertIn(1, dic_results)

    def test_default_aggregation_func_inverse_distance(self):
        self.context.use_exponential_fitness = False
        objectives, obj_noise, _ = self.context.default_aggregation_func(self.replicate_results, sample_id=1)
        self.assertAlmostEqual(objectives['obj1'], 1.0, places=3)

    def test_default_aggregation_func_string_distance_function(self):
        self.context.distance_functions['obj1']['function'] = 'SumSquaredDifferences'
        objectives, _, _ = self.context.default_aggregation_func(self.replicate_results, sample_id=2)
        self.assertAlmostEqual(objectives['obj1'], 1.0, places=3)

    def test_default_aggregation_func_nan_fitness_is_clamped(self):
        self.context.distance_functions['obj1'] = {'function': lambda dObs, dModel: float('nan'), 'weight': 1.0}
        objectives, _, _ = self.context.default_aggregation_func(self.replicate_results, sample_id=3)
        self.assertEqual(objectives['obj1'], 1e-8)

    def test_default_aggregation_func_noise_is_standard_error_of_the_mean(self):
        """The noise returned for each objective must be the SEM of the replicate
        fitnesses (std / sqrt(n)), not the raw replicate std. The reported objective is the
        replicate MEAN, so its uncertainty (train_Yvar for the GP, SNR in analyze_convergence)
        shrinks with the number of replicates. Using the raw std overstated the noise by
        sqrt(n_replicates) (regression test)."""
        # Per-replicate distance = model value at t=0, with inverse-distance fitness 1/(1+d):
        #   d = [0, 1, 0, 1]  ->  fitness = [1.0, 0.5, 1.0, 0.5]
        self.context.use_exponential_fitness = False
        self.context.distance_functions = {
            'obj1': {'function': lambda dObs, dModel: float(dModel['value'][0]), 'weight': 1.0},
            'obj2': {'function': lambda dObs, dModel: 0.0, 'weight': 1.0},
        }
        t = np.array([0., 1., 2., 3.])
        replicates = [
            {'time': t, 'obj1': np.array([d, 0., 0., 0.]), 'obj2': np.zeros(4)}
            for d in (0.0, 1.0, 0.0, 1.0)
        ]
        fitness = np.array([1.0, 0.5, 1.0, 0.5])

        objectives, obj_noise, _ = self.context.default_aggregation_func(replicates, sample_id=0)

        self.assertAlmostEqual(objectives['obj1'], fitness.mean(), places=12)
        self.assertAlmostEqual(obj_noise['obj1'], fitness.std() / np.sqrt(len(replicates)), places=12)
        self.assertAlmostEqual(obj_noise['obj1'], 0.125, places=12)
        # strictly smaller than the raw replicate std by exactly sqrt(n)
        self.assertAlmostEqual(fitness.std() / obj_noise['obj1'], np.sqrt(len(replicates)), places=12)
        # a constant objective has zero noise
        self.assertEqual(obj_noise['obj2'], 0.0)

    def test_default_aggregation_func_noise_shrinks_with_more_replicates(self):
        """Same per-replicate spread, 4x as many replicates -> half the reported noise."""
        self.context.use_exponential_fitness = False
        self.context.distance_functions = {
            'obj1': {'function': lambda dObs, dModel: float(dModel['value'][0]), 'weight': 1.0},
            'obj2': {'function': lambda dObs, dModel: 0.0, 'weight': 1.0},
        }
        t = np.array([0., 1., 2., 3.])

        def reps(pattern):
            return [{'time': t, 'obj1': np.array([d, 0., 0., 0.]), 'obj2': np.zeros(4)} for d in pattern]

        _, noise_4, _ = self.context.default_aggregation_func(reps([0.0, 1.0] * 2), sample_id=0)
        _, noise_16, _ = self.context.default_aggregation_func(reps([0.0, 1.0] * 8), sample_id=1)
        self.assertAlmostEqual(noise_16['obj1'], noise_4['obj1'] / 2.0, places=12)

    # -- save_results_to_db ----------------------------------------------------------

    def test_save_results_to_db_success(self):
        with patch('uq_physicell.bo.bo_context.insert_output') as mock_insert:
            self.context.save_results_to_db(5, {'obj1': 1.0}, {'obj1': 0.1}, {0: {}}, seeds=[42, 43])
        mock_insert.assert_called_once()
        args, kwargs = mock_insert.call_args
        self.assertEqual(args[0], '/fake/db.db')
        self.assertEqual(args[1], 5)
        self.assertEqual(kwargs['seeds'], json.dumps([42, 43]))

    def test_save_results_to_db_none_seeds(self):
        with patch('uq_physicell.bo.bo_context.insert_output') as mock_insert:
            self.context.save_results_to_db(1, {'obj1': 1.0}, {'obj1': 0.0}, {})
        self.assertIsNone(mock_insert.call_args.kwargs['seeds'])

    def test_save_results_to_db_error_propagates(self):
        with patch('uq_physicell.bo.bo_context.insert_output', side_effect=RuntimeError('db fail')):
            with self.assertRaises(RuntimeError):
                self.context.save_results_to_db(1, {}, {}, {})

    # -- update_bo_iterations ---------------------------------------------------------

    def test_update_bo_iterations(self):
        original = self.context.batch_size_bo
        self.context.update_bo_iterations(3)
        self.assertEqual(self.context.batch_size_bo, original + 3)

    # -- evaluate_params ----------------------------------------------------------------

    def test_evaluate_params_default_path(self):
        def fake_default_run(sample_id, replicate_id, params, seed):
            return (
                {'time': np.array([0., 1., 2., 3.]), 'obj1': np.array([0.1, 0.2, 0.3, 0.4]), 'obj2': np.array([0.8, 0.7, 0.6, 0.5])},
                seed,
            )
        self.context.default_run_single_replicate = fake_default_run
        with patch('uq_physicell.bo.bo_context.concurrent.futures.ProcessPoolExecutor', _FakeProcessPoolExecutor):
            objectives, obj_noise, dic_results, seeds = self.context.evaluate_params({'param1': 0.5, 'param2': 1.0}, 0)
        self.assertAlmostEqual(objectives['obj1'], 1.0, places=3)
        self.assertEqual(len(seeds), 2)

    def test_evaluate_params_custom_functions(self):
        def custom_run(sample_id, replicate_id, params, fixed_params, model_config):
            return {'value': params['param1']}

        def custom_agg(replicate_results, sample_id, distance_functions, dic_obsData):
            return {'obj1': 0.42}, {'obj1': 0.0}, {'raw': replicate_results}

        self.context.custom_run_single_replicate_func = custom_run
        self.context.custom_aggregation_func = custom_agg
        with patch('uq_physicell.bo.bo_context.concurrent.futures.ProcessPoolExecutor', _FakeProcessPoolExecutor):
            objectives, obj_noise, dic_results, seeds = self.context.evaluate_params({'param1': 0.5, 'param2': 1.0}, 1)
        self.assertEqual(objectives, {'obj1': 0.42})
        self.assertEqual(seeds, [None, None])

    # -- generate_and_evaluate_samples -----------------------------------------------

    def test_generate_and_evaluate_samples(self):
        self.context.num_initial_samples = 3

        def fake_evaluate(params, sample_id):
            return ({'obj1': 1.0, 'obj2': 2.0}, {'obj1': 0.1, 'obj2': 0.1}, {'dummy': 1}, [1, 2])

        self.context.evaluate_params = fake_evaluate
        self.context.save_results_to_db = MagicMock()
        with patch('uq_physicell.bo.bo_context.concurrent.futures.ProcessPoolExecutor', _FakeProcessPoolExecutor), \
             patch('uq_physicell.bo.bo_context.insert_samples') as mock_insert_samples:
            train_x, train_obj, train_obj_std = self.context.generate_and_evaluate_samples(start_sample_id=0, iteration_id=0)
        self.assertEqual(tuple(train_x.shape), (3, 2))
        self.assertEqual(tuple(train_obj.shape), (3, 2))
        self.assertTrue(torch.all(train_obj == torch.tensor([1.0, 2.0], dtype=torch.float64)))
        self.assertEqual(mock_insert_samples.call_count, 3)
        self.assertEqual(self.context.save_results_to_db.call_count, 3)

    # -- generate_initial_samples_from_db ---------------------------------------------

    def _patched_db_loaders(self, df_param_space=None, df_qois=None, samples=None, df_qois_data=None):
        if df_param_space is None:
            df_param_space = pd.DataFrame({
                'ParamName': ['param1', 'param2'],
                'lower_bound': [0.0, 0.5],
                'upper_bound': [1.0, 2.0],
            })
        if df_qois is None:
            df_qois = pd.DataFrame({'a': [0]})
        if samples is None:
            samples = {0: {'param1': 0.5, 'param2': 1.0}, 1: {'param1': 0.2, 'param2': 1.5}}
        if df_qois_data is None:
            df_qois_data = pd.DataFrame({
                'SampleID': [0, 0, 1, 1], 'ReplicateID': [0, 1, 0, 1],
                'obj1': [1, 2, 3, 4], 'obj2': [5, 6, 7, 8],
            })
        return patch.multiple(
            'uq_physicell.bo.bo_context',
            load_ma_parameter_samples=MagicMock(return_value=df_param_space),
            load_ma_qois=MagicMock(return_value=df_qois),
            load_ma_samples=MagicMock(return_value=samples),
            calculate_qoi_from_db_file=MagicMock(return_value=df_qois_data),
            insert_samples=MagicMock(),
        )

    def test_generate_initial_samples_from_db_success(self):
        self.context.default_aggregation_func = MagicMock(return_value=({'obj1': 1.0, 'obj2': 2.0}, {'obj1': 0.1, 'obj2': 0.2}, {}))
        self.context.save_results_to_db = MagicMock()
        with self._patched_db_loaders():
            train_x, train_obj, train_obj_std = self.context.generate_initial_samples_from_db(start_sample_id=0, iteration_id=0)
        self.assertEqual(tuple(train_x.shape), (2, 2))
        self.assertEqual(tuple(train_obj.shape), (2, 2))
        self.assertEqual(self.context.save_results_to_db.call_count, 2)

    def test_generate_initial_samples_from_db_param_not_found(self):
        self.context.search_space = dict(self.context.search_space)
        self.context.search_space['param3'] = {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0}
        with self._patched_db_loaders():
            with self.assertRaises(ValueError):
                self.context.generate_initial_samples_from_db()

    def test_generate_initial_samples_from_db_bounds_mismatch(self):
        # lower_bound in the DB is well below the context's current lower_bound (0.0),
        # so `lower_bound - db_lower_bound > 1e-8` triggers the mismatch check.
        bad_param_space = pd.DataFrame({
            'ParamName': ['param1', 'param2'],
            'lower_bound': [-1.0, 0.5],
            'upper_bound': [1.0, 2.0],
        })
        with self._patched_db_loaders(df_param_space=bad_param_space):
            with self.assertRaises(ValueError):
                self.context.generate_initial_samples_from_db()

    def test_generate_initial_samples_from_db_qois_not_empty(self):
        with self._patched_db_loaders(df_qois=pd.DataFrame({'a': [1]})):
            with self.assertRaises(ValueError):
                self.context.generate_initial_samples_from_db()

    # -- load_existing_data / _validate_loaded_data / _reconstruct_training_data -----

    def _sample_loaded_dataframes(self):
        df_metadata = pd.DataFrame({
            'Effective_Run_Hash': [None],
            'BO_Options': [json.dumps(self.context._build_bo_options_record())],
        })
        df_param_space = pd.DataFrame({'ParamName': ['param1', 'param2']})
        df_qois = pd.DataFrame({'QoI_Name': ['obj1', 'obj2']})
        df_samples = pd.DataFrame({
            'SampleID': [0, 0, 1, 1],
            'IterationID': [0, 0, 0, 0],
            'ParamName': ['param1', 'param2', 'param1', 'param2'],
            'ParamValue': [0.5, 1.25, 0.0, 0.5],
        })
        df_output = pd.DataFrame({
            'SampleID': [0, 1],
            'ObjFunc': [{'obj1': 0.8, 'obj2': 0.6}, {'obj1': 0.5, 'obj2': 0.5}],
            'Noise_Std': [{'obj1': 0.05, 'obj2': 0.02}, {'obj1': 0.01, 'obj2': 0.01}],
        })
        return df_metadata, df_param_space, df_qois, df_samples, df_output

    def test_load_existing_data_with_gp_models(self):
        df_metadata, df_param_space, df_qois, df_samples, df_output = self._sample_loaded_dataframes()
        df_gp_models = pd.DataFrame({'IterationID': [0, 1], 'Score': [0.3, 0.5]})
        with patch('uq_physicell.bo.bo_context.load_structure',
                   return_value=(df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output)):
            train_x, train_obj, train_obj_std, latest_iteration, latest_hv = self.context.load_existing_data()
        self.assertEqual(latest_iteration, 1)
        self.assertEqual(latest_hv, 0.5)
        self.assertEqual(tuple(train_x.shape), (2, 2))

    def test_load_existing_data_without_gp_models(self):
        df_metadata, df_param_space, df_qois, df_samples, df_output = self._sample_loaded_dataframes()
        df_gp_models = pd.DataFrame({'IterationID': [], 'Score': []})
        with patch('uq_physicell.bo.bo_context.load_structure',
                   return_value=(df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output)):
            _, _, _, latest_iteration, latest_hv = self.context.load_existing_data()
        self.assertEqual(latest_iteration, -1)
        self.assertEqual(latest_hv, 0.0)

    def test_validate_loaded_data_success(self):
        df_metadata, df_param_space, df_qois, _, _ = self._sample_loaded_dataframes()
        self.context._validate_loaded_data(df_metadata, df_param_space, df_qois)  # should not raise

    def test_validate_loaded_data_param_mismatch(self):
        df_metadata, _, df_qois, _, _ = self._sample_loaded_dataframes()
        df_param_space_bad = pd.DataFrame({'ParamName': ['param1']})
        with self.assertRaises(ValueError):
            self.context._validate_loaded_data(df_metadata, df_param_space_bad, df_qois)

    def test_validate_loaded_data_qoi_mismatch(self):
        df_metadata, df_param_space, _, _, _ = self._sample_loaded_dataframes()
        df_qois_bad = pd.DataFrame({'QoI_Name': ['obj1']})
        with self.assertRaises(ValueError):
            self.context._validate_loaded_data(df_metadata, df_param_space, df_qois_bad)

    def test_validate_loaded_data_hash_mismatch(self):
        df_metadata, df_param_space, df_qois, _, _ = self._sample_loaded_dataframes()
        self.context.config_fingerprint = {'effective_run_hash': 'current_hash'}
        df_metadata['Effective_Run_Hash'] = ['stored_hash']
        with self.assertRaises(ValueError):
            self.context._validate_loaded_data(df_metadata, df_param_space, df_qois)

    def test_validate_loaded_data_bo_options_mismatch_warns_only(self):
        df_metadata, df_param_space, df_qois, _, _ = self._sample_loaded_dataframes()
        opts = self.context._build_bo_options_record()
        opts['use_exponential_fitness'] = not opts['use_exponential_fitness']
        df_metadata['BO_Options'] = [json.dumps(opts)]
        self.context._validate_loaded_data(df_metadata, df_param_space, df_qois)  # should not raise

    def test_validate_loaded_data_bad_json_options_handled(self):
        df_metadata, df_param_space, df_qois, _, _ = self._sample_loaded_dataframes()
        df_metadata['BO_Options'] = ['not-json']
        self.context._validate_loaded_data(df_metadata, df_param_space, df_qois)  # should not raise

    def test_reconstruct_training_data_success(self):
        _, _, _, df_samples, df_output = self._sample_loaded_dataframes()
        train_x, train_obj, train_obj_std = self.context._reconstruct_training_data(df_samples, df_output)
        self.assertEqual(tuple(train_x.shape), (2, 2))
        self.assertTrue(torch.allclose(train_x[0], torch.tensor([0.5, 0.5], dtype=torch.float64)))
        self.assertTrue(torch.allclose(train_obj[0], torch.tensor([0.8, 0.6], dtype=torch.float64)))

    def test_reconstruct_training_data_not_normalized_raises(self):
        _, _, _, _, df_output = self._sample_loaded_dataframes()
        df_samples_bad = pd.DataFrame({
            'SampleID': [0, 0],
            'ParamName': ['param1', 'param2'],
            'ParamValue': [-10.0, 1.25],
        })
        with self.assertRaises(ValueError):
            self.context._reconstruct_training_data(df_samples_bad, df_output.iloc[[0]])

    # -- analyze_convergence (multi-objective) ----------------------------------------

    def test_analyze_convergence_insufficient_data(self):
        result = self.context.analyze_convergence([0.1] * 5, torch.rand(5, 2, dtype=torch.float64),
                                                    torch.rand(5, 2, dtype=torch.float64), torch.rand(5, 2, dtype=torch.float64), iteration=5)
        self.assertEqual(result['status'], 'insufficient_data')

    def test_analyze_convergence_in_progress(self):
        n = 20
        hvs = list(np.linspace(0.1, 0.9, n))
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_obj = torch.tensor(np.tile([0.95, 0.9], (n, 1)), dtype=torch.float64)
        train_obj_std = torch.tensor(np.tile([0.001, 0.001], (n, 1)), dtype=torch.float64)
        result = self.context.analyze_convergence(hvs, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'in_progress')

    def test_analyze_convergence_converged(self):
        n = 20
        hvs = [0.8] * n
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_x[0] = torch.tensor([0.0, 0.0]); train_x[1] = torch.tensor([1.0, 1.0])
        train_obj = torch.tensor(np.tile([0.95, 0.9], (n, 1)), dtype=torch.float64)
        train_obj_std = torch.tensor(np.tile([0.001, 0.001], (n, 1)), dtype=torch.float64)
        result = self.context.analyze_convergence(hvs, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'converged')
        self.assertFalse(result['noise_limited'])

    def test_analyze_convergence_noise_limited(self):
        n = 20
        hvs = [0.8] * n
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_x[0] = torch.tensor([0.0, 0.0]); train_x[1] = torch.tensor([1.0, 1.0])
        train_obj = torch.tensor(np.tile([0.95, 0.9], (n, 1)), dtype=torch.float64)
        train_obj_std = torch.tensor(np.tile([0.3, 0.3], (n, 1)), dtype=torch.float64)
        result = self.context.analyze_convergence(hvs, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'converged_noise_limited')
        self.assertTrue(result['noise_limited'])

    def test_analyze_convergence_stuck_suboptimal(self):
        n = 20
        hvs = [0.1] * 10 + [0.1000000001] * 10
        train_x = torch.full((n, 2), 0.5, dtype=torch.float64)
        train_obj = torch.tensor(np.tile([0.1, 0.1], (n, 1)), dtype=torch.float64)
        train_obj_std = torch.tensor(np.tile([0.001, 0.001], (n, 1)), dtype=torch.float64)
        result = self.context.analyze_convergence(hvs, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'stuck_suboptimal')
        self.assertTrue(result['needs_restart'])

    def test_analyze_convergence_stagnant_warning(self):
        n = 20
        hvs = [0.1] * 10 + [0.1000000001] * 10
        train_x = torch.tensor(np.random.RandomState(2).uniform(0.3, 0.7, (n, 2)), dtype=torch.float64)
        train_obj = torch.tensor(np.tile([0.5, 0.45], (n, 1)), dtype=torch.float64)
        train_obj_std = torch.tensor(np.tile([0.001, 0.001], (n, 1)), dtype=torch.float64)
        result = self.context.analyze_convergence(hvs, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'stagnant_warning')

    # -- analyze_convergence_single_objective -----------------------------------------

    def test_analyze_convergence_single_objective_insufficient_data(self):
        result = self.context.analyze_convergence_single_objective([0.1] * 5, torch.rand(5, 1, dtype=torch.float64),
                                                                     torch.rand(5, 1, dtype=torch.float64), torch.rand(5, 2, dtype=torch.float64), iteration=5)
        self.assertEqual(result['status'], 'insufficient_data')

    def test_analyze_convergence_single_objective_converged(self):
        n = 20
        bestf = [0.9] * n
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_x[0] = torch.tensor([0.0, 0.0]); train_x[1] = torch.tensor([1.0, 1.0])
        train_obj = torch.full((n, 1), 0.9, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.001, dtype=torch.float64)
        result = self.context.analyze_convergence_single_objective(bestf, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'converged')

    def test_analyze_convergence_single_objective_noise_limited(self):
        n = 20
        bestf = [0.9] * n
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_x[0] = torch.tensor([0.0, 0.0]); train_x[1] = torch.tensor([1.0, 1.0])
        train_obj = torch.full((n, 1), 0.9, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.3, dtype=torch.float64)
        result = self.context.analyze_convergence_single_objective(bestf, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'converged_noise_limited')

    def test_analyze_convergence_single_objective_stuck_suboptimal(self):
        n = 20
        bestf = [0.1] * 10 + [0.1000000001] * 10
        train_x = torch.full((n, 2), 0.5, dtype=torch.float64)
        train_obj = torch.full((n, 1), 0.9, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.001, dtype=torch.float64)
        result = self.context.analyze_convergence_single_objective(bestf, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'stuck_suboptimal')

    def test_analyze_convergence_single_objective_in_progress(self):
        n = 20
        bestf = list(np.linspace(0.1, 0.9, n))
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (n, 2)), dtype=torch.float64)
        train_x[0] = torch.tensor([0.0, 0.0]); train_x[1] = torch.tensor([1.0, 1.0])
        train_obj = torch.full((n, 1), 0.9, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.001, dtype=torch.float64)
        result = self.context.analyze_convergence_single_objective(bestf, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'in_progress')

    def test_analyze_convergence_single_objective_stagnant_warning(self):
        n = 20
        bestf = [0.1] * 10 + [0.1000000001] * 10
        train_x = torch.tensor(np.random.RandomState(2).uniform(0.3, 0.7, (n, 2)), dtype=torch.float64)
        train_obj = torch.full((n, 1), 0.9, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.001, dtype=torch.float64)
        result = self.context.analyze_convergence_single_objective(bestf, train_obj, train_obj_std, train_x, iteration=20)
        self.assertEqual(result['status'], 'stagnant_warning')

    # -- _analyze_pareto_front / _analyze_parameter_coverage / _estimate_acquisition_diversity --

    def test_analyze_pareto_front_uses_cached_data(self):
        self.context._cached_pareto_data = {'pareto_ratio': 0.5, 'pareto_quality': 0.7, 'pareto_spread': 0.1, 'n_pareto_points': 3}
        result = self.context._analyze_pareto_front(np.random.rand(5, 2))
        self.assertEqual(result['pareto_quality'], 0.7)
        self.assertIsNone(self.context._cached_pareto_data)

    def test_analyze_pareto_front_computed_from_values(self):
        fitness = np.array([[0.9, 0.8], [0.5, 0.5], [0.95, 0.85]])
        result = self.context._analyze_pareto_front(fitness)
        self.assertGreaterEqual(result['n_pareto_points'], 1)

    def test_analyze_pareto_front_single_point(self):
        fitness = np.array([[0.9, 0.9], [0.1, 0.1]])
        result = self.context._analyze_pareto_front(fitness)
        self.assertEqual(result['n_pareto_points'], 1)

    def test_analyze_parameter_coverage_full_range(self):
        train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64)
        result = self.context._analyze_parameter_coverage(train_x)
        self.assertAlmostEqual(result['coverage'], 1.0, places=5)
        self.assertEqual(len(result['coverage_per_dim']), 2)

    def test_analyze_parameter_coverage_single_sample(self):
        train_x = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        result = self.context._analyze_parameter_coverage(train_x)
        self.assertEqual(result['uniformity'], 0.0)

    def test_estimate_acquisition_diversity_small_n(self):
        train_x = torch.rand(5, 2, dtype=torch.float64)
        self.assertEqual(self.context._estimate_acquisition_diversity(train_x), 1.0)

    def test_estimate_acquisition_diversity_normal(self):
        train_x = torch.tensor(np.random.RandomState(0).uniform(0, 1, (20, 2)), dtype=torch.float64)
        diversity = self.context._estimate_acquisition_diversity(train_x)
        self.assertGreaterEqual(diversity, 0.0)
        self.assertLessEqual(diversity, 1.0)


class _FakeProcessPoolExecutor:
    """Synchronous stand-in for ProcessPoolExecutor so tests avoid real multiprocessing/pickling."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, fn, *iterables):
        return [fn(*args) for args in zip(*iterables)]


class TestModuleLevelHelpers(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('test_logger_helpers')
        self.logger.setLevel(logging.CRITICAL)

    def test_convergence_status_to_json_none_or_empty(self):
        self.assertIsNone(_convergence_status_to_json(None))
        self.assertIsNone(_convergence_status_to_json({}))

    def test_convergence_status_to_json_serializes_numpy_types(self):
        result = {'converged': True, 'coverage': np.float64(0.5), 'coverage_per_dim': np.array([0.1, 0.2])}
        parsed = json.loads(_convergence_status_to_json(result))
        self.assertEqual(parsed['coverage'], 0.5)
        self.assertEqual(parsed['coverage_per_dim'], [0.1, 0.2])

    def test_fit_gp_models_independent(self):
        n = 6
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 2, dtype=torch.float64)
        train_obj_std = torch.full((n, 2), 0.05, dtype=torch.float64)
        model = _fit_gp_models(train_x, train_obj, train_obj_std, use_correlated_gp=False, logger=self.logger)
        self.assertIsInstance(model, ModelListGP)

    def test_fit_gp_models_correlated(self):
        n = 6
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 2, dtype=torch.float64)
        train_obj_std = torch.full((n, 2), 0.05, dtype=torch.float64)
        model = _fit_gp_models(train_x, train_obj, train_obj_std, use_correlated_gp=True, logger=self.logger)
        self.assertIsInstance(model, MultiTaskGP)

    def test_fit_gp_models_correlated_single_objective_falls_back(self):
        n = 6
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 1, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.05, dtype=torch.float64)
        model = _fit_gp_models(train_x, train_obj, train_obj_std, use_correlated_gp=True, logger=self.logger)
        self.assertIsInstance(model, ModelListGP)

    def test_optimize_acquisition_function_and_extract_pareto(self):
        n = 6
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 2, dtype=torch.float64)
        train_obj_std = torch.full((n, 2), 0.05, dtype=torch.float64)
        model = _fit_gp_models(train_x, train_obj, train_obj_std, use_correlated_gp=False, logger=self.logger)
        ctx = MagicMock()
        ctx.search_space = {'param1': {}, 'param2': {}}
        ctx.samples_per_batch_act_func = 4
        ctx.ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
        ctx.batch_size_per_iteration = 1
        ctx.num_restarts_act_func = 2
        ctx.raw_samples_act_func = 8
        ctx.logger = self.logger
        candidates = _optimize_acquisition_function(model, train_x, ctx)
        self.assertEqual(tuple(candidates.shape), (1, 2))
        self.assertIsNotNone(ctx._cached_pareto_data)
        self.assertIn('hypervolume', ctx._cached_pareto_data)

    def test_optimize_acquisition_function_error_propagates(self):
        ctx = MagicMock()
        ctx.search_space = {'param1': {}, 'param2': {}}
        ctx.samples_per_batch_act_func = 4
        ctx.ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
        ctx.batch_size_per_iteration = 1
        ctx.num_restarts_act_func = 2
        ctx.raw_samples_act_func = 8
        ctx.logger = self.logger
        model = MagicMock()
        with patch('uq_physicell.bo.bo_context.optimize_acqf', side_effect=RuntimeError('boom')), \
             patch('uq_physicell.bo.bo_context.qLogNoisyExpectedHypervolumeImprovement'):
            with self.assertRaises(RuntimeError):
                _optimize_acquisition_function(model, torch.rand(4, 2, dtype=torch.float64), ctx)

    def test_extract_pareto_and_hypervolume_no_partitioning(self):
        ctx = MagicMock(); ctx.logger = self.logger
        acq = MagicMock(spec=[])
        self.assertIsNone(_extract_pareto_and_hypervolume_from_acqf(acq, ctx))

    def test_extract_pareto_and_hypervolume_success(self):
        ctx = MagicMock(); ctx.logger = self.logger
        acq = MagicMock()
        acq.partitioning.pareto_Y = torch.tensor([[0.5, 0.6], [0.7, 0.4]], dtype=torch.float64)
        acq._hypervolumes = torch.tensor([0.3, 0.35])
        result = _extract_pareto_and_hypervolume_from_acqf(acq, ctx)
        self.assertEqual(result['n_pareto_points'], 2)
        self.assertIsNotNone(result['hypervolume'])

    def test_extract_pareto_and_hypervolume_list_pareto_y(self):
        ctx = MagicMock(); ctx.logger = self.logger
        acq = MagicMock()
        acq.partitioning.pareto_Y = [torch.tensor([[0.5, 0.6]], dtype=torch.float64)]
        acq._hypervolumes = torch.tensor(0.2)
        result = _extract_pareto_and_hypervolume_from_acqf(acq, ctx)
        self.assertEqual(result['n_pareto_points'], 1)

    def test_extract_pareto_and_hypervolume_exception_returns_none(self):
        ctx = MagicMock(); ctx.logger = self.logger
        acq = MagicMock()
        acq.partitioning.pareto_Y = "not-a-tensor"
        self.assertIsNone(_extract_pareto_and_hypervolume_from_acqf(acq, ctx))

    def test_extract_pareto_and_hypervolume_hv_extraction_failure(self):
        ctx = MagicMock(); ctx.logger = self.logger
        acq = MagicMock()
        acq.partitioning.pareto_Y = torch.tensor([[0.5, 0.6], [0.7, 0.4]], dtype=torch.float64)
        acq._hypervolumes = "bad"
        self.assertIsNone(_extract_pareto_and_hypervolume_from_acqf(acq, ctx))


class TestBayesianOptimizationLoops(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger('test_logger_loops')
        self.logger.setLevel(logging.CRITICAL)

    def _make_ctx(self, n_qois):
        ctx = MagicMock()
        ctx.logger = self.logger
        ctx.batch_size_bo = 1
        ctx.batch_size_per_iteration = 1
        ctx.num_restarts_act_func = 2
        ctx.raw_samples_act_func = 8
        ctx.samples_per_batch_act_func = 4
        ctx.search_space = {
            'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
            'param2': {'type': 'real', 'lower_bound': 0.5, 'upper_bound': 2.0},
        }
        ctx.db_path = '/fake/db.db'
        ctx.workers_out = 1
        ctx.cancel_requested = False
        ctx.qoi_details = {'QOI_Name': [f'obj{i}' for i in range(1, n_qois + 1)]}
        return ctx

    def test_single_objective_loop_runs_one_iteration(self):
        ctx = self._make_ctx(1)
        ctx.evaluate_params = MagicMock(return_value=({'obj1': 0.9}, {'obj1': 0.01}, {}, [1]))
        ctx.save_results_to_db = MagicMock()
        n = 8
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 1, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.02, dtype=torch.float64)
        with patch('uq_physicell.bo.bo_context.concurrent.futures.ProcessPoolExecutor', _FakeProcessPoolExecutor), \
             patch('uq_physicell.bo.bo_context.insert_samples') as mock_samples, \
             patch('uq_physicell.bo.bo_context.insert_gp_models') as mock_gp:
            single_objective_bayesian_optimization(ctx, train_x, train_obj, train_obj_std, start_iteration=1)
        mock_samples.assert_called_once()
        mock_gp.assert_called_once()
        ctx.save_results_to_db.assert_called_once()

    def test_single_objective_loop_cancellation_skips_iteration(self):
        ctx = self._make_ctx(1)
        ctx.cancel_requested = True
        n = 8
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 1, dtype=torch.float64)
        train_obj_std = torch.full((n, 1), 0.02, dtype=torch.float64)
        with patch('uq_physicell.bo.bo_context.insert_gp_models') as mock_gp:
            single_objective_bayesian_optimization(ctx, train_x, train_obj, train_obj_std, start_iteration=1)
        mock_gp.assert_not_called()

    def test_multi_objective_loop_runs_one_iteration(self):
        ctx = self._make_ctx(2)
        ctx.ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
        ctx.use_correlated_gp = False
        ctx.evaluate_params = MagicMock(return_value=({'obj1': 0.9, 'obj2': 0.8}, {'obj1': 0.01, 'obj2': 0.01}, {}, [1]))
        ctx.save_results_to_db = MagicMock()
        n = 8
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 2, dtype=torch.float64)
        train_obj_std = torch.full((n, 2), 0.02, dtype=torch.float64)
        with patch('uq_physicell.bo.bo_context.concurrent.futures.ProcessPoolExecutor', _FakeProcessPoolExecutor), \
             patch('uq_physicell.bo.bo_context.insert_samples') as mock_samples, \
             patch('uq_physicell.bo.bo_context.insert_gp_models') as mock_gp:
            multi_objective_bayesian_optimization(
                ctx, train_x, train_obj, train_obj_std, start_iteration=1, latest_hypervolume=0.0, resume_from_db=False
            )
        mock_samples.assert_called_once()
        mock_gp.assert_called_once()
        ctx.save_results_to_db.assert_called_once()

    def test_multi_objective_loop_cancellation_skips_iteration(self):
        ctx = self._make_ctx(2)
        ctx.ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
        ctx.cancel_requested = True
        n = 8
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_obj = torch.rand(n, 2, dtype=torch.float64)
        train_obj_std = torch.full((n, 2), 0.02, dtype=torch.float64)
        with patch('uq_physicell.bo.bo_context.insert_gp_models') as mock_gp:
            multi_objective_bayesian_optimization(
                ctx, train_x, train_obj, train_obj_std, start_iteration=1, latest_hypervolume=0.5, resume_from_db=True
            )
        mock_gp.assert_not_called()


if __name__ == '__main__':
    unittest.main()