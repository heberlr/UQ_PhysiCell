#!/usr/bin/env python
"""Test suite for Model Analysis database modular load functions."""

import os
import tempfile
import pickle
import pytest
import pandas as pd
import numpy as np

from uq_physicell.database.ma_db import (
    create_structure,
    insert_metadata,
    insert_param_space,
    insert_qois,
    insert_samples,
    insert_output,
    load_metadata,
    load_parameter_space,
    load_qois,
    load_samples,
    load_output,
    load_data_unserialized,
    load_structure
)


@pytest.fixture
def sample_database():
    """Create a temporary database with sample data for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_file = tmp.name
    
    # Create structure
    create_structure(db_file)
    
    # Insert test data
    insert_metadata(db_file, "Sobol", "/path/to/config.ini", "test_model")
    
    params = {
        'param1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'ref_value': 0.5, 'perturbation': [0.1, 0.2]},
        'param2': {'lower_bound': 0.0, 'upper_bound': 10.0, 'ref_value': 5.0, 'perturbation': [1.0, 2.0]}
    }
    insert_param_space(db_file, params)
    
    qois = {
        'total_cells': 'lambda data: data["cells"].sum()', 
        'max_radius': 'lambda data: data["radius"].max()'
    }
    insert_qois(db_file, qois)
    
    samples = {
        0: {'param1': 0.3, 'param2': 3.0},
        1: {'param1': 0.7, 'param2': 7.0}
    }
    insert_samples(db_file, samples)
    
    # Create mock output data
    for sample_id in [0, 1]:
        for replicate_id in [0, 1]:
            data = pd.DataFrame({
                'time': [0, 1, 2],
                'total_cells': [10, 20, 30],
                'max_radius': [5.0, 10.0, 15.0]
            })
            serialized = pickle.dumps(data)
            insert_output(db_file, sample_id, replicate_id, serialized)
    
    yield db_file
    
    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)


class TestModularLoadFunctions:
    """Test suite for modular database load functions."""
    
    def test_load_metadata(self, sample_database):
        """Test load_metadata function."""
        df_metadata = load_metadata(sample_database)
        assert df_metadata.shape[0] == 1
        assert df_metadata['Sampler'].values[0] == 'Sobol'
        assert df_metadata['Ini_File_Path'].values[0] == '/path/to/config.ini'
        assert df_metadata['StructureName'].values[0] == 'test_model'
    
    def test_load_parameter_space(self, sample_database):
        """Test load_parameter_space function."""
        df_params = load_parameter_space(sample_database)
        assert df_params.shape[0] == 2
        assert 'param1' in df_params['ParamName'].values
        assert 'param2' in df_params['ParamName'].values
        
        # Check that perturbation is converted to numpy array
        assert isinstance(df_params['perturbation'].iloc[0], np.ndarray)
        assert len(df_params['perturbation'].iloc[0]) == 2
    
    def test_load_qois(self, sample_database):
        """Test load_qois function."""
        df_qois = load_qois(sample_database)
        assert df_qois.shape[0] == 2
        assert 'total_cells' in df_qois['QOI_Name'].values
        assert 'max_radius' in df_qois['QOI_Name'].values
    
    def test_load_samples(self, sample_database):
        """Test load_samples function."""
        dic_samples = load_samples(sample_database)
        assert len(dic_samples) == 2
        assert dic_samples[0]['param1'] == 0.3
        assert dic_samples[0]['param2'] == 3.0
        assert dic_samples[1]['param1'] == 0.7
        assert dic_samples[1]['param2'] == 7.0
    
    def test_load_output_full(self, sample_database):
        """Test load_output with full data loading."""
        df_output = load_output(sample_database, load_data=True)
        assert df_output.shape[0] == 4  # 2 samples * 2 replicates
        assert 'SampleID' in df_output.columns
        assert 'ReplicateID' in df_output.columns
        assert 'Data' in df_output.columns
        
        # Check that data is deserialized
        assert isinstance(df_output['Data'].iloc[0], pd.DataFrame)
    
    def test_load_output_metadata_only(self, sample_database):
        """Test load_output with metadata only (no data loading)."""
        df_output = load_output(sample_database, load_data=False)
        assert df_output.shape[0] == 4
        assert 'SampleID' in df_output.columns
        assert 'ReplicateID' in df_output.columns
        assert df_output['Data'] is not None
    
    def test_load_output_filter_by_sample(self, sample_database):
        """Test load_output with sample_ids filter."""
        df_output = load_output(sample_database, sample_ids=[0], load_data=True)
        assert df_output.shape[0] == 2  # Only sample 0, both replicates
        assert all(df_output['SampleID'] == 0)
    
    def test_load_output_filter_by_replicate(self, sample_database):
        """Test load_output with replicate_ids filter."""
        df_output = load_output(sample_database, replicate_ids=[0], load_data=True)
        assert df_output.shape[0] == 2  # Only replicate 0, both samples
        assert all(df_output['ReplicateID'] == 0)
    
    def test_load_output_filter_combined(self, sample_database):
        """Test load_output with both sample_ids and replicate_ids filters."""
        df_output = load_output(sample_database, sample_ids=[1], replicate_ids=[1], load_data=True)
        assert df_output.shape[0] == 1
        assert df_output['SampleID'].values[0] == 1
        assert df_output['ReplicateID'].values[0] == 1
    
    def test_load_data_unserialized_full(self, sample_database):
        """Test load_data_unserialized with full data."""
        df_unserialized = load_data_unserialized(sample_database)
        assert df_unserialized.shape[0] == 4
        
        # Check that QoI columns are expanded
        assert 'total_cells_0' in df_unserialized.columns
        assert 'total_cells_1' in df_unserialized.columns
        assert 'total_cells_2' in df_unserialized.columns
        assert 'time_0' in df_unserialized.columns
        assert 'max_radius_0' in df_unserialized.columns
    
    def test_load_data_unserialized_filtered(self, sample_database):
        """Test load_data_unserialized with sample filter."""
        df_unserialized = load_data_unserialized(sample_database, sample_ids=[0])
        assert df_unserialized.shape[0] == 2
        assert all(df_unserialized['SampleID'] == 0)
    
    def test_load_structure_backward_compatibility_full(self, sample_database):
        """Test load_structure with load_result=True (backward compatibility)."""
        metadata, params, qois, samples, results = load_structure(sample_database, load_result=True)
        
        assert isinstance(metadata, pd.DataFrame)
        assert metadata.shape[0] == 1
        
        assert isinstance(params, pd.DataFrame)
        assert params.shape[0] == 2
        
        assert isinstance(qois, pd.DataFrame)
        assert qois.shape[0] == 2
        
        assert isinstance(samples, dict)
        assert len(samples) == 2
        
        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] == 4
        assert 'total_cells_0' in results.columns  # QoI expansion
    
    def test_load_structure_backward_compatibility_metadata(self, sample_database):
        """Test load_structure with load_result=False (backward compatibility)."""
        metadata, params, qois, samples, ids = load_structure(sample_database, load_result=False)
        
        assert isinstance(metadata, pd.DataFrame)
        assert isinstance(params, pd.DataFrame)
        assert isinstance(qois, pd.DataFrame)
        assert isinstance(samples, dict)
        assert isinstance(ids, pd.DataFrame)
        
        assert ids.shape[0] == 4
        assert 'SampleID' in ids.columns
        assert 'ReplicateID' in ids.columns
        assert ids['Data'] is not None


class TestDatabaseCreation:
    """Test suite for database creation functions."""
    
    def test_create_structure(self):
        """Test create_structure function."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_file = tmp.name
        
        try:
            create_structure(db_file)
            assert os.path.exists(db_file)
            
            # Verify tables exist
            import sqlite3
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            assert 'Metadata' in tables
            assert 'ParameterSpace' in tables
            assert 'QoIs' in tables
            assert 'Samples' in tables
            assert 'Output' in tables
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)
    
    def test_insert_functions(self):
        """Test all insert functions."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_file = tmp.name
        
        try:
            create_structure(db_file)
            
            # Test insert_metadata
            insert_metadata(db_file, "TestSampler", "/test/path", "test_struct")
            df = load_metadata(db_file)
            assert df['Sampler'].values[0] == 'TestSampler'
            
            # Test insert_param_space
            params = {'p1': {'lower_bound': 0, 'upper_bound': 1, 'ref_value': 0.5, 'perturbation': [0.1]}}
            insert_param_space(db_file, params)
            df = load_parameter_space(db_file)
            assert df.shape[0] == 1
            
            # Test insert_qois
            qois = {'qoi1': 'lambda x: x'}
            insert_qois(db_file, qois)
            df = load_qois(db_file)
            assert df.shape[0] == 1
            
            # Test insert_samples
            samples = {0: {'p1': 0.5}}
            insert_samples(db_file, samples)
            dic = load_samples(db_file)
            assert len(dic) == 1
            
            # Test insert_output
            data = pd.DataFrame({'col': [1, 2, 3]})
            serialized = pickle.dumps(data)
            insert_output(db_file, 0, 0, serialized)
            df = load_output(db_file, load_data=False)
            assert df.shape[0] == 1
            
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
