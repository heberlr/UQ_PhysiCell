#!/usr/bin/env python
"""Test suite for the BO calibration database module (uq_physicell.database.bo_db)."""

import io
import json
import os
import pickle
import sqlite3
import tempfile

import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP

from uq_physicell.database.bo_db import (
    create_structure,
    _apply_schema_migrations,
    migrate_bo_database,
    insert_metadata,
    insert_param_space,
    insert_qois,
    insert_gp_models,
    insert_samples,
    insert_output,
    load_metadata,
    load_parameter_space,
    load_qois,
    load_gp_models,
    load_samples,
    load_output,
    load_structure,
)


def make_gp_model():
    X = torch.rand(5, 1, dtype=torch.double)
    Y = torch.sin(X * 2 * torch.pi)
    gp = SingleTaskGP(X, Y)
    return ModelListGP(gp)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sample_database(db_path):
    create_structure(db_path)

    metadata = {
        'BO_Method': 'Bayesian Optimization',
        'ObsData_Path': 'observed_data.csv',
        'Ini_File_Path': 'config.ini',
        'StructureName': 'PhysiCell',
        'BO_Options': json.dumps({'num_iterations': 10}),
        'Ini_Hash': 'ini_hash',
        'XML_Hash': 'xml_hash',
        'Rules_Hash': 'rules_hash',
        'Structure_Config_Hash': 'struct_hash',
        'Effective_Run_Hash': 'effective_hash',
    }
    insert_metadata(db_path, metadata)

    param_space = {
        'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
        'param2': {'type': 'real', 'lower_bound': 1.0, 'upper_bound': 5.0, 'regulates': 'param1'},
    }
    insert_param_space(db_path, param_space)

    qois = {
        'QOI_Name': ['total_cells', 'max_radius'],
        'QOI_Function': ["lambda data: data['cell_count'].sum()", "lambda data: data['radius'].max()"],
        'ObsData_Column': ['Cells', 'Radius'],
        'QoI_distanceFunction': ['SSE', 'SSE'],
        'QoI_distanceWeight': [1.0, 2.0],
    }
    insert_qois(db_path, qois)

    gp_model = make_gp_model()
    insert_gp_models(db_path, iteration_id=0, gp_model=gp_model, score=0.5, convergence_status=json.dumps({'converged': False}))
    insert_gp_models(db_path, iteration_id=1, gp_model=gp_model, score=0.8)

    samples = {
        0: {'param1': 0.5, 'param2': 2.0},
        1: {'param1': 0.3, 'param2': 3.0},
    }
    insert_samples(db_path, iteration_id=0, samples=samples)

    for sample_id in (0, 1):
        obj_func = pickle.dumps([0.1 * sample_id])
        noise_std = pickle.dumps([0.01])
        data = pickle.dumps(pd.DataFrame({'time': [0, 1], 'cell_count': [1, 2]}))
        insert_output(db_path, sample_id, obj_func, noise_std, data, seeds=json.dumps([1000 + sample_id]))

    return db_path


class TestCreateStructure:
    def test_creates_all_tables(self, db_path):
        create_structure(db_path)
        conn = sqlite3.connect(db_path)
        try:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        finally:
            conn.close()
        expected = {'Metadata', 'ParameterSpace', 'QoIs', 'GP_Models', 'Samples', 'Output'}
        assert expected.issubset(tables)

    def test_idempotent(self, db_path):
        create_structure(db_path)
        create_structure(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(GP_Models)").fetchall()]
        finally:
            conn.close()
        assert 'Score' in cols
        assert 'ConvergenceStatus' in cols

    def test_invalid_path_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            create_structure('/nonexistent_dir_xyz/does_not_exist/db.sqlite')


class TestSchemaMigrations:
    def _legacy_db(self, path):
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE Metadata (
                BO_Method TEXT PRIMARY KEY,
                ObsData_Path TEXT,
                Ini_File_Path TEXT,
                StructureName TEXT,
                uq_physicell_version TEXT,
                pcdl_version TEXT,
                botorch_version TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE ParameterSpace (
                ParamName TEXT PRIMARY KEY,
                Type TEXT,
                Lower_Bound DOUBLE,
                Upper_Bound DOUBLE,
                Regulates TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE QoIs (
                QOI_Name TEXT PRIMARY KEY,
                QOI_Function TEXT,
                ObsData_Column TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE GP_Models (
                IterationID INTEGER,
                GP_Model BLOB,
                Hypervolume DOUBLE,
                PRIMARY KEY (IterationID)
            )
        """)
        cursor.execute("""
            CREATE TABLE Samples (
                IterationID INTEGER,
                SampleID INTEGER,
                ParamName TEXT,
                ParamValue DOUBLE,
                PRIMARY KEY (IterationID, SampleID, ParamName)
            )
        """)
        cursor.execute("""
            CREATE TABLE Output (
                SampleID INTEGER,
                ObjFunc BLOB,
                Noise_Std BLOB,
                Data BLOB,
                PRIMARY KEY (SampleID)
            )
        """)
        conn.commit()
        conn.close()

    def test_migrations_add_missing_columns(self, db_path):
        self._legacy_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        applied = _apply_schema_migrations(cursor)
        conn.commit()
        conn.close()

        assert 'Metadata.BO_Options added' in applied
        assert any('Hypervolume renamed to Score' in a for a in applied)
        assert 'GP_Models.ConvergenceStatus added' in applied
        assert 'Output.Seeds added' in applied

        conn = sqlite3.connect(db_path)
        try:
            meta_cols = [row[1] for row in conn.execute("PRAGMA table_info(Metadata)").fetchall()]
            gp_cols = [row[1] for row in conn.execute("PRAGMA table_info(GP_Models)").fetchall()]
            out_cols = [row[1] for row in conn.execute("PRAGMA table_info(Output)").fetchall()]
        finally:
            conn.close()
        assert 'BO_Options' in meta_cols
        assert 'Ini_Hash' in meta_cols
        assert 'Score' in gp_cols
        assert 'Hypervolume' not in gp_cols
        assert 'ConvergenceStatus' in gp_cols
        assert 'Seeds' in out_cols

    def test_migrations_idempotent_on_current_schema(self, sample_database):
        conn = sqlite3.connect(sample_database)
        cursor = conn.cursor()
        applied = _apply_schema_migrations(cursor)
        conn.commit()
        conn.close()
        assert applied == []

    def test_create_structure_migrates_legacy_db(self, db_path):
        self._legacy_db(db_path)
        create_structure(db_path)
        conn = sqlite3.connect(db_path)
        try:
            gp_cols = [row[1] for row in conn.execute("PRAGMA table_info(GP_Models)").fetchall()]
        finally:
            conn.close()
        assert 'Score' in gp_cols
        assert 'Hypervolume' not in gp_cols


class TestMigrateBoDatabase:
    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError, match="not found"):
            migrate_bo_database('/tmp/does_not_exist_bo_db_xyz.db')

    def test_not_a_bo_database_raises(self, db_path):
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE SomeOtherTable (id INTEGER)")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="not a BO calibration database"):
            migrate_bo_database(db_path)

    def test_already_current_returns_empty_list(self, sample_database):
        applied = migrate_bo_database(sample_database)
        assert applied == []

    def test_migrates_legacy_database(self, db_path):
        TestSchemaMigrations()._legacy_db(db_path)
        applied = migrate_bo_database(db_path)
        assert len(applied) > 0
        # Calling again should now be a no-op
        applied_again = migrate_bo_database(db_path)
        assert applied_again == []


class TestInsertAndLoad:
    def test_load_metadata(self, sample_database):
        df = load_metadata(sample_database)
        assert df.shape[0] == 1
        assert df['BO_Method'].values[0] == 'Bayesian Optimization'
        assert df['ObsData_Path'].values[0] == 'observed_data.csv'
        assert df['Ini_Hash'].values[0] == 'ini_hash'

    def test_insert_metadata_replaces_existing(self, sample_database):
        metadata = {
            'BO_Method': 'Bayesian Optimization',
            'ObsData_Path': 'new_data.csv',
            'Ini_File_Path': 'config2.ini',
            'StructureName': 'PhysiCell',
        }
        insert_metadata(sample_database, metadata)
        df = load_metadata(sample_database)
        assert df.shape[0] == 1
        assert df['ObsData_Path'].values[0] == 'new_data.csv'
        assert df['BO_Options'].values[0] is None

    def test_load_parameter_space(self, sample_database):
        df = load_parameter_space(sample_database)
        assert df.shape[0] == 2
        assert set(df['ParamName']) == {'param1', 'param2'}
        row2 = df[df['ParamName'] == 'param2'].iloc[0]
        assert row2['regulates'] == 'param1'
        row1 = df[df['ParamName'] == 'param1'].iloc[0]
        assert row1['regulates'] is None

    def test_load_qois(self, sample_database):
        df = load_qois(sample_database)
        assert df.shape[0] == 2
        assert set(df['QoI_Name']) == {'total_cells', 'max_radius'}
        assert set(df['QoI_distanceWeight']) == {1.0, 2.0}

    def test_load_gp_models(self, sample_database):
        df = load_gp_models(sample_database)
        assert df.shape[0] == 2
        assert set(df['IterationID']) == {0, 1}
        row0 = df[df['IterationID'] == 0].iloc[0]
        assert row0['Score'] == 0.5
        assert row0['ConvergenceStatus'] == {'converged': False}
        row1 = df[df['IterationID'] == 1].iloc[0]
        assert row1['ConvergenceStatus'] is None
        assert row0['GP_Model'] is not None

    def test_insert_gp_models_replace(self, sample_database):
        gp_model = make_gp_model()
        insert_gp_models(sample_database, iteration_id=0, gp_model=gp_model, score=0.99)
        df = load_gp_models(sample_database)
        row0 = df[df['IterationID'] == 0].iloc[0]
        assert row0['Score'] == 0.99

    def test_load_samples_all(self, sample_database):
        df = load_samples(sample_database)
        assert df.shape[0] == 4  # 2 samples * 2 params
        assert set(df['ParamName']) == {'param1', 'param2'}

    def test_load_samples_filtered(self, sample_database):
        insert_samples(sample_database, iteration_id=1, samples={2: {'param1': 0.9, 'param2': 4.5}})
        df_iter0 = load_samples(sample_database, iteration_ids=[0])
        assert set(df_iter0['IterationID']) == {0}
        df_iter1 = load_samples(sample_database, iteration_ids=[1])
        assert set(df_iter1['IterationID']) == {1}
        assert df_iter1.shape[0] == 2

    def test_load_output_full(self, sample_database):
        df = load_output(sample_database)
        assert df.shape[0] == 2
        assert isinstance(df['Data'].iloc[0], pd.DataFrame)
        assert df['Seeds'].iloc[0] == [1000]
        assert df['Seeds'].iloc[1] == [1001]

    def test_load_output_no_data(self, sample_database):
        df = load_output(sample_database, load_data=False)
        assert list(df.columns) == ['SampleID']
        assert df.shape[0] == 2

    def test_load_output_filtered(self, sample_database):
        df = load_output(sample_database, sample_ids=[1])
        assert df.shape[0] == 1
        assert df['SampleID'].values[0] == 1

    def test_load_output_null_seeds(self, sample_database):
        insert_output(sample_database, 2, pickle.dumps([1.0]), pickle.dumps([0.1]),
                      pickle.dumps(pd.DataFrame({'a': [1]})), seeds=None)
        df = load_output(sample_database, sample_ids=[2])
        assert df['Seeds'].iloc[0] is None


class TestLoadStructure:
    def test_full_load(self, sample_database):
        metadata, params, qois, gp_models, samples, output = load_structure(sample_database)
        assert metadata.shape[0] == 1
        assert params.shape[0] == 2
        assert qois.shape[0] == 2
        assert gp_models.shape[0] == 2
        assert samples.shape[0] == 4
        assert output.shape[0] == 2
        assert isinstance(output['Data'].iloc[0], pd.DataFrame)

    def test_load_data_false(self, sample_database):
        metadata, params, qois, gp_models, samples, output = load_structure(sample_database, load_data=False)
        assert gp_models.empty
        assert list(gp_models.columns) == ['IterationID', 'GP_Model', 'Score', 'ConvergenceStatus']
        assert list(output.columns) == ['SampleID']
        assert samples.shape[0] == 4

    def test_migrates_legacy_by_default(self, db_path):
        TestSchemaMigrations()._legacy_db(db_path)
        with pytest.warns(UserWarning, match="Upgraded"):
            metadata, params, qois, gp_models, samples, output = load_structure(db_path, load_data=False)
        assert isinstance(metadata, pd.DataFrame)

    def test_migrate_false_skips_migration(self, db_path):
        TestSchemaMigrations()._legacy_db(db_path)
        # Legacy Metadata table has no BO_Options column; load_metadata should still work
        metadata, params, qois, gp_models, samples, output = load_structure(db_path, load_data=False, migrate=False)
        assert isinstance(metadata, pd.DataFrame)
        conn = sqlite3.connect(db_path)
        try:
            gp_cols = [row[1] for row in conn.execute("PRAGMA table_info(GP_Models)").fetchall()]
        finally:
            conn.close()
        # Not migrated: legacy Hypervolume column is still present
        assert 'Hypervolume' in gp_cols
        assert 'Score' not in gp_cols


class TestInsertErrorHandling:
    def test_insert_metadata_missing_key_raises(self, db_path):
        create_structure(db_path)
        with pytest.raises(KeyError):
            insert_metadata(db_path, {'BO_Method': 'x'})

    def test_insert_param_space_bad_table_raises_runtime_error(self, db_path):
        # No create_structure called: ParameterSpace table doesn't exist.
        with pytest.raises(RuntimeError):
            insert_param_space(db_path, {'p': {'type': 'real', 'lower_bound': 0, 'upper_bound': 1}})

    def test_insert_output_duplicate_sample_id_raises(self, sample_database):
        with pytest.raises(RuntimeError):
            insert_output(sample_database, 0, pickle.dumps([1]), pickle.dumps([1]), pickle.dumps({'a': 1}))
