#!/usr/bin/env python
"""Test suite for uq_physicell.database.utils (generic sqlite helper functions)."""

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from uq_physicell.database.utils import (
    update_db_value,
    add_db_entry,
    remove_db_entry,
    remove_db_table,
    _create_table,
    _alter_table_add_column,
    download_file,
    get_database_type,
)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        path = tmp.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def _make_table(db_path, table_name, columns_def, rows=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE {table_name} ({columns_def})")
    if rows:
        placeholders = ','.join('?' * len(rows[0]))
        cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", rows)
    conn.commit()
    conn.close()


def _fetch_all(db_path, query, params=()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


class TestUpdateDbValue:
    def test_updates_matching_rows(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER, name TEXT', rows=[(1, 'old'), (2, 'old')])
        update_db_value(db_path, 'T', 'name', 'new', 'old')
        rows = _fetch_all(db_path, "SELECT name FROM T")
        assert [r[0] for r in rows] == ['new', 'new']

    def test_no_match_leaves_table_unchanged(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER, name TEXT', rows=[(1, 'a')])
        update_db_value(db_path, 'T', 'name', 'new', 'does_not_exist')
        rows = _fetch_all(db_path, "SELECT name FROM T")
        assert [r[0] for r in rows] == ['a']

    def test_missing_table_raises_operational_error(self, db_path):
        with pytest.raises(sqlite3.OperationalError):
            update_db_value(db_path, 'NoSuchTable', 'col', 'new', 'old')


class TestCreateTableAndAlter:
    def test_create_table_creates_id_column(self, db_path):
        _create_table(db_path, 'Foo')
        cols = [row[1] for row in _fetch_all(db_path, "PRAGMA table_info(Foo)")]
        assert cols == ['id']

    def test_create_table_idempotent(self, db_path):
        _create_table(db_path, 'Foo')
        _create_table(db_path, 'Foo')  # should not raise
        tables = _fetch_all(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        assert [t[0] for t in tables].count('Foo') == 1

    def test_alter_table_add_column(self, db_path):
        _create_table(db_path, 'Foo')
        _alter_table_add_column(db_path, 'Foo', 'bar', 'TEXT')
        cols = [row[1] for row in _fetch_all(db_path, "PRAGMA table_info(Foo)")]
        assert 'bar' in cols

    def test_alter_table_default_applies_to_existing_rows(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER', rows=[(1,), (2,)])
        _alter_table_add_column(db_path, 'T', 'note', 'TEXT')
        rows = _fetch_all(db_path, "SELECT note FROM T")
        assert [r[0] for r in rows] == ['', '']


class TestAddDbEntry:
    def test_creates_table_and_column_when_missing(self, db_path):
        add_db_entry(db_path, 'Results', 'score', 3.14)
        rows = _fetch_all(db_path, "SELECT score FROM Results")
        assert rows == [(3.14,)]

    def test_appends_to_existing_column(self, db_path):
        add_db_entry(db_path, 'Results', 'score', 1.0)
        add_db_entry(db_path, 'Results', 'score', 2.0)
        rows = _fetch_all(db_path, "SELECT score FROM Results ORDER BY score")
        assert [r[0] for r in rows] == [1.0, 2.0]

    def test_adds_new_column_to_existing_table(self, db_path):
        add_db_entry(db_path, 'Results', 'score', 1.0)
        add_db_entry(db_path, 'Results', 'label', 'run1')
        cols = [row[1] for row in _fetch_all(db_path, "PRAGMA table_info(Results)")]
        assert 'score' in cols
        assert 'label' in cols

    def test_string_value_roundtrip(self, db_path):
        add_db_entry(db_path, 'Results', 'name', 'hello')
        rows = _fetch_all(db_path, "SELECT name FROM Results")
        assert rows == [('hello',)]


class TestRemoveDbEntry:
    def test_removes_matching_rows(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER, name TEXT', rows=[(1, 'a'), (2, 'b'), (3, 'a')])
        remove_db_entry(db_path, 'T', 'name', 'a')
        rows = _fetch_all(db_path, "SELECT id FROM T")
        assert [r[0] for r in rows] == [2]

    def test_no_match_is_noop(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER, name TEXT', rows=[(1, 'a')])
        remove_db_entry(db_path, 'T', 'name', 'missing')
        rows = _fetch_all(db_path, "SELECT id FROM T")
        assert [r[0] for r in rows] == [1]


class TestRemoveDbTable:
    def test_removes_existing_table(self, db_path):
        _make_table(db_path, 'T', 'id INTEGER')
        remove_db_table(db_path, 'T')
        tables = _fetch_all(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
        assert 'T' not in [t[0] for t in tables]

    def test_missing_table_is_noop(self, db_path):
        # Should not raise even though the table never existed.
        remove_db_table(db_path, 'DoesNotExist')


class TestDownloadFile:
    def test_skips_download_if_file_exists(self, tmp_path):
        existing = tmp_path / "already_here.txt"
        existing.write_text("data")
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_file(str(existing))
            mock_retrieve.assert_not_called()

    def test_downloads_with_default_base_url(self, tmp_path):
        target = tmp_path / "new_file.txt"
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_file(str(target))
            mock_retrieve.assert_called_once()
            args, _ = mock_retrieve.call_args
            assert args[0] == "https://zenodo.org/records/21496966/files/" + str(target)
            assert args[1] == str(target)

    def test_downloads_with_custom_url(self, tmp_path):
        target = tmp_path / "new_file2.txt"
        with patch('urllib.request.urlretrieve') as mock_retrieve:
            download_file(str(target), custom_url="https://example.com/custom_file")
            args, _ = mock_retrieve.call_args
            assert args[0] == "https://example.com/custom_file"


class TestGetDatabaseType:
    def test_model_analysis(self, db_path):
        _make_table(db_path, 'Metadata', 'Sampler TEXT, Ini_File_Path TEXT', [('Sobol', 'm.ini')])
        assert get_database_type(db_path) == 'MA'

    def test_bayesian_optimization(self, db_path):
        _make_table(db_path, 'Metadata', 'BO_Method TEXT, ObsData_Path TEXT', [('qNEHVI', 'obs.csv')])
        assert get_database_type(db_path) == 'BO'

    def test_abc_from_metadata(self, db_path):
        _make_table(db_path, 'Metadata', 'Method TEXT, ObsData_Path TEXT', [('ABC', 'obs.csv')])
        assert get_database_type(db_path) == 'ABC'

    def test_abc_from_pyabc_tables_without_metadata(self, db_path):
        _make_table(db_path, 'abc_smc', 'id INTEGER PRIMARY KEY')
        assert get_database_type(db_path) == 'ABC'

    def test_metadata_with_other_method_is_unknown(self, db_path):
        _make_table(db_path, 'Metadata', 'Method TEXT', [('Other',)])
        assert get_database_type(db_path) is None

    def test_no_metadata_table_is_unknown(self, db_path):
        _make_table(db_path, 'Samples', 'SampleID INTEGER')
        assert get_database_type(db_path) is None

    def test_missing_file(self, tmp_path):
        missing = tmp_path / 'missing.db'
        assert get_database_type(str(missing)) is None
        assert not missing.exists()  # must not create the file

    def test_non_sqlite_file(self, tmp_path):
        not_db = tmp_path / 'notes.db'
        not_db.write_text('this is not a sqlite database')
        assert get_database_type(str(not_db)) is None

    def test_backward_compatible_imports(self):
        from uq_physicell.database import get_database_type as from_package
        from uq_physicell.database.ma_db import get_database_type as from_ma_db
        assert from_package is get_database_type
        assert from_ma_db is get_database_type

