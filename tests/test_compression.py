"""Tests for uq_physicell.database.compression, in particular migrate_to_zstd's
handling of individual rows that fail to migrate."""

import os
import pickle
import sqlite3
import tempfile

import pandas as pd
import pytest

from uq_physicell.database import ma_db
from uq_physicell.database.compression import migrate_to_zstd


@pytest.fixture
def input_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_file = tmp.name
    ma_db.create_structure(db_file)
    ma_db.insert_metadata(db_file, "Sobol", "/path/to/config.ini", "test_model")
    ma_db.insert_samples(db_file, {0: {'p1': 0.5}, 1: {'p1': 0.7}})
    for sample_id, replicate_id, value in [(0, 0, 1), (1, 0, 2)]:
        serialized = pickle.dumps(pd.DataFrame({'col': [value]}))
        ma_db.insert_output(db_file, sample_id, replicate_id, serialized, seed=1000 + sample_id)
    yield db_file
    if os.path.exists(db_file):
        os.remove(db_file)


def test_migrate_to_zstd_happy_path(input_db):
    output_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    os.remove(output_db)
    try:
        stats = migrate_to_zstd(input_db, output_db, verbose=False)
        assert stats['num_replicates'] == 2
        assert stats['num_failed'] == 0
        assert stats['failed_rows'] == []

        df = ma_db.load_output(output_db, load_data=False, load_seed=True)
        assert set(df['SampleID']) == {0, 1}
        assert set(df['Seed']) == {1000, 1001}
    finally:
        if os.path.exists(output_db):
            os.remove(output_db)


def test_migrate_to_zstd_reports_and_skips_corrupt_rows(input_db):
    """Regression test: a row that fails to decompress/insert must be logged and
    reported in stats, and genuinely excluded from output_db and its size/count
    totals -- not silently counted as processed while actually missing.

    Previously the except block built a ValueError and immediately discarded it
    (raised nothing, logged nothing), and processed_count/total_original_size
    were already incremented before the row was actually inserted, so a failed
    row looked identical to a successful one in the returned stats.
    """
    # Corrupt one row's Data blob directly: a zstd magic-byte prefix followed by
    # garbage makes decompress_data() genuinely raise (unlike plain garbage bytes,
    # which decompress_data treats as already-uncompressed and passes through).
    corrupt_blob = b"\x28\xb5\x2f\xfd" + b"not a real zstd frame, just garbage bytes"
    conn = sqlite3.connect(input_db)
    conn.execute("UPDATE Output SET Data = ? WHERE SampleID = 1", (corrupt_blob,))
    conn.commit()
    conn.close()

    output_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    os.remove(output_db)
    try:
        stats = migrate_to_zstd(input_db, output_db, verbose=False)

        # The corrupt row must be visible as a failure, not silently swallowed.
        assert stats['num_failed'] == 1
        assert len(stats['failed_rows']) == 1
        failed_sample_id, failed_replicate_id, _err = stats['failed_rows'][0]
        assert (failed_sample_id, failed_replicate_id) == (1, 0)

        # Only the good row actually counts as processed / contributes to sizes.
        assert stats['num_replicates'] == 1

        # And the corrupt row must genuinely be absent from output_db, not just
        # under-counted in stats.
        df = ma_db.load_output(output_db, load_data=False)
        assert set(df['SampleID']) == {0}
    finally:
        if os.path.exists(output_db):
            os.remove(output_db)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
