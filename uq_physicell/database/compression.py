"""
Decompression utilities for zstd and zlib compressed data.

Provides magic byte detection, format-agnostic decompression, and database migration.
"""

import logging
import sqlite3
from pathlib import Path

# Try to import zstandard
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    import zlib

logger = logging.getLogger(__name__)


def compress_data(data: bytes, level: int = 10) -> bytes:
    """
    Compress bytes using zstd (or zlib as fallback).
    
    Args:
        data: Raw bytes to compress
        level: Compression level (1-22 for zstd, 1-9 for zlib)
    
    Returns:
        Compressed bytes
    """
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    else:
        import zlib
        return zlib.compress(data, level=9)


def decompress_data(data: bytes) -> bytes:
    """
    Automatically detect and decompress data (zstd or zlib).
    
    Detects compression format by magic bytes:
      - zstd: 0x28 0xb5 0x2f 0xfd
      - zlib: 0x78 followed by 0x01, 0x5e, 0x9c, or 0xda
    
    Args:
        data: Compressed bytes
    
    Returns:
        Decompressed bytes
    
    Raises:
        RuntimeError: If decompression fails or data is corrupted
    """
    if data is None or data == b'':
        return data
    
    # Check for zstd magic bytes (0x28, 0xb5, 0x2f, 0xfd)
    if len(data) >= 4 and data[0:4] == b'\x28\xb5\x2f\xfd':
        if HAS_ZSTD:
            try:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
            except Exception as e:
                raise RuntimeError(f"Failed to decompress zstd data: {e}")
        else:
            raise RuntimeError("zstandard library not available for decompression")
    
    # Check for zlib magic bytes (0x78 followed by 0x01, 0x5e, 0x9c, 0xda)
    if len(data) >= 2 and data[0] == 0x78 and data[1] in (0x01, 0x5e, 0x9c, 0xda):
        try:
            import zlib
            return zlib.decompress(data)
        except Exception as e:
            raise RuntimeError(f"Failed to decompress zlib data: {e}")
    
    # Data is not compressed, return as-is
    return data


# ============================================================================
# DATABASE MIGRATION
# ============================================================================

def migrate_to_zstd(input_db: str, output_db: str, verbose: bool = True) -> dict:
    """Migrate existing database to zstd compression.
    
    This function leverages existing ma_db functions to read from source database,
    create target database structure, and write data with zstd compression.
    
    Args:
        input_db: Path to source database (uncompressed or other compression)
        output_db: Path to destination database (will be created with zstd)
        verbose: Print progress information
    
    Returns:
        dict: Compression statistics with keys:
            - original_data_size_mb: Total uncompressed data size
            - compressed_data_size_mb: Total compressed data size
            - num_samples: Number of samples processed
            - num_replicates: Number of replicates processed
            - compression_ratio_pct: Compression percentage achieved
    
    Raises:
        FileNotFoundError: If input database does not exist
        RuntimeError: If migration fails at any stage
    
    Example:
        >>> stats = migrate_to_zstd('old.db', 'new.db')
        >>> print(f"Compression ratio: {stats['compression_ratio_pct']:.1f}%")
    """
    from . import ma_db  # Import here to avoid circular imports
    
    print(f"Starting migration: {input_db} → {output_db}")
    
    # Check input exists
    if not Path(input_db).exists():
        raise FileNotFoundError(f"Input database not found: {input_db}")
    
    # Initialize stats
    stats = {
        'original_data_size_mb': 0,
        'compressed_data_size_mb': 0,
        'num_samples': 0,
        'num_replicates': 0,
        'compression_ratio_pct': 0,
    }
    
    try:
        # [1/4] Load metadata, parameters, QoIs, and samples using ma_db functions
        print("[1/4] Reading metadata and parameters...")
        df_metadata = ma_db.load_metadata(input_db)
        df_params = ma_db.load_parameter_space(input_db)
        df_qois = ma_db.load_qois(input_db)
        dic_samples = ma_db.load_samples(input_db)
        
        if verbose:
            print(f"   Metadata entries: {len(df_metadata)}")
            print(f"   Parameters: {len(df_params)}")
            print(f"   QoIs: {len(df_qois)}")
            print(f"   Samples: {len(dic_samples)}")
        
        # [2/4] Create new database structure
        print("[2/4] Creating target database structure...")
        ma_db.create_structure(output_db)
        
        # [3/4] Write metadata, parameters, QoIs, and samples to target db
        print("[3/4] Writing metadata and parameters...")
        
        # Extract metadata from DataFrame
        if not df_metadata.empty:
            metadata_row = df_metadata.iloc[0]
            ma_db.insert_metadata(
                output_db,
                metadata_row.get('Sampler', ''),
                metadata_row.get('Ini_File_Path', ''),
                metadata_row.get('StructureName', ''),
                ini_hash=metadata_row.get('Ini_Hash', None),
                xml_hash=metadata_row.get('XML_Hash', None),
                rules_hash=metadata_row.get('Rules_Hash', None),
                structure_config_hash=metadata_row.get('Structure_Config_Hash', None),
                effective_run_hash=metadata_row.get('Effective_Run_Hash', None),
            )
        
        # Convert DataFrame params to dict format for insert_param_space
        if not df_params.empty:
            params_dict = {}
            for _, row in df_params.iterrows():
                params_dict[row['ParamName']] = {
                    'lower_bound': row['lower_bound'],
                    'upper_bound': row['upper_bound'],
                    'ref_value': row['ref_value'],
                    'perturbation': row['perturbation']
                }
            ma_db.insert_param_space(output_db, params_dict)
        
        # Convert QoIs DataFrame to dict
        if not df_qois.empty and df_qois.iloc[0]['QOI_Name'] is not None:
            qois_dict = dict(zip(df_qois['QOI_Name'], df_qois['QOI_Function']))
            ma_db.insert_qois(output_db, qois_dict)
        
        # Insert samples
        if dic_samples:
            ma_db.insert_samples(output_db, dic_samples)
        
        if verbose:
            print(f"   Wrote metadata, {len(df_params)} params, {len(df_qois)} QoIs, "
                       f"{len(dic_samples)} samples")
        
        # [4/4] Compress and write output data
        print("[4/4] Compressing and writing output data...")
        
        total_original_size = 0
        total_compressed_size = 0
        processed_count = 0
        
        # Load output metadata (without deserializing to keep as raw bytes)
        df_output = ma_db.load_output(input_db, load_data=False)
        
        # Process each row
        for _, row in df_output.iterrows():
            sample_id = int(row['SampleID'])
            replicate_id = int(row['ReplicateID'])
            
            # Load the actual data blob for this row
            conn = sqlite3.connect(input_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT Data FROM Output WHERE SampleID=? AND ReplicateID=?",
                (sample_id, replicate_id)
            )
            data_row = cursor.fetchone()
            conn.close()
            
            if data_row is None:
                logger.warning(f"No data found for SampleID={sample_id}, ReplicateID={replicate_id}")
                continue
            
            data_blob = bytes(data_row['Data']) if isinstance(data_row['Data'], memoryview) else data_row['Data']
            blob_size = len(data_blob)
            total_original_size += blob_size
            processed_count += 1
            
            try:
                # Decompress if needed, then insert with compression
                decompressed = decompress_data(data_blob)
                
                # Insert with compress=True to apply zstd compression
                ma_db.insert_output(output_db, sample_id, replicate_id, decompressed, compress=True)
                
                # Calculate compressed size from what was just inserted
                compressed_blob = compress_data(decompressed)
                total_compressed_size += len(compressed_blob)
                
            except Exception as e:
                ValueError(f"Failed to process SampleID={sample_id}, ReplicateID={replicate_id}: {e}")
            
            if verbose and (processed_count % 100 == 0):
                print(f"   Processed {processed_count} rows: "
                           f"{total_original_size/1e6:.1f} MB → "
                           f"{total_compressed_size/1e6:.1f} MB")
        
        from .ma_db import _disable_wal_mode
        _disable_wal_mode(output_db)  # Ensure WAL mode is disabled for final database

        stats['original_data_size_mb'] = total_original_size / 1e6
        stats['compressed_data_size_mb'] = total_compressed_size / 1e6
        stats['num_samples'] = len(dic_samples)
        stats['num_replicates'] = processed_count
        if total_original_size > 0:
            stats['compression_ratio_pct'] = 100 * (1 - total_compressed_size / total_original_size)
        
        # Log statistics
        print(f"\n{'='*70}")
        print("COMPRESSION STATISTICS")
        print(f"{'='*70}")
        print(f"Original Data BLOB size:      {stats['original_data_size_mb']:>10.2f} MB")
        print(f"Compressed Data BLOB size:    {stats['compressed_data_size_mb']:>10.2f} MB")
        print(f"Compression ratio:            {stats['compression_ratio_pct']:>10.1f}%")
        print(f"Number of samples:            {stats['num_samples']:>10}")
        print(f"Number of replicates:         {stats['num_replicates']:>10}")
        print(f"{'='*70}")
        
        codec = "zstd" if HAS_ZSTD else "zlib"
        print(f"✓ Compression codec: {codec}")
        print("✓ Migration complete!")
        
        return stats
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
