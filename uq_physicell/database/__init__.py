"""
Database utilities for UQ PhysiCell.
"""

# Import submodules for dot notation access
from . import ma_db
from . import bo_db

# Import utility functions that don't have naming conflicts
from .ma_db import get_database_type, safe_pickle_loads
from .utils import (
    add_db_entry,
    remove_db_entry,
    remove_db_table,
    update_db_value,
)

__all__ = [
    'ma_db',
    'bo_db', 
    'get_database_type',
    'safe_pickle_loads',
    'add_db_entry',
    'remove_db_entry',
    'remove_db_table',
    'update_db_value',
]