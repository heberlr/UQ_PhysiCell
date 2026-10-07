import os
import sqlite3
from typing import Optional, Union

def update_db_value(db_file:str, table_name:str, column_name:str, new_value:Union[str, int, float], old_value:Union[str, int, float]):
    """
    Update a specific value in the database.
    Parameters: 
    - db_file: Path to the database file.
    - table_name: Name of the table to update.
    - column_name: Name of the column to update.
    - new_value: The new value to set.
    - condition_column: The column to use for the condition.
    - condition_value: The value to match in the condition column.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''UPDATE {table_name} SET {column_name} = ? WHERE {column_name} = ?''', (new_value, old_value))
    conn.commit()
    conn.close()

def add_db_entry(db_file:str, table_name:str, column_name:str, value:Union[str, int, float]):
    """
    Add a new entry to the database.
    Parameters:
    - db_file: Path to the database file.
    - table_name: Name of the table to insert into.
    - column_name: Name of the column to insert into.
    - value: The value to insert.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # If table does not exits, add it
    if not cursor.fetchall():
        _create_table(db_file, table_name)
    # Get existing columns
    cursor.execute(f'''PRAGMA table_info({table_name})''')
    columns = [column[1] for column in cursor.fetchall()]
    if column_name in columns:
        cursor.execute(f'''INSERT INTO {table_name} ({column_name}) VALUES (?)''', (value,))
    else:
        # If column does not exist, we can add it
        _alter_table_add_column(db_file, table_name, column_name, value.__class__.__name__.upper())
        cursor.execute(f'''INSERT INTO {table_name} ({column_name}) VALUES (?)''', (value,))
    conn.commit()
    conn.close()

def remove_db_entry(db_file:str, table_name:str, column_name:str, value:Union[str, int, float]):
    """
    Remove an entry from the database.
    Parameters:
    - db_file: Path to the database file.
    - table_name: Name of the table to remove from.
    - column_name: Name of the column to match.
    - value: The value to match for removal.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''DELETE FROM {table_name} WHERE {column_name} = ?''', (value,))
    conn.commit()
    conn.close()

def remove_db_table(db_file:str, table_name:str):
    """
    Remove a table from the database.
    Parameters:
    - db_file: Path to the database file.
    - table_name: Name of the table to remove.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f'''DROP TABLE IF EXISTS {table_name}''')
    conn.commit()
    conn.close()

def _create_table(db_file:str, table_name:str):
    """
    Create a new table in the database.
    Parameters:
    - db_file: Path to the database file.
    - table_name: Name of the table to create.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

def _alter_table_add_column(db_file:str, table_name:str, new_column_name:str, column_type:str):
    """
    Add a new column to an existing table in the database.
    Parameters:
    - db_file: Path to the database file.
    - table_name: Name of the table to alter.
    - new_column_name: Name of the new column to add.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {new_column_name} {column_type} DEFAULT ''")
    conn.commit()
    conn.close()

def download_file(file_name, base_url="https://zenodo.org/records/21496966/files/", custom_url=None):
    """
    Download file from url_base by default or from a custom url if provided. If the file already exists, it will not be downloaded again.
    Parameters:
    - file_name: Name of the file to download.
    - base_url: Base URL to download the file from. Defaults to "https://zenodo.org/records/21496966/files/".
    - custom_url: Custom URL to download the file from. If provided, this will override the base_url.
    """
    import urllib.request, os
    if not os.path.exists(file_name):
        if custom_url is not None:
            url_path = custom_url
        else:
            url_path = base_url + file_name
        print(f"Downloading {file_name} from {url_path} ...")
        urllib.request.urlretrieve(url_path, file_name)


def get_database_type(db_file: str) -> Optional[str]:
    """Determine which kind of UQ-PhysiCell study wrote a database file.

    The kind is read from the ``Metadata`` table: model-analysis databases have a
    ``Sampler`` column, BO databases a ``BO_Method`` column, and ABC databases a
    ``Method`` column set to ``'ABC'``. An ABC database written before its
    ``Metadata`` row existed is still recognized by pyABC's ``abc_smc`` table.
    Nothing is deserialized, so this is safe to call on untrusted files.

    Args:
        db_file (str): Path to the SQLite database file to examine.

    Returns:
        str or None: ``'MA'`` for Model Analysis, ``'BO'`` for Bayesian Optimization,
        ``'ABC'`` for ABC-SMC calibration (read it with ``pyabc.History``), or None if
        the file does not exist, is not a SQLite database, or is not a recognized
        UQ-PhysiCell database.

    Example:
        >>> db_type = get_database_type('analysis.db')
        >>> if db_type == 'MA':
        ...     print("This is a Model Analysis database")
        >>> elif db_type == 'ABC':
        ...     print("This is an ABC database - read it with pyabc.History")
    """
    if not os.path.isfile(db_file):
        return None

    conn = sqlite3.connect(db_file)
    try:
        cursor = conn.cursor()
        tables = {row[0] for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if 'Metadata' in tables:
            columns = [col[1] for col in cursor.execute("PRAGMA table_info(Metadata)").fetchall()]
            # Model analysis database should have one column as 'Sampler'
            if 'Sampler' in columns:
                return 'MA'
            # BO database should have one column as 'BO_Method'
            if 'BO_Method' in columns:
                return 'BO'
            # ABC database has a 'Method' column holding 'ABC'
            if 'Method' in columns:
                row = cursor.execute("SELECT Method FROM Metadata LIMIT 1").fetchone()
                if row and row[0] == 'ABC':
                    return 'ABC'
        # pyABC's own run table, present even without UQ-PhysiCell's Metadata row
        if 'abc_smc' in tables:
            return 'ABC'
        return None
    except sqlite3.DatabaseError:
        # Not a SQLite database (or a corrupted one)
        return None
    finally:
        conn.close()
