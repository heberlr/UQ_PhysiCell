import sqlite3
from typing import Union

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