import sqlite3
from typing import Union

def _update_db_value(db_file:str, table_name:str, column_name:str, new_value:Union[str, int, float], old_value:Union[str, int, float]):
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