import sqlite3
import pandas as pd
import io
import pickle
import torch
from botorch.models.model_list_gp_regression import ModelListGP

def create_structure(db_path:str):
    """
    Create the Bayesian Optimization (BO) database structure with six tables:
    1. Metadata: Stores information about the calibration (method, observed data path, .ini config path, and model structure name).
    2. ParameterSpace: Stores the parameter space information (ParamName, Type, Lower_Bound, Upper_Bound, Regulates).
    3. QoIs: Stores the quantities of interest (QoI_Name, QoI_Function, ObsData_Column, QoI_distanceFunction, QoI_distanceWeight).
    4. GP_Models: Stores the Gaussian Process models (IterationID, GP_Model, Hypervolume).
    5. Samples: Stores the samples (IterationID, SampleID, ParamName, ParamValue).
    6. Output: Stores the output of the simulations (SampleID, Objective_Function, Noise_Std, and Data).
    """
    try:
        # Connect to the database (create it if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Metadata (
                BO_Method TEXT PRIMARY KEY,
                ObsData_Path TEXT,
                Ini_File_Path TEXT,
                StructureName TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ParameterSpace (
                ParamName TEXT PRIMARY KEY,
                Type TEXT,
                Lower_Bound DOUBLE,
                Upper_Bound DOUBLE,
                Regulates TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS QoIs (
                QOI_Name TEXT PRIMARY KEY,
                QOI_Function TEXT,
                ObsData_Column TEXT,
                QoI_distanceFunction TEXT DEFAULT '',
                QoI_distanceWeight DOUBLE DEFAULT 0.0
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS GP_Models (
                IterationID INTEGER,
                GP_Model BLOB,
                Hypervolume DOUBLE,
                PRIMARY KEY (IterationID)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Samples (
                IterationID INTEGER,
                SampleID INTEGER,
                ParamName TEXT,
                ParamValue DOUBLE,
                PRIMARY KEY (IterationID, SampleID, ParamName)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Output (
                SampleID INTEGER,
                ObjFunc BLOB,
                Noise_Std BLOB,
                Data BLOB,
                PRIMARY KEY (SampleID)
            )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error generating tables: {e}")
    
def insert_metadata(db_path:str, metadata:dict):
    """
    Insert BO metadata information into the Metadata table.
    Parameters:
    - db_path: Path to the database file.
    - metadata: Dictionary containing BO metadata information.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO Metadata (BO_Method, ObsData_Path, Ini_File_Path, StructureName)
            VALUES (?, ?, ?, ?)
        """, (metadata['BO_Method'], 
              metadata['ObsData_Path'], 
              metadata['Ini_File_Path'], 
              metadata['StructureName']))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO Metadata: {e}")
    
def insert_param_space(db_path:str, param_space:dict):
    """
    Insert BO parameter space information into the ParameterSpace table.
    Parameters:
    - db_path: Path to the database file.
    - param_space: Dictionary containing parameter space information.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for param_name, details in param_space.items():
            cursor.execute("""
                INSERT INTO ParameterSpace (ParamName, Type, Lower_Bound, Upper_Bound, Regulates)
                VALUES (?, ?, ?, ?, ?)
            """, (param_name, 
                  details['type'], 
                  details['lower_bound'], details['upper_bound'], 
                  details.get('regulates', None)))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO Parameter Space: {e}")
    
def insert_qois(db_path:str, qois:dict):
    """
    Insert QoIs into the QoIs table.
    Parameters:
    - db_path: Path to the database file.
    - qois: Dictionary of QoIs (keys as names, values as lambda functions or strings).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for id_qoi, qoi_name in enumerate(qois["QOI_Name"]):
            cursor.execute("""
                INSERT INTO QoIs (QOI_Name, QOI_Function, ObsData_Column, QoI_distanceFunction, QoI_distanceWeight)
                VALUES (?, ?, ?, ?, ?)
            """, (qoi_name, qois['QOI_Function'][id_qoi], 
                  qois['ObsData_Column'][id_qoi], 
                  qois['QoI_distanceFunction'][id_qoi],
                  qois['QoI_distanceWeight'][id_qoi]))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO QoIs: {e}") 
    
def insert_gp_models(db_path:str, iteration_id:int, gp_model:ModelListGP, hypervolume:float):
    """
    Insert Gaussian Process model into the GP_Models table.
    Parameters:
    - db_path: Path to the database file.
    - iteration_id: The iteration ID for the GP model.
    - gp_model: The Gaussian Process model to be stored.
    - hypervolume: The hypervolume value for this iteration.
    """
    try:
        # Serialize the GP model into a binary object
        buffer = io.BytesIO()
        torch.save(gp_model.state_dict(), buffer, _use_new_zipfile_serialization=False)
        gp_model_binary = buffer.getvalue()
        # Connect to the database and insert the GP model with timeout
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO GP_Models (IterationID, GP_Model, Hypervolume)
            VALUES (?, ?, ?)
        """, (iteration_id, gp_model_binary, hypervolume))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO GP Model: {e}")
    
def insert_samples(db_path:str, iteration_id:int, samples:dict):
    """
    Insert samples into the Samples table.
    Parameters:
    - db_path: Path to the database file.
    - iteration_id: The iteration ID for the samples.
    - samples: Dictionary of samples (keys as SampleID, values as dictionaries of ParamName and ParamValue).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        for sample_id, params in samples.items():
            for param_name, param_value in params.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO Samples (IterationID, SampleID, ParamName, ParamValue)
                    VALUES (?, ?, ?, ?)
                """, (iteration_id, int(sample_id), param_name, param_value))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO Samples: {e}")

def insert_output(db_path:str, sample_id:int, obj_func:bytes, noise_std:bytes, data:bytes):
    """
    Insert simulation results into the Output table.
    Parameters:
    - db_path: Path to the database file.
    - sample_id: The sample ID.
    - obj_func: The objective function values (as binary).
    - noise_std: The noise standard deviation values (as binary).
    - data: The simulation results data (as binary).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Output (SampleID, ObjFunc, Noise_Std, Data)
            VALUES (?, ?, ?, ?)
        """, (int(sample_id), sqlite3.Binary(obj_func), sqlite3.Binary(noise_std), sqlite3.Binary(data)))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO Output: {e}")
    
def load_metadata(db_file: str) -> pd.DataFrame:
    """Load metadata from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with metadata information.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_metadata = load_metadata('calibration.db')
        >>> print(df_metadata['BO_Method'].values[0])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        # Load with flexible column handling for backward compatibility
        cursor.execute("PRAGMA table_info(Metadata)")
        metadata_columns_info = cursor.fetchall()
        metadata_column_names = [col[1] for col in metadata_columns_info]
        
        cursor.execute('SELECT * FROM Metadata')
        metadata = cursor.fetchall()
        
        if metadata:
            df_metadata = pd.DataFrame(metadata, columns=metadata_column_names)
        else:
            # Create empty DataFrame with expected schema
            df_metadata = pd.DataFrame(columns=['BO_Method', 'ObsData_Path', 'Ini_File_Path', 'StructureName'])
        return df_metadata
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO metadata: {e}")
    finally:
        conn.close()

def load_parameter_space(db_file: str) -> pd.DataFrame:
    """Load parameter space from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['ParamName', 'type', 'lower_bound', 'upper_bound', 'regulates'].
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_params = load_parameter_space('calibration.db')
        >>> print(df_params[['ParamName', 'lower_bound', 'upper_bound']])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM ParameterSpace')
        param_space = cursor.fetchall()
        df_param_space = pd.DataFrame(param_space, columns=['ParamName', 'type', 'lower_bound', 'upper_bound', 'regulates'])
        return df_param_space
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO parameter space: {e}")
    finally:
        conn.close()

def load_qois(db_file: str) -> pd.DataFrame:
    """Load quantities of interest (QoIs) from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['QoI_Name', 'QoI_Type', 'ObsData_Column', 
                     'QoI_distanceFunction', 'QoI_distanceWeight'].
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_qois = load_qois('calibration.db')
        >>> print(df_qois['QoI_Name'].to_list())
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM QoIs')
        qois = cursor.fetchall()
        df_qois = pd.DataFrame(qois, columns=['QoI_Name', 'QoI_Type', 'ObsData_Column', 
                                             'QoI_distanceFunction', 'QoI_distanceWeight'])
        return df_qois
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO QoIs: {e}")
    finally:
        conn.close()

def load_gp_models(db_file: str) -> pd.DataFrame:
    """Load Gaussian Process models from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['IterationID', 'GP_Model', 'Hypervolume'].
                     GP_Model column contains deserialized torch model objects.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_gp_models = load_gp_models('calibration.db')
        >>> print(f"Loaded {len(df_gp_models)} GP models")
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM GP_Models')
        gp_models = cursor.fetchall()
        df_gp_models = pd.DataFrame(gp_models, columns=['IterationID', 'GP_Model', 'Hypervolume'])
        # Deserialize the GP_Model column
        df_gp_models['GP_Model'] = df_gp_models['GP_Model'].apply(
            lambda x: torch.load(io.BytesIO(x), map_location=torch.device('cpu'))
        )
        return df_gp_models
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO GP models: {e}")
    finally:
        conn.close()

def load_samples(db_file: str, iteration_ids: list = None) -> pd.DataFrame:
    """Load parameter samples from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
        iteration_ids (list, optional): List of specific iteration IDs to load.
                                       If None, loads all iterations.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['IterationID', 'SampleID', 'ParamName', 'ParamValue'].
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_samples = load_samples('calibration.db')
        >>> # Load specific iterations
        >>> df_samples = load_samples('calibration.db', iteration_ids=[0, 1, 2])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        if iteration_ids is None:
            cursor.execute('SELECT * FROM Samples')
        else:
            placeholders = ','.join('?' * len(iteration_ids))
            cursor.execute(f'SELECT * FROM Samples WHERE IterationID IN ({placeholders})', iteration_ids)
        
        samples = cursor.fetchall()
        df_samples = pd.DataFrame(samples, columns=['IterationID', 'SampleID', 'ParamName', 'ParamValue'])
        return df_samples
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO samples: {e}")
    finally:
        conn.close()

def load_output(db_file: str, sample_ids: list = None, load_data: bool = True) -> pd.DataFrame:
    """Load simulation output from the BO database.
    
    Args:
        db_file (str): Path to the SQLite database file.
        sample_ids (list, optional): List of specific sample IDs to load.
                                    If None, loads all samples.
        load_data (bool, optional): If True, deserializes the ObjFunc, Noise_Std, and Data columns.
                                   If False, only loads SampleID metadata.
                                   Default is True.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['SampleID', 'ObjFunc', 'Noise_Std', 'Data'] if load_data=True,
                     or ['SampleID'] if load_data=False.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> # Load all output with deserialization
        >>> df_output = load_output('calibration.db')
        >>> 
        >>> # Load specific samples without deserialization
        >>> df_output = load_output('calibration.db', sample_ids=[0, 1, 2], load_data=False)
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        if sample_ids is None:
            cursor.execute('SELECT * FROM Output')
        else:
            placeholders = ','.join('?' * len(sample_ids))
            cursor.execute(f'SELECT * FROM Output WHERE SampleID IN ({placeholders})', sample_ids)
        
        output = cursor.fetchall()
        
        if load_data:
            df_output = pd.DataFrame(output, columns=['SampleID', 'ObjFunc', 'Noise_Std', 'Data'])
            # Deserialize the columns
            df_output['ObjFunc'] = df_output['ObjFunc'].apply(pickle.loads)
            df_output['Noise_Std'] = df_output['Noise_Std'].apply(pickle.loads)
            df_output['Data'] = df_output['Data'].apply(pickle.loads)
        else:
            df_output = pd.DataFrame(output, columns=['SampleID', 'ObjFunc', 'Noise_Std', 'Data'])
            df_output = df_output[['SampleID']]
        
        return df_output
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO output: {e}")
    finally:
        conn.close()

def load_structure(db_file: str, load_data: bool = True) -> tuple:
    """Load the complete BO database structure using modular load functions.
    
    This is a convenience wrapper that loads all tables from the database.
    For more control over what data is loaded, use the individual load functions:
    - load_metadata(db_file)
    - load_parameter_space(db_file)
    - load_qois(db_file)
    - load_gp_models(db_file)
    - load_samples(db_file, iteration_ids=None)
    - load_output(db_file, sample_ids=None, load_data=True)
    
    Args:
        db_file (str): Path to the SQLite database file.
        load_data (bool, optional): If True, deserializes GP models and output data.
                                   If False, only loads metadata without deserialization.
                                   Default is True.
    
    Returns:
        tuple: A 6-tuple containing:
            - df_metadata (pd.DataFrame): Metadata information
            - df_param_space (pd.DataFrame): Parameter space definitions
            - df_qois (pd.DataFrame): Quantities of interest definitions
            - df_gp_models (pd.DataFrame): Gaussian Process models
            - df_samples (pd.DataFrame): Parameter samples
            - df_output (pd.DataFrame): Simulation output
    
    Raises:
        RuntimeError: If any database loading fails.
    
    Example:
        >>> # Load everything with full data
        >>> metadata, params, qois, gp_models, samples, output = load_structure('calibration.db')
        >>> 
        >>> # Load only metadata (no deserialization)
        >>> metadata, params, qois, gp_models, samples, output = load_structure('calibration.db', load_data=False)
    """
    df_metadata = load_metadata(db_file)
    df_param_space = load_parameter_space(db_file)
    df_qois = load_qois(db_file)
    
    if load_data:
        df_gp_models = load_gp_models(db_file)
        df_samples = load_samples(db_file)
        df_output = load_output(db_file, load_data=True)
    else:
        df_gp_models = pd.DataFrame(columns=['IterationID', 'GP_Model', 'Hypervolume'])
        df_samples = load_samples(db_file)
        df_output = load_output(db_file, load_data=False)
    
    return df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output