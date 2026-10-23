import sqlite3
import os
import pandas as pd
import io
import json
import pickle
import warnings
from typing import Any, TYPE_CHECKING

from uq_physicell import __version__ as uq_physicell_version
from pcdl import __version__ as pcdl_version

try:
    from botorch import __version__ as botorch_version
except ImportError:
    botorch_version = "Not installed"
    warnings.warn("Botorch is not available. Some features may be limited.")

try:
    import torch
except ImportError:
    torch = None
    warnings.warn("PyTorch is not available. GP model serialization/deserialization will be disabled.")

if TYPE_CHECKING:
    from botorch.models.model_list_gp_regression import ModelListGP
else:
    ModelListGP = Any

def create_structure(db_path:str):
    """
    Create the SQLite database structure for storing Bayesian Optimization (BO) calibration data. This function initializes the database with the necessary tables to store metadata, parameter space definitions, quantities of interest (QoIs), Gaussian Process models, samples, and simulation output.
    
    Args:
        db_path (str): Path to the SQLite database file.
    
    Tables Created:
        - Metadata: Stores information about the calibration (method, observed data path, .ini config path, model structure name, uq_physicell_version, pcdl_version, botorch_version, BO_Options: a JSON string with the resolved BO configuration, and the PhysiCell effective-config fingerprint hashes Ini_Hash / XML_Hash / Rules_Hash / Structure_Config_Hash / Effective_Run_Hash).
        - ParameterSpace: Stores the parameter space information (ParamName, Type, Lower_Bound, Upper_Bound, Regulates).
        - QoIs: Stores the quantities of interest (QoI_Name, QoI_Function, ObsData_Column, QoI_distanceFunction, QoI_distanceWeight).
        - GP_Models: Stores the Gaussian Process models (IterationID, GP_Model, Score, ConvergenceStatus).
                     Score is the per-iteration frontier metric: hypervolume for multi-objective runs,
                     best fitness value for single-objective runs. Databases created before this rename
                     used the column name 'Hypervolume' and are migrated in place by create_structure().
                     ConvergenceStatus is a JSON string with the per-iteration convergence analysis (nullable).
        - Samples: Stores the samples (IterationID, SampleID, ParamName, ParamValue).
        - Output: Stores the output of the simulations (SampleID, ObjFunc, Noise_Std, Data, Seeds).
                  Seeds is a JSON list of the PhysiCell random seed used for each replicate of the
                  sample, ordered by replicate id (nullable; entries are null where a seed was not recorded).
    
    Example:
        >>> create_structure('calibration.db')
        # Database created with the necessary tables for BO calibration.
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
                StructureName TEXT,
                uq_physicell_version TEXT,
                pcdl_version TEXT,
                botorch_version TEXT,
                BO_Options TEXT,
                Ini_Hash TEXT,
                XML_Hash TEXT,
                Rules_Hash TEXT,
                Structure_Config_Hash TEXT,
                Effective_Run_Hash TEXT
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
                Score DOUBLE,
                ConvergenceStatus TEXT,
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
                Seeds TEXT,
                PRIMARY KEY (SampleID)
            )
        """)
        # Bring an existing database up to the current schema (additive, idempotent)
        _apply_schema_migrations(cursor)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error generating tables: {e}")


def _apply_schema_migrations(cursor) -> list:
    """Apply additive, idempotent schema migrations to an existing BO database.

    Assumes the base tables already exist. Safe to call repeatedly and on
    already-current databases. Returns the list of migrations that were applied.

    Migrations:
        - Metadata.BO_Options              : column added if missing
        - Metadata config-fingerprint hashes (Ini_Hash, XML_Hash, Rules_Hash,
          Structure_Config_Hash, Effective_Run_Hash) : columns added if missing
        - GP_Models.Hypervolume -> Score   : column renamed if still legacy-named
        - GP_Models.ConvergenceStatus      : column added if missing
        - Output.Seeds                     : column added if missing
    """
    applied = []

    meta_cols = [row[1] for row in cursor.execute("PRAGMA table_info(Metadata)").fetchall()]
    if meta_cols and "BO_Options" not in meta_cols:
        cursor.execute("ALTER TABLE Metadata ADD COLUMN BO_Options TEXT")
        applied.append("Metadata.BO_Options added")
    if meta_cols:
        for col in ("Ini_Hash", "XML_Hash", "Rules_Hash", "Structure_Config_Hash", "Effective_Run_Hash"):
            if col not in meta_cols:
                cursor.execute(f"ALTER TABLE Metadata ADD COLUMN {col} TEXT")
                applied.append(f"Metadata.{col} added")

    gp_cols = [row[1] for row in cursor.execute("PRAGMA table_info(GP_Models)").fetchall()]
    if gp_cols and "Score" not in gp_cols and "Hypervolume" in gp_cols:
        cursor.execute("ALTER TABLE GP_Models RENAME COLUMN Hypervolume TO Score")
        applied.append("GP_Models.Hypervolume renamed to Score")
        gp_cols = [row[1] for row in cursor.execute("PRAGMA table_info(GP_Models)").fetchall()]
    if gp_cols and "ConvergenceStatus" not in gp_cols:
        cursor.execute("ALTER TABLE GP_Models ADD COLUMN ConvergenceStatus TEXT")
        applied.append("GP_Models.ConvergenceStatus added")

    out_cols = [row[1] for row in cursor.execute("PRAGMA table_info(Output)").fetchall()]
    if out_cols and "Seeds" not in out_cols:
        cursor.execute("ALTER TABLE Output ADD COLUMN Seeds TEXT")
        applied.append("Output.Seeds added")

    return applied


def migrate_bo_database(db_path: str) -> list:
    """Upgrade an existing BO calibration database file to the current schema, in place.

    Runs the additive, idempotent migrations in :func:`_apply_schema_migrations` (see there
    for the list). Safe to call on an already-current database (returns an empty list).

    Args:
        db_path (str): Path to an existing BO calibration ``.db`` file.

    Returns:
        list: Human-readable names of the migrations that were applied (empty if already current).

    Raises:
        RuntimeError: If the file does not exist, is not a BO calibration database,
            or the migration fails (e.g. the file is read-only).
    """
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database file not found: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if not {"Metadata", "GP_Models"}.issubset(tables):
            raise RuntimeError(f"'{db_path}' is not a BO calibration database.")
        applied = _apply_schema_migrations(cursor)
        conn.commit()
        return applied
    except sqlite3.Error as e:
        raise RuntimeError(f"Error migrating BO database '{db_path}': {e}")
    finally:
        conn.close()


def insert_metadata(db_path:str, metadata:dict):
    """
    Insert BO metadata information into the Metadata table.

    Args:
        db_path (str): Path to the database file.
        metadata (dict): Dictionary containing BO metadata information. The optional
            'BO_Options' key holds a JSON string with the resolved BO configuration.

    Example:
        >>> metadata = {
        ...     'BO_Method': 'Bayesian Optimization',
        ...     'ObsData_Path': 'observed_data.csv',
        ...     'Ini_File_Path': 'config.ini',
        ...     'StructureName': 'PhysiCell',
        ...     'BO_Options': '{"num_iterations": 10, "use_exponential_fitness": true}'
        ... }
        >>> insert_metadata('calibration.db', metadata)
        # Metadata inserted into the database.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO Metadata (BO_Method, ObsData_Path, Ini_File_Path, StructureName, uq_physicell_version, pcdl_version, botorch_version, BO_Options,
                                             Ini_Hash, XML_Hash, Rules_Hash, Structure_Config_Hash, Effective_Run_Hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (metadata['BO_Method'],
              metadata['ObsData_Path'],
              metadata['Ini_File_Path'],
              metadata['StructureName'],
              uq_physicell_version,
              pcdl_version,
              botorch_version,
              metadata.get('BO_Options'),
              metadata.get('Ini_Hash'),
              metadata.get('XML_Hash'),
              metadata.get('Rules_Hash'),
              metadata.get('Structure_Config_Hash'),
              metadata.get('Effective_Run_Hash')))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO Metadata: {e}")
    
def insert_param_space(db_path:str, param_space:dict):
    """
    Insert BO parameter space information into the ParameterSpace table.
    
    Args:
        db_path (str): Path to the database file.
        param_space (dict): Dictionary containing parameter space information.

    Example:
        >>> param_space = {
        ...     'param1': {'type': 'real', 'lower_bound': 0.0, 'upper_bound': 1.0},
        ...     'param2': {'type': 'real', 'lower_bound': 1.0, 'upper_bound': 5.0}
        ... }
        >>> insert_param_space('calibration.db', param_space)
        # Parameter space information inserted into the database.
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

    Args:
        db_path (str): Path to the database file.
        qois (dict): Dictionary of QoIs (keys as names, values as lambda functions or strings).
    
    Example:
        >>> qois = {
        ...     'total_cells': "lambda data: data['cell_count'].sum()",
        ...     'max_radius': "lambda data: data['radius'].max()"
        ... }
        >>> insert_qois('calibration.db', qois)
        # QoIs inserted into the database.
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
    
def insert_gp_models(db_path:str, iteration_id:int, gp_model:ModelListGP, score:float, convergence_status:str=None):
    """
    Insert Gaussian Process model into the GP_Models table.

    Args:
        db_path (str): Path to the database file.
        iteration_id (int): The iteration ID for the GP model.
        gp_model (ModelListGP): The Gaussian Process model to be stored.
        score (float): The per-iteration frontier metric: hypervolume for multi-objective runs,
            best fitness value for single-objective runs.
        convergence_status (str, optional): JSON string with the convergence analysis for this iteration. Defaults to None.

    Example:
        >>> from botorch.models import SingleTaskGP
        >>> from botorch.models.model_list_gp_regression import ModelListGP
        >>> import torch
        >>> # Create a simple GP model for demonstration
        >>> X = torch.rand(10, 1)
        >>> Y = torch.sin(X * 2 * torch.pi) + 0.1 * torch.randn_like(X)
        >>> gp = SingleTaskGP(X, Y)
        >>> model_list = ModelListGP(gp)
        >>> insert_gp_models('calibration.db', iteration_id=0, gp_model=model_list, score=0.5)
    """
    try:
        if torch is None:
            raise RuntimeError("PyTorch is not available. Cannot serialize GP models.")
        # Serialize the GP model into a binary object
        buffer = io.BytesIO()
        torch.save(gp_model.state_dict(), buffer, _use_new_zipfile_serialization=False)
        gp_model_binary = buffer.getvalue()
        # Connect to the database and insert the GP model with timeout
        conn = sqlite3.connect(db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO GP_Models (IterationID, GP_Model, Score, ConvergenceStatus)
            VALUES (?, ?, ?, ?)
        """, (iteration_id, gp_model_binary, score, convergence_status))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting BO GP Model: {e}")
    
def insert_samples(db_path:str, iteration_id:int, samples:dict):
    """
    Insert samples into the Samples table.
    
    Args:
        db_path (str): Path to the database file.
        iteration_id (int): The iteration ID for the samples.
        samples (dict): Dictionary of samples (keys as SampleID, values as dictionaries of ParamName and ParamValue).
    
    Example:
        >>> samples = {
        ...     1: {'param1': 0.5, 'param2': 1.0},
        ...     2: {'param1': 0.3, 'param2': 1.2}
        ... }
        >>> insert_samples('calibration.db', iteration_id=0, samples=samples)
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

def insert_output(db_path:str, sample_id:int, obj_func:bytes, noise_std:bytes, data:bytes, seeds:str=None):
    """
    Insert simulation results into the Output table.

    Args:
        db_path (str): Path to the database file.
        sample_id (int): The sample ID.
        obj_func (bytes): The objective function values (as binary).
        noise_std (bytes): The noise standard deviation values (as binary).
        data (bytes): The simulation results data (as binary).
        seeds (str, optional): JSON list of the PhysiCell random seed for each replicate,
            ordered by replicate id. Stored as NULL when None. Defaults to None.

    Example:
        >>> insert_output('calibration.db', sample_id=1, obj_func=b'...', noise_std=b'...', data=b'...', seeds='[123, 456]')
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Output (SampleID, ObjFunc, Noise_Std, Data, Seeds)
            VALUES (?, ?, ?, ?, ?)
        """, (int(sample_id), sqlite3.Binary(obj_func), sqlite3.Binary(noise_std), sqlite3.Binary(data), seeds))
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
        cursor.execute('SELECT * FROM Metadata')
        metadata = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        df_metadata = pd.DataFrame(metadata, columns=column_names)
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
        # pandas >= 3 stores missing strings as NaN; keep None for "regulates nothing"
        df_param_space['regulates'] = df_param_space['regulates'].astype(object).where(df_param_space['regulates'].notna(), None)
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
        pd.DataFrame: DataFrame with columns ['IterationID', 'GP_Model', 'Score', 'ConvergenceStatus']
                     ('ConvergenceStatus' absent for databases created before that column existed).
                     A legacy 'Hypervolume' column is returned as-is; call migrate_bo_database() (or
                     load_structure(), which migrates by default) to rename it to 'Score' on disk.
                     GP_Model contains deserialized torch objects; ConvergenceStatus is parsed from JSON to a dict.

    Raises:
        sqlite3.Error: If database connection or query fails.

    Example:
        >>> df_gp_models = load_gp_models('calibration.db')
        >>> print(f"Loaded {len(df_gp_models)} GP models")
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        if torch is None:
            raise RuntimeError("PyTorch is not available. Cannot deserialize GP models.")
        cursor.execute('SELECT * FROM GP_Models')
        gp_models = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        df_gp_models = pd.DataFrame(gp_models, columns=columns)
        # Deserialize the GP_Model column
        df_gp_models['GP_Model'] = df_gp_models['GP_Model'].apply(
            lambda x: torch.load(io.BytesIO(x), map_location=torch.device('cpu')) if x is not None else None
        )
        if 'ConvergenceStatus' in df_gp_models.columns:
            df_gp_models['ConvergenceStatus'] = df_gp_models['ConvergenceStatus'].apply(
                lambda s: json.loads(s) if isinstance(s, str) and s else None
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
        pd.DataFrame: DataFrame with columns ['SampleID', 'ObjFunc', 'Noise_Std', 'Data', 'Seeds']
                     if load_data=True ('Seeds' absent for databases created before that column existed;
                     parsed from JSON to a list of per-replicate seeds), or ['SampleID'] if load_data=False.

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
        columns = [description[0] for description in cursor.description]

        if load_data:
            df_output = pd.DataFrame(output, columns=columns)
            # Deserialize the columns
            df_output['ObjFunc'] = df_output['ObjFunc'].apply(pickle.loads)
            df_output['Noise_Std'] = df_output['Noise_Std'].apply(pickle.loads)
            df_output['Data'] = df_output['Data'].apply(pickle.loads)
            if 'Seeds' in df_output.columns:
                df_output['Seeds'] = df_output['Seeds'].apply(lambda s: json.loads(s) if s else None)
        else:
            df_output = pd.DataFrame(output, columns=columns)
            df_output = df_output[['SampleID']]
        
        return df_output
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading BO output: {e}")
    finally:
        conn.close()

def load_structure(db_file: str, load_data: bool = True, migrate: bool = True) -> tuple:
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
        migrate (bool, optional): If True (default), upgrade an older database file to the
                                  current schema in place before loading (see migrate_bo_database()).
                                  Set False to load a read-only / legacy database without touching it.

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
    if migrate:
        applied = migrate_bo_database(db_file)
        if applied:
            warnings.warn(f"Upgraded '{db_file}' to the current BO schema: {', '.join(applied)}")

    df_metadata = load_metadata(db_file)
    df_param_space = load_parameter_space(db_file)
    df_qois = load_qois(db_file)

    if load_data:
        df_gp_models = load_gp_models(db_file)
        df_samples = load_samples(db_file)
        df_output = load_output(db_file, load_data=True)
    else:
        df_gp_models = pd.DataFrame(columns=['IterationID', 'GP_Model', 'Score', 'ConvergenceStatus'])
        df_samples = load_samples(db_file)
        df_output = load_output(db_file, load_data=False)
    
    return df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output