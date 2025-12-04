import sqlite3
import pandas as pd
import io
import pickle
import torch
from botorch.models.model_list_gp_regression import ModelListGP

def create_structure(db_path:str):
    """
    Create the Bayesian Optimization (BO) database structure with six tables:
    1. Metadata: Stores information about the calibration (method, observed data path, enhancement strategy parameters, .ini config path, and model structure name).
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
                Enhancement_Strategy TEXT DEFAULT 'none',
                Diversity_Weight DOUBLE DEFAULT 0.0,
                Uncertainty_Weight DOUBLE DEFAULT 0.0,
                Constraint_Strength DOUBLE DEFAULT 0.0,
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
            INSERT OR REPLACE INTO Metadata (BO_Method, ObsData_Path, Enhancement_Strategy, Diversity_Weight, Uncertainty_Weight, Constraint_Strength, Ini_File_Path, StructureName)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (metadata['BO_Method'], 
              metadata['ObsData_Path'], 
              metadata.get("Enhancement_Strategy", "none"), 
              metadata.get("Diversity_Weight", 0.0), 
              metadata.get("Uncertainty_Weight", 0.0),
              metadata.get("Constraint_Strength", 0.0),
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
    
def load_structure(db_file:str) -> tuple:
    """
    Load the structure of the BO database.
    Parameters:
    - db_file: Path to the database file.
    Return:
    - df_metadata: DataFrame containing metadata.
    - param_space: DataFrame containing parameter space information.
    - qois: DataFrame containing QoIs.
    - gp_models: DataFrame containing GP models.
    - samples: DataFrame containing samples.
    - output: DataFrame containing output data.
    """
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Load Metadata with flexible column handling
    cursor.execute("PRAGMA table_info(Metadata)")
    metadata_columns_info = cursor.fetchall()
    metadata_column_names = [col[1] for col in metadata_columns_info]
    
    cursor.execute('SELECT * FROM Metadata')
    metadata = cursor.fetchall()
    
    # Create DataFrame with available columns
    if metadata:
        df_metadata = pd.DataFrame(metadata, columns=metadata_column_names)
    else:
        # Create empty DataFrame with new schema
        df_metadata = pd.DataFrame(columns=['BO_Method', 'ObsData_Path', 'Enhancement_Strategy', 'Diversity_Weight', 'Uncertainty_Weight', 'Constraint_Strength', 'Ini_File_Path', 'StructureName'])
    
    # Load Parameter Space
    cursor.execute('SELECT * FROM ParameterSpace')
    param_space = cursor.fetchall()
    df_param_space = pd.DataFrame(param_space, columns=['ParamName', 'type', 'lower_bound', 'upper_bound', 'regulates'])
    
    # Load QoIs
    cursor.execute('SELECT * FROM QoIs')
    qois = cursor.fetchall()
    df_qois = pd.DataFrame(qois, columns=['QoI_Name', 'QoI_Type', 'ObsData_Column', 'QoI_distanceFunction', 'QoI_distanceWeight'])

    # Load GP Models
    cursor.execute('SELECT * FROM GP_Models')
    gp_models = cursor.fetchall()
    df_gp_models = pd.DataFrame(gp_models, columns=['IterationID', 'GP_Model', 'Hypervolume'])
    # Deserialize the GP_Model column
    df_gp_models['GP_Model'] = df_gp_models['GP_Model'].apply(lambda x: torch.load(io.BytesIO(x), map_location=torch.device('cpu')))

    # Load Samples
    cursor.execute('SELECT * FROM Samples')
    samples = cursor.fetchall()
    df_samples = pd.DataFrame(samples, columns=['IterationID', 'SampleID', 'ParamName', 'ParamValue'])

    # Load Output
    cursor.execute('SELECT * FROM Output')
    output = cursor.fetchall()
    df_output = pd.DataFrame(output, columns=['SampleID', 'ObjFunc', 'Noise_Std', 'Data'])
    # Deserialize the columns
    df_output['ObjFunc'] = df_output['ObjFunc'].apply(pickle.loads)
    df_output['Noise_Std'] = df_output['Noise_Std'].apply(pickle.loads)
    df_output['Data'] = df_output['Data'].apply(pickle.loads)

    conn.close()

    return df_metadata, df_param_space, df_qois, df_gp_models, df_samples, df_output