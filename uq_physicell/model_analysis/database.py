import os
import sqlite3
import pandas as pd
import numpy as np
import pickle

from uq_physicell import PhysiCell_Model

def create_structure(db_file:str):
    """
    Create the database structure with five tables:
    1. Metadata: Stores simulations metadata (sampler, .ini config path, and model structure name).
    2. ParameterSpace: Stores the parameter space information (ParamName, Lower_Bound, Upper_Bound, ReferenceValue, Perturbation).
    3. QoIs: Stores the quantities of interest (QoI_Name, QoI_Function).
    4. Samples: Stores the samples (SampleID, ParamName, ParamValue).
    5. Output: Stores the output of the simulations (SampleID, ReplicateID, Data).
    Parameters:
    - db_file: Path to the database file.
    """
    if os.path.exists(db_file):
        try: os.remove(db_file)
        except Exception as e:
            raise RuntimeError(f"Error removing existing database file: {e}")
    try:
        # Create a new SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Create Metadata table
        cursor.execute('''
            CREATE TABLE Metadata (
                Sampler TEXT,
                Ini_File_Path TEXT,
                StructureName TEXT
            )
        ''')
        # Create ParameterSpace table
        cursor.execute('''
            CREATE TABLE ParameterSpace (
                ParamName TEXT,
                Lower_Bound DOUBLE,
                Upper_Bound DOUBLE,
                ReferenceValue DOUBLE,
                Perturbation TEXT
            )
        ''')
        # Create QoIs table
        cursor.execute('''
            CREATE TABLE QoIs (
                QOI_Name TEXT,
                QOI_Function TEXT
            )
        ''')
        # Create Samples table
        cursor.execute('''
            CREATE TABLE Samples (
                SampleID INTEGER,
                ParamName TEXT,
                ParamValue DOUBLE
            )
        ''')
        # Create Output table
        cursor.execute('''
            CREATE TABLE Output (
                SampleID INTEGER,
                ReplicateID INTEGER,
                Data BLOB
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error generating tables: {e}")

def insert_metadata(db_file: str, sampler:str, ini_file_path: str, strucName: str):
    """
    Insert metadata information into the Metadata table.
    Parameters:
    - db_file: SQLite cursor object.
    - sampler: Sampling method.
    - ini_file_path: Path to the initial file.
    - strucName: Structure name.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO Metadata (Sampler, Ini_File_Path, StructureName)
            VALUES (?, ?, ?)
        ''', (sampler, ini_file_path, strucName))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting: {e}")

def insert_param_space(db_file: str, params_dict: dict):
    """
    Insert parameter space information into the ParameterSpace table.
    Parameters:
    - db_file: Path to the database file.
    - params_dict: Dictionary containing parameter names and their properties.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        for param_name, properties in params_dict.items():
            if type(properties['perturbation']) == list:
                properties['perturbation'] = str(properties['perturbation'])
            cursor.execute('''
                INSERT INTO ParameterSpace (ParamName, Lower_Bound, Upper_Bound, ReferenceValue, Perturbation)
                VALUES (?, ?, ?, ?, ?)
            ''', (param_name, properties['lower_bounds'], properties['upper_bounds'], properties['ref_value'], properties['perturbation']))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting parameter space into the database: {e}")

def insert_qois(db_file: str, qois_dic:dict):
    """
    Insert QoIs into the QoIs table.
    Parameters:
    - db_file: Path to the database file.
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings).
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        # If qois_dic is empty, insert a placeholder
        if not qois_dic:
            cursor.execute('INSERT INTO QoIs (QOI_Name, QOI_Function) VALUES (?, ?)', (None, None))
            conn.commit()
            conn.close()
            return
        else: # Insert each QoI into the QoIs table
            for qoi_name, qoi_func in qois_dic.items():
                cursor.execute('INSERT INTO QoIs (QOI_Name, QOI_Function) VALUES (?, ?)', (qoi_name, qoi_func))
            conn.commit()
            conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting QoIs into the database: {e}")

def insert_samples(db_file: str, dic_samples:dict):
    """
    Insert sample parameters into the Samples table.
    Parameters:
    - db_file: Path to the database file.
    - dic_samples: Dictionary of the dictionaries of samples
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        for sample_id, params in dic_samples.items():
            for param_name, param_value in params.items():
                cursor.execute('INSERT INTO Samples (SampleID, ParamName, ParamValue) VALUES (?, ?, ?)', (sample_id, param_name, param_value))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting samples into the database: {e}")

def insert_output(db_file: str, sample_id:int, replicate_id:int, result_data:bytes):
    """
    Insert simulation results into the Output table.
    Parameters:
    - db_file: Path to the database file.
    - sample_id: The sample ID.
    - replicate_id: The replicate ID.
    - result_data: The simulation results data (as binary).
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO Output (SampleID, ReplicateID, Data) VALUES (?, ?, ?)',
                       (sample_id, int(replicate_id), sqlite3.Binary(result_data)))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting output into the database: {e}")
    
def load_structure(db_file:str) -> tuple:
    """
    Load the database structure and return the dataframes for metadata, inputs, and results.
    Parameters:
    - db_file: Path to the database file.
    Return:
    - df_metadata: DataFrame containing metadata information.
    - df_parameter_space: DataFrame containing parameter space information.
    - df_qois: DataFrame containing QoIs information.
    - dic_samples: Dictionary containing samples.
    - df_results: DataFrame containing simulation results.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Load Metadata
    cursor.execute('SELECT * FROM Metadata')
    metadata = cursor.fetchall()         
    df_metadata = pd.DataFrame(metadata, columns=['Sampler', 'Ini_File_Path', 'StructureName'])
    
    # Load Parameter Space
    cursor.execute('SELECT * FROM ParameterSpace')
    parameter_space = cursor.fetchall()
    df_parameter_space = pd.DataFrame(parameter_space, columns=['ParamName', 'Lower_Bound', 'Upper_Bound', 'ReferenceValue', 'Perturbation'])
    # Convert Perturbation column from string representation of lists to a numpy array of floats
    # This assumes that Perturbation is stored as a string representation of a list, e.g., "[0.1, 0.2, 0.3]"
    df_parameter_space['Perturbation'] = df_parameter_space['Perturbation'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

    # Load QoIs
    cursor.execute('SELECT * FROM QoIs')
    qois = cursor.fetchall()
    df_qois = pd.DataFrame(qois, columns=['QOI_Name', 'QOI_Function'])
    
    # Load Samples
    cursor.execute('SELECT * FROM Samples')
    samples = cursor.fetchall()
    df_samples = pd.DataFrame(samples, columns=['SampleID', 'ParamName', 'ParamValue'])
    # Convert df_samples to a dictionary of dictionaries with external keys sampleID and internal keys ParamName - sorted by sampleID
    dic_samples = df_samples.pivot(index="SampleID", columns="ParamName", values="ParamValue").sort_index().to_dict(orient="index")
    
    # Load Output
    cursor.execute('SELECT * FROM Output')
    output = cursor.fetchall()
    conn.close()
    df_output = pd.DataFrame(output, columns=['SampleID', 'ReplicateID', 'Data'])
    # Deserialize the Data column
    df_output['Data'] = df_output['Data'].apply(pickle.loads)
    # If QoIs are not None - converts df_output['Data'] to qois columns
    print("df_qois['QOI_Name'].values[0]:", df_qois['QOI_Name'].values[0])
    if df_qois['QOI_Name'].values[0] != None: 
        # Convert Data column to qois
        df_data_unserialized.drop(columns=['Data'], inplace=True)
        for qoi in df_qois['QOI_Name']:
            for i in range(df_output.shape[0]):
                # print(f"index: {i} Extracting {qoi} from Data for SampleID: {df_output.at[i, 'SampleID']}, ReplicateID: {df_output.at[i, 'ReplicateID']} Results: {df_output.at[i, 'Data'][qoi].to_numpy()}")
                qoi_values = df_output.at[i, 'Data'][qoi].to_numpy()
                time_values = df_output.at[i, 'Data']['time'].to_numpy()
                # Save time series data
                for id in range(len(qoi_values)):
                    df_data_unserialized.at[i, f"{qoi}_{id}"] = qoi_values[id]
                    df_data_unserialized.at[i, f"time_{id}"] = time_values[id]
    # If QoIs are None - keep the Data column as is
    else:
        df_data_unserialized = df_output

    return df_metadata, df_parameter_space, df_qois, dic_samples, df_data_unserialized

def check_simulations_db(PhysiCellModel:PhysiCell_Model, sampler:str, param_dict:dict, dic_samples:dict, qois_dic:dict, db_file:str) -> tuple:
    """
    Check if the database file exists and if all simulations have been completed.
    Parameters:
    - PhysiCellModel: The PhysiCell model instance.
    - sampler: The sampler used for sensitivity analysis.
    - param_dict: Dictionary containing parameter names, reference values, lower bounds, upper bounds, and perturbations.
    - dic_samples: Dictionary of the dictionaries of samples
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings) - If empty store all data as a list of mcds.
    - db_file: Path to the database file.
    Return:
    - exist_db: Boolean indicating if the database file exists and is valid.
    - parameters_missing: List of dictionaries of missing parameters.
    - samples_missing: List of missing samples.
    - replicates_missing: List of missing replicates.
    """
    parameters_missing, samples_missing, replicates_missing = [], [], []
    if not os.path.exists(db_file):
        return False, parameters_missing, samples_missing, replicates_missing

    try:
       
        # Load the database structure
        df_metadata, df_parameter_space, df_qois, dic_samples_db, df_data_unserialized = load_structure(db_file)

        # Check if Metadata matches the expected values
        metadata_checks = {
            "Sampler": sampler,
            "Ini_File_Path": PhysiCellModel.configFilePath,
            "StructureName": PhysiCellModel.keyModel,
        }
        for key, expected in metadata_checks.items():
            # print(f"Checking {key}: Expected: {expected}, Found: {df_metadata[key].iloc[0]}")
            if df_metadata[key].values[0] != expected:
                raise ValueError(f"{key} mismatch. Expected: {expected}, Found: {df_metadata[key].values[0]}.")

        # Check if ParameterSpace matches the expected values
        for sample_id, params in dic_samples.items():
            if not np.array_equal(dic_samples_db[sample_id].values(), params.values()):
                raise ValueError(f"ParameterSpace mismatch for SampleID {sample_id}. Expected: {params.values()}, Found: {dic_samples_db[sample_id].values()}.")

        # Check if QoIs match the expected values
        if qois_dic: # not None
            if df_qois['QOI_Name'].to_list() != list(qois_dic.keys()):
                raise ValueError(f"QoIs mismatch. Expected: {list(qois_dic.keys())}, Found: {df_qois['QOI_Name'].to_list()}.")
            if df_qois['QOI_Function'].to_list() != list(qois_dic.values()):
                raise ValueError(f"QoI Functions mismatch. Expected: {list(qois_dic.values())}, Found: {df_qois['QOI_Function'].to_list()}.")

        # Check for missing samples
        missing_samples = [sample_id for sample_id in dic_samples.keys() if sample_id not in set(dic_samples_db.keys())]
        print(f"Missing samples: {missing_samples}")
        
        # Add missing samples to the database
        if missing_samples:
            print(f"Adding {len(missing_samples)} missing samples to the database.")
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            insert_samples(cursor, dic_samples, missing_samples)
            conn.commit()
            conn.close()

        # Check for missing replicates
        completed_replicates = df_data_unserialized.groupby("SampleID")["ReplicateID"].apply(set).to_dict()
        # print(f"Completed replicates: {completed_replicates}")
        for sample_id in dic_samples.keys():
            missing_replicates = set(range(PhysiCellModel.numReplicates)) - completed_replicates.get(sample_id, set())
            if missing_replicates:
                # Add from dic_samples if missing samples - Meaning that extra samples were added
                # else add from db_samples to avoid duplicates
                if missing_samples:
                    parameters_missing.extend(dic_samples[sample_id] for _ in missing_replicates)
                else:
                    parameters_missing.extend(dic_samples_db[sample_id] for _ in missing_replicates)
                samples_missing.extend(sample_id for _ in missing_replicates)
                replicates_missing.extend(missing_replicates)

        if replicates_missing:
            print(f"Missing {len(replicates_missing)} simulations.")
        # print(len(parameters_missing), samples_missing, replicates_missing)
    except Exception as e:
        raise ValueError(f"Error while checking the database: {e}")

    return True, parameters_missing, samples_missing, replicates_missing

def get_database_type(db_file:str) -> bool:
    """
    Check if the database file is a valid Model Analysis or Bayesian Optimization database.
    Parameters:
    - db_file: Path to the database file.
    Return:
    - Database type: MA, BO, or None if the database is not valid.
    """
    if not os.path.exists(db_file):
        return None

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Check if Metadata table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Metadata'")
    if not cursor.fetchone():
        return False
    # Check if required columns exist in Metadata table
    cursor.execute("PRAGMA table_info(Metadata)")
    columns = [col[1] for col in cursor.fetchall()]
    conn.close()
    # Model analysis database should have one column as 'Sampler'
    if 'Sampler' in columns:
        return "MA"
    # BO database should have one column as 'BO_Method'
    elif 'BO_Method' in columns:
        return "BO"
    else:
        return None