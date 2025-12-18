import os
import sqlite3
import pandas as pd
import numpy as np
import pickle

from uq_physicell import PhysiCell_Model

def safe_pickle_loads(data):
    """Safely deserialize pickled data from SQLite BLOB storage.
    
    This function handles different data types that might be returned from SQLite
    when retrieving BLOB data, including bytes, buffer objects, and memoryview.
    
    Args:
        data: The data to deserialize, which can be bytes, buffer, memoryview, or already deserialized.
    
    Returns:
        The deserialized Python object, or raises an error if deserialization fails.
    
    Raises:
        RuntimeError: If data remains in binary format after deserialization attempts.
    
    Example:
        >>> import pickle
        >>> original_data = {'key': 'value'}
        >>> serialized = pickle.dumps(original_data)
        >>> result = safe_pickle_loads(serialized)
        >>> print(result)  # {'key': 'value'}
    """
    # If data is None or empty, return as-is
    if data is None or data == b'':
        return data
    
    try:
        # Handle different SQLite BLOB return types
        if isinstance(data, bytes):
            result = pickle.loads(data)
        elif isinstance(data, memoryview):
            result = pickle.loads(bytes(data))
        elif hasattr(data, 'tobytes'):
            result = pickle.loads(data.tobytes())
        elif hasattr(data, '__bytes__'):
            result = pickle.loads(bytes(data))
        # Handle SQLite3.Row or other database-specific types
        elif hasattr(data, 'keys') and hasattr(data, '__getitem__'):
            return data
        else:
            result = data
        
        # Check if result is still binary data
        if isinstance(result, bytes) and len(result) > 0:
            raise RuntimeError(f"Data remains in binary format after deserialization: {str(result)[:50]}...")
        
        return result
            
    except Exception as e:
        if "remains in binary format" in str(e):
            raise  # Re-raise our custom error
        raise RuntimeError(f"Error deserializing data: {e}, data type: {type(data)}")

def create_structure(db_file: str):
    """Create the SQLite database structure for storing simulation analysis results.
    
    This function initializes a SQLite database with five tables designed to store
    all components of a sensitivity analysis or uncertainty quantification study.
    
    Args:
        db_file (str): Path to the SQLite database file to be created.
    
    Tables Created:
        - Metadata: Stores simulation metadata (sampler, config path, model structure)
        - ParameterSpace: Stores parameter definitions (name, bounds, reference values)
        - QoIs: Stores quantities of interest definitions (name, function)
        - Samples: Stores parameter samples (sample ID, parameter name, value)
        - Output: Stores simulation results (sample ID, replicate ID, serialized data)
    
    Note:
        If the database file already exists, it will be removed and recreated
        to ensure a clean structure.
    
    Example:
        >>> create_structure('sensitivity_analysis.db')
        >>> # Database created with all required tables
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

def insert_metadata(db_file: str, sampler: str, ini_file_path: str, strucName: str):
    """Insert metadata information into the Metadata table.
    
    Args:
        db_file (str): Path to the SQLite database file.
        sampler (str): Name of the sampling method used (e.g., 'Sobol', 'LHS', 'Morris').
        ini_file_path (str): Path to the PhysiCell configuration (.ini) file.
        strucName (str): Name or identifier of the model structure used.
    
    Raises:
        sqlite3.Error: If database connection or insertion fails.
    
    Example:
        >>> insert_metadata('study.db', 'Sobol', 'config/params.ini', 'tumor_growth')
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
    """Insert parameter space information into the ParameterSpace table.
    
    Args:
        db_file (str): Path to the SQLite database file.
        params_dict (dict): Dictionary containing parameter names and their properties.
            Each parameter should have keys: 'lower_bound', 'upper_bound', 
            'ref_value', and 'perturbation'.
    
    Raises:
        RuntimeError: If parameter insertion fails due to database errors.
    
    Example:
        >>> params = {
        ...     'param1': {
        ...         'lower_bound': 0.0,
        ...         'upper_bound': 1.0,
        ...         'ref_value': 0.5,
        ...         'perturbation': 0.1
        ...     }
        ... }
        >>> insert_param_space('study.db', params)
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        for param_name, properties in params_dict.items():
            if param_name == "samples": continue
            properties['perturbation'] = properties.get('perturbation', None)  # If 'perturbation' key does not exist, then it will be None
            properties['lower_bound'] = properties.get('lower_bound', None)  # If 'lower_bound' key does not exist, then it will be None
            properties['upper_bound'] = properties.get('upper_bound', None)  # If 'upper_bound' key does not exist, then it will be None
            # Convert list to string if it's a list
            if type(properties['perturbation']) == list:
                properties['perturbation'] = str(properties['perturbation'])
            print(f"Inserting parameter: {param_name} with properties: {properties}")
            cursor.execute('''
                INSERT INTO ParameterSpace (ParamName, Lower_Bound, Upper_Bound, ReferenceValue, Perturbation)
                VALUES (?, ?, ?, ?, ?)
            ''', (param_name, properties['lower_bound'], properties['upper_bound'], properties['ref_value'], properties['perturbation']))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting parameter space into the database: {e}")

def insert_qois(db_file: str, qois_dic: dict):
    """Insert quantities of interest (QoIs) into the QoIs table.
    
    Args:
        db_file (str): Path to the SQLite database file.
        qois_dic (dict): Dictionary of QoIs where keys are QoI names and values 
            are either lambda functions or string representations of functions.
    
    Raises:
        RuntimeError: If QoI insertion fails due to database errors.
    
    Example:
        >>> qois = {
        ...     'total_cells': lambda data: data['cell_count'].sum(),
        ...     'max_radius': 'lambda data: data["radius"].max()'
        ... }
        >>> insert_qois('study.db', qois)
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        print(f"Inserting {qois_dic} QoIs into the database")
        # If qois_dic is None or empty, insert a placeholder
        if qois_dic is None:
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

def insert_samples(db_file: str, dic_samples: dict):
    """Insert sample parameters into the Samples table.
    
    Args:
        db_file (str): Path to the SQLite database file.
        dic_samples (dict): Nested dictionary where outer keys are sample IDs 
            and inner dictionaries contain parameter names and values.
    
    Raises:
        RuntimeError: If sample insertion fails due to database errors.
    
    Example:
        >>> samples = {
        ...     0: {'param1': 0.5, 'param2': 1.2},
        ...     1: {'param1': 0.8, 'param2': 0.9}
        ... }
        >>> insert_samples('study.db', samples)
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

def insert_output(db_file: str, sample_id: int, replicate_id: int, result_data: bytes):
    """Insert simulation results into the Output table.
    
    Args:
        db_file (str): Path to the SQLite database file.
        sample_id (int): Unique identifier for the parameter sample.
        replicate_id (int): Identifier for the simulation replicate.
        result_data (bytes): Serialized simulation results data.
    
    Raises:
        RuntimeError: If output insertion fails due to database errors.
    
    Example:
        >>> import pickle
        >>> data = {'cells': [1, 2, 3], 'time': [0, 1, 2]}
        >>> serialized_data = pickle.dumps(data)
        >>> insert_output('study.db', 0, 1, serialized_data)
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

def load_metadata(db_file: str) -> pd.DataFrame:
    """Load metadata from the database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Sampler', 'Ini_File_Path', 'StructureName'].
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_metadata = load_metadata('study.db')
        >>> print(df_metadata['Sampler'].values[0])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM Metadata')
        metadata = cursor.fetchall()
        df_metadata = pd.DataFrame(metadata, columns=['Sampler', 'Ini_File_Path', 'StructureName'])
        return df_metadata
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading metadata: {e}")
    finally:
        conn.close()

def load_parameter_space(db_file: str) -> pd.DataFrame:
    """Load parameter space from the database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['ParamName', 'lower_bound', 'upper_bound', 
                      'ref_value', 'perturbation']. Perturbation column is converted 
                      from string to numpy array.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_params = load_parameter_space('study.db')
        >>> print(df_params[['ParamName', 'ReferenceValue']])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM ParameterSpace')
        parameter_space = cursor.fetchall()
        df_parameter_space = pd.DataFrame(parameter_space, 
                                         columns=['ParamName', 'lower_bound', 'upper_bound', 
                                                 'ref_value', 'perturbation'])
        # Convert Perturbation column from string representation to numpy array
        df_parameter_space['perturbation'] = df_parameter_space['perturbation'].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else x
        )
        return df_parameter_space
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading parameter space: {e}")
    finally:
        conn.close()

def load_qois(db_file: str) -> pd.DataFrame:
    """Load quantities of interest (QoIs) from the database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['QOI_Name', 'QOI_Function']. 
                     Returns DataFrame with None values if no QoIs are defined.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> df_qois = load_qois('study.db')
        >>> print(df_qois['QOI_Name'].to_list())
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM QoIs')
        qois = cursor.fetchall()
        if qois:
            df_qois = pd.DataFrame(qois, columns=['QOI_Name', 'QOI_Function'])
        else:
            df_qois = pd.DataFrame(columns=['QOI_Name', 'QOI_Function'])
            df_qois.loc[0] = [None, None]
        return df_qois
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading QoIs: {e}")
    finally:
        conn.close()

def load_samples(db_file: str) -> dict:
    """Load parameter samples from the database.
    
    Args:
        db_file (str): Path to the SQLite database file.
    
    Returns:
        dict: Dictionary where keys are sample IDs and values are dictionaries 
              of parameter names and values.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
    
    Example:
        >>> dic_samples = load_samples('study.db')
        >>> print(f"Sample 0 parameters: {dic_samples[0]}")
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM Samples')
        samples = cursor.fetchall()
        df_samples = pd.DataFrame(samples, columns=['SampleID', 'ParamName', 'ParamValue'])
        # Convert to dictionary of dictionaries sorted by SampleID
        dic_samples = df_samples.pivot(index="SampleID", columns="ParamName", 
                                      values="ParamValue").sort_index().to_dict(orient="index")
        return dic_samples
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading samples: {e}")
    finally:
        conn.close()

def load_output(db_file: str, sample_ids: list = None, replicate_ids: list = None, 
                load_data: bool = True) -> pd.DataFrame:
    """Load simulation output from the database with flexible filtering options.
    
    This function allows selective loading of simulation results, which is useful
    for memory efficiency and performance when working with large databases.
    
    Args:
        db_file (str): Path to the SQLite database file.
        sample_ids (list, optional): List of specific sample IDs to load. 
                                    If None, loads all samples.
        replicate_ids (list, optional): List of specific replicate IDs to load. 
                                       If None, loads all replicates.
        load_data (bool, optional): If True, deserializes the Data column. 
                                   If False, only loads SampleID and ReplicateID. 
                                   Default is True.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['SampleID', 'ReplicateID', 'Data'] if load_data=True,
                     or ['SampleID', 'ReplicateID'] if load_data=False. 
                     When load_data=True, the Data column contains deserialized objects.
    
    Raises:
        sqlite3.Error: If database connection or query fails.
        RuntimeError: If data deserialization fails.
    
    Examples:
        >>> # Load only metadata (SampleID, ReplicateID)
        >>> df = load_output('study.db', load_data=False)
        >>> 
        >>> # Load specific sample
        >>> df = load_output('study.db', sample_ids=[0, 1])
        >>> 
        >>> # Load specific replicate across all samples
        >>> df = load_output('study.db', replicate_ids=[0])
        >>> 
        >>> # Load specific sample and replicate combination
        >>> df = load_output('study.db', sample_ids=[5], replicate_ids=[0])
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        # Build the SQL query with optional filters
        query = 'SELECT SampleID, ReplicateID{} FROM Output'.format(
            ', Data' if load_data else ''
        )
        conditions = []
        params = []
        
        if sample_ids is not None:
            placeholders = ','.join('?' * len(sample_ids))
            conditions.append(f'SampleID IN ({placeholders})')
            params.extend(sample_ids)
        
        if replicate_ids is not None:
            placeholders = ','.join('?' * len(replicate_ids))
            conditions.append(f'ReplicateID IN ({placeholders})')
            params.extend(replicate_ids)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        cursor.execute(query, params)
        output = cursor.fetchall()
        
        if load_data:
            df_output = pd.DataFrame(output, columns=['SampleID', 'ReplicateID', 'Data'])
            # Deserialize the Data column
            df_output['Data'] = df_output['Data'].apply(safe_pickle_loads)
        else:
            df_output = pd.DataFrame(output, columns=['SampleID', 'ReplicateID'])
            # Placeholder column for consistency, dataframe is db from summary function or list of mcds
            df_output_test = load_output(db_file, sample_ids=[0], replicate_ids=[0])
            # If QoIs are already in the database and 'Data' column is not present
            if not 'Data' in df_output_test.columns:
                pass # No action needed, 'Data' column is not expected
            # Check if 'Data' column in df_qois_data is a series of mcds list - Case of db generated by generic summary function with NO QoI functions
            elif type(df_output_test['Data'].iloc[0]) == list:
                df_output['Data'] = [[] for _ in range(len(df_output))]
            # Check if 'Data' column in df_qois_data is a DataFrame - Case of db generated by custom summary function
            elif isinstance(df_output_test['Data'].iloc[0], pd.DataFrame):
                df_output['Data'] = [pd.DataFrame() for _ in range(len(df_output))]
        
        return df_output
    
    except sqlite3.Error as e:
        raise RuntimeError(f"Error loading output: {e}")
    finally:
        conn.close()

def load_data_unserialized(db_file: str, sample_ids: list = None, 
                          replicate_ids: list = None) -> pd.DataFrame:
    """Load and expand simulation output into time-series columns based on QoIs.
    
    This function loads the output data and, if QoIs are defined, expands each QoI
    and time array into separate columns (e.g., 'qoi_0', 'qoi_1', 'time_0', 'time_1').
    If no QoIs are defined, returns the raw deserialized data.
    
    Args:
        db_file (str): Path to the SQLite database file.
        sample_ids (list, optional): List of specific sample IDs to load. 
                                    If None, loads all samples.
        replicate_ids (list, optional): List of specific replicate IDs to load. 
                                       If None, loads all replicates.
    
    Returns:
        pd.DataFrame: DataFrame with SampleID, ReplicateID, and either:
                     - Expanded QoI time-series columns if QoIs are defined
                     - Raw Data column if no QoIs are defined
    
    Raises:
        RuntimeError: If data loading or processing fails.
    
    Examples:
        >>> # Load all data with QoI expansion
        >>> df = load_data_unserialized('study.db')
        >>> print(df.columns)  # ['SampleID', 'ReplicateID', 'total_cells_0', 'time_0', ...]
        >>> 
        >>> # Load specific sample
        >>> df = load_data_unserialized('study.db', sample_ids=[0])
    """
    # Load QoIs and output data
    df_qois = load_qois(db_file)
    df_output = load_output(db_file, sample_ids=sample_ids, 
                           replicate_ids=replicate_ids, load_data=True)
    
    # Initialize result DataFrame
    df_data_unserialized = df_output.copy()
    
    # If QoIs are defined, expand them into columns
    if df_qois['QOI_Name'].values[0] is not None:
        df_data_unserialized.drop(columns=['Data'], inplace=True)
        
        for qoi in df_qois['QOI_Name']:
            for i in range(df_output.shape[0]):
                qoi_values = df_output.at[i, 'Data'][qoi].to_numpy()
                time_values = df_output.at[i, 'Data']['time'].to_numpy()
                
                # Save time series data as separate columns
                for idx in range(len(qoi_values)):
                    df_data_unserialized.at[i, f"{qoi}_{idx}"] = qoi_values[idx]
                    df_data_unserialized.at[i, f"time_{idx}"] = time_values[idx]
    
    return df_data_unserialized
    
def load_structure(db_file: str, load_result: bool = True) -> tuple:
    """Load the complete database structure and return all data components.
    
    This is a convenience wrapper function that calls all modular load functions.
    For more control over what data is loaded, use the individual load functions:
    - load_metadata(db_file)
    - load_parameter_space(db_file)
    - load_qois(db_file)
    - load_samples(db_file)
    - load_output(db_file, sample_ids=None, replicate_ids=None, load_data=True)
    - load_data_unserialized(db_file, sample_ids=None, replicate_ids=None)
    
    Args:
        db_file (str): Path to the SQLite database file.
        load_result (bool, optional): If True, loads and deserializes all output data 
                                     with QoI expansion. If False, only loads SampleID 
                                     and ReplicateID. Default is True.
    
    Returns:
        tuple: A 5-tuple containing:
            - df_metadata (pd.DataFrame): Metadata information (sampler, config, structure)
            - df_parameter_space (pd.DataFrame): Parameter space definitions
            - df_qois (pd.DataFrame): Quantities of interest definitions
            - dic_samples (dict): Dictionary of parameter samples by sample ID
            - df_results (pd.DataFrame): Simulation results (expanded or metadata only)
    
    Raises:
        RuntimeError: If database loading fails.
    
    Examples:
        >>> # Load everything with full data
        >>> metadata, params, qois, samples, results = load_structure('study.db')
        >>> print(f"Loaded {len(samples)} samples with {len(results)} results")
        >>> 
        >>> # Load only metadata (no deserialization)
        >>> metadata, params, qois, samples, ids = load_structure('study.db', load_result=False)
        >>> print(f"Found {len(ids)} simulation results")
    """
    # Load all components using modular functions
    df_metadata = load_metadata(db_file)
    df_parameter_space = load_parameter_space(db_file)
    df_qois = load_qois(db_file)
    dic_samples = load_samples(db_file)
    
    if not load_result:
        # Only load SampleID and ReplicateID without data
        df_results = load_output(db_file, load_data=False)
    else:
        # Load and expand all data with QoI processing
        df_results = load_data_unserialized(db_file)
    
    return df_metadata, df_parameter_space, df_qois, dic_samples, df_results

def check_simulations_db(PhysiCellModel: PhysiCell_Model, sampler: str, param_dict: dict, 
                        dic_samples: dict, qois_dic: dict, db_file: str) -> tuple:
    """Check database existence and identify missing simulations.
    
    This function verifies if a database exists and contains all required simulation
    results. It compares the expected samples against completed simulations to
    identify any missing runs.
    
    Args:
        PhysiCellModel (PhysiCell_Model): The PhysiCell model instance for simulations.
        sampler (str): Name of the sampling method used (e.g., 'Sobol', 'LHS').
        param_dict (dict): Parameter space definition with bounds and reference values.
        dic_samples (dict): Dictionary of parameter samples to be simulated.
        qois_dic (dict): Dictionary of quantities of interest definitions.
        db_file (str): Path to the SQLite database file.
    
    Returns:
        tuple: A 2-tuple containing:
            - exist_db (bool): True if database exists and contains valid structure
            - parameters_missing (list): List of dictionaries for missing parameter sets
    
    Example:
        >>> model = PhysiCell_Model('config.ini')
        >>> exists, missing = check_simulations_db(
        ...     model, 'Sobol', params, samples, qois, 'study.db'
        ... )
        >>> print(f"Database exists: {exists}, Missing: {len(missing)} simulations")
    """
    parameters_missing, samples_missing, replicates_missing = [], [], []
    if not os.path.exists(db_file):
        return False, parameters_missing, samples_missing, replicates_missing

    try:
        # Load the database structure
        df_metadata, df_parameter_space, df_qois, dic_samples_db, df_data_unserialized = load_structure(db_file, load_result=False)

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
            # Use set items comparison to ignore order of keys
            if set(dic_samples_db[sample_id].items()) != set(params.items()):
                raise ValueError(f"ParameterSpace mismatch for SampleID {sample_id}. Expected: {params}, Found: {dic_samples_db[sample_id]}.")

        # Check if QoIs match the expected values
        if qois_dic:
            if df_qois['QOI_Name'].to_list() != list(qois_dic.keys()):
                raise ValueError(f"QoIs mismatch. Expected: {list(qois_dic.keys())}, Found: {df_qois['QOI_Name'].to_list()}.")
            if df_qois['QOI_Function'].to_list() != list(qois_dic.values()):
                raise ValueError(f"QoI Functions mismatch. Expected: {list(qois_dic.values())}, Found: {df_qois['QOI_Function'].to_list()}.")

        # Check for missing samples
        missing_samples = [sample_id for sample_id in dic_samples.keys() if sample_id not in set(dic_samples_db.keys())]
        # print(f"Missing samples: {missing_samples}")
        
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

def get_database_type(db_file: str) -> bool:
    """Determine the type of analysis database (Model Analysis or Bayesian Optimization).
    
    This function examines the database structure to identify whether it contains
    Model Analysis (MA) or Bayesian Optimization (BO) data based on table schemas.
    
    Args:
        db_file (str): Path to the SQLite database file to examine.
    
    Returns:
        str or None: Returns 'MA' for Model Analysis, 'BO' for Bayesian Optimization,
                    or None if the database type cannot be determined.
    
    Example:
        >>> db_type = get_database_type('analysis.db')
        >>> if db_type == 'MA':
        ...     print("This is a Model Analysis database")
        >>> elif db_type == 'BO':
        ...     print("This is a Bayesian Optimization database")
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