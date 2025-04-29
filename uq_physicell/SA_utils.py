import os
import sqlite3
import io
import pickle
import numpy as np
import pandas as pd
from typing import Union

from uq_physicell import PhysiCell_Model, generic_QoI

# Helper function to create named functions from strings
def create_named_function_from_string(func_str, qoi_name):
    """
    Dynamically creates a named function from a string and assigns it to the module's global scope.
    Parameters:
    - func_str: The string representation of the function.
    - qoi_name: The name of the function to be created.
    Return:
    - The created function.
    """
    # Check if the function already exists in the global scope
    func_name = f"named_{qoi_name}"
    if func_name not in globals():
        exec(
            f"def {func_name}(*args, **kwargs):\n"
            f"    return eval({repr(func_str)}, {{'len': len, 'pd': pd, 'np': np}})(*args, **kwargs)",
            globals()
        )
    return globals()[func_name]

def summary_function(outputPath, summaryFile, dic_params, SampleID, ReplicateID, qoi_functions, drop_columns):
    """
    A standalone function to encapsulate the summary function logic.
    Parameters:
    - outputPath: Path to the output folder
    - summaryFile: Path to the summary file
    - dic_params: Dictionary of parameters
    - SampleID
    - ReplicateID
    - qoi_functions: Dictionary of QoI functions (keys as names, values as lambda functions or strings)
    Return:
    - The result of the generic_QoI function.
    """
    try:
        return generic_QoI(
            outputPath=outputPath,
            summaryFile=summaryFile,
            dic_params=dic_params,
            SampleID=SampleID,
            ReplicateID=ReplicateID,
            qoi_funcs=qoi_functions,
            mode='time_series',
            RemoveFolder=True,
            drop_columns=drop_columns
        )
    except Exception as e:
        raise RuntimeError(f"Error in summary function: {e}")

def run_replicate(PhysiCellModel, sample_id, replicate_id, ParametersXML, ParametersRules, qoi_functions, drop_columns):
    """
    Run a single replicate of the simulation and return the results.
    Parameters:
    - PhysiCellModel: The PhysiCell model instance.
    - sample_id: The sample ID.
    - replicate_id: The replicate ID.
    - ParametersXML: The parameters for the XML.
    - ParametersRules: The parameters for the rules.
    - qoi_functions: The QoI functions or None.
    Return:
    - if qoi_functions: The QoIs of the simulation.
    - if qoi_functions==None: list of mcds from pcdl.
    """
    # Check if qoi_functions is not None
    if qoi_functions:
        # Recreate QoI functions from their string representations
        recreated_qoi_funcs = {
            qoi_name: create_named_function_from_string(qoi_value, qoi_name)
            for qoi_name, qoi_value in qoi_functions.items()
        }
    else: 
        recreated_qoi_funcs = None

    # Pass the updated qoi_functions to the PhysiCellModel
    result_data = PhysiCellModel.RunModel(
        sample_id, replicate_id, ParametersXML,
        ParametersRules=ParametersRules,
        SummaryFunction=lambda *args: summary_function(*args, qoi_functions=recreated_qoi_funcs, drop_columns=drop_columns),
    )
    # print(f"Simulation completed for SampleID: {sample_id}, ReplicateID: {replicate_id}\n Result.head(): {result_data.head()}")
    
    # Serialize the DataFrame using pickle
    serialized_result = pickle.dumps(result_data)  # Convert to binary using pickle

    return sample_id, replicate_id, serialized_result

def create_db_structure(db_file):
    """
    Create the database structure with three tables:
    1. Metadata: Stores information about the sensitivity analysis (SA_type, SA_method, SA_sampler, etc.).
    2. Inputs: Stores sample_id, param_name, and param_value.
    3. Results: Stores sample_id, replicate_id, and result data as binary.
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
                SA_Type TEXT,
                SA_Method TEXT,
                SA_Sampler TEXT,
                Num_Samples INTEGER,
                Param_Names TEXT,
                Bounds TEXT,
                Reference_Values TEXT,
                Perturbations TEXT,
                QoIs TEXT,
                QoIs_Functions TEXT,
                Ini_File_Path TEXT,
                StructureName TEXT
            )
        ''')
        # Create Inputs table
        cursor.execute('''
            CREATE TABLE Inputs (
                SampleID INTEGER,
                ParamName TEXT,
                ParamValue DOUBLE
            )
        ''')
        # Create Results table
        cursor.execute('''
            CREATE TABLE Results (
                SampleID INTEGER,
                ReplicateID INTEGER,
                Data BLOB
            )
        ''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        raise RuntimeError(f"Error generating tables: {e}")

def convert_to_str(param_names: list, bounds: Union[list, None], ref_values: list, pert: list, qois: Union[list, None], qois_fun: Union[list, None]) -> tuple:
    """
    Convert lists to strings for database storage.
    Parameters:
    - param_names: List of parameter names.
    - bounds: List of bounds for the parameters.
    - ref_values: List of reference values for the parameters.
    - pert: List of perturbations for the parameters.
    - qois: List of QoIs (keys as names, values as lambda functions or strings).
    - qois_fun: List of QoI functions (if any).
    Return:
    - Tuple of strings representing the converted values.
    """
    # QoIs dictionary
    # Check if qois and qois_fun are None
    if (qois == None) & (qois_fun == None):
        qois = None; qois_fun = None
    if ( (qois == None) & (qois_fun == None) ):
        qois_str = 'None'; qois_fun_str = 'None'
    # QoIs are lists of name and functions
    else: 
        # Convert the QoIs list to a comma-separated string
        qois_str = ', '.join(qois) if isinstance(qois, list) else ValueError("QoIs should be a list or a string")
        qois_fun_str = ', '.join(qois_fun) if isinstance(qois_fun, list) else ValueError("QoIs_Functions should be a string")
    # Bounds if None or list of list
    if bounds == None: 
        bounds_str = 'None'
    else: 
        bounds_str = ', '.join([str(bounds_elem) for bounds_elem in bounds])
    # Perturbations is a list or list of list
    pert_str = ', '.join([str(p) for p in pert])
    # Parameter names and reference values are lists
    param_names_str = ', '.join(param_names)
    ref_values_str = ', '.join([str(val) for val in ref_values])
    return param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str

def insert_metadata(cursor, SA_type, SA_method, SA_sampler, num_samples, param_names, bounds, ref_values, pert, qois_dic, ini_file_path, strucName):
    """
    Insert metadata information into the Metadata table.
    Parameters:
    - cursor: SQLite cursor object.
    - SA_type: Sensitivity analysis type.
    - SA_method: Sensitivity analysis method.
    - SA_sampler: Sampling method.
    - num_samples: Number of samples.
    - param_names: List of parameter names.
    - bounds: List of bounds for the parameters.
    - ref_values: List of reference values for the parameters.
    - pert: List of perturbations for the parameters.
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings) - If empty store all data as a list of mcds.
    - ini_file_path: Path to the initial file.
    - strucName: Structure name.
    """
    # Check if qois_dic is empty
    qoi_keys = list(qois_dic.keys()) if qois_dic else None
    qois_func = list(qois_dic.values()) if qois_dic else None
    # Convert the values to strings for database storage
    try:
        param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str = convert_to_str(
            param_names, bounds, ref_values, pert, qoi_keys, qois_func
        )
    except Exception as e:
        raise ValueError(f"Error converting values to strings: {e}")

    try:
        cursor.execute('''
            INSERT INTO Metadata (SA_Type, SA_Method, SA_Sampler, Num_Samples, Param_Names, Bounds, Reference_Values, Perturbations, QoIs, QoIs_Functions, Ini_File_Path, StructureName)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (SA_type, SA_method, SA_sampler, num_samples, param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str, ini_file_path, strucName))
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting: {e}")

def insert_inputs(cursor, dic_samples, sample_ids=None):
    """
    Insert input parameters into the Inputs table.
    Parameters:
    - cursor: SQLite cursor object.
    - dic_samples: Dictionary of the dictionaries of samples
    - sample_ids: List of sample IDs to insert. If None, all samples will be inserted.
    """
    sample_ids = sample_ids if sample_ids is not None else dic_samples.keys()
    try:
        for samp_id in sample_ids:
            for param_name, param_value in dic_samples[samp_id].items():
                cursor.execute('INSERT INTO Inputs (SampleID, ParamName, ParamValue) VALUES (?, ?, ?)',
                               (samp_id, param_name, param_value))
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting inputs into the database: {e}")

def insert_output(cursor, sample_id, replicate_id, result_data):
    """
    Insert simulation results into the Results table.
    Parameters:
    - cursor: SQLite cursor object.
    - sample_id: The sample ID.
    - replicate_id: The replicate ID.
    - result_data: The simulation results data (as binary).
    """
    try:
        cursor.execute('INSERT INTO Results (SampleID, ReplicateID, Data) VALUES (?, ?, ?)',
                       (sample_id, int(replicate_id), sqlite3.Binary(result_data)))
    except sqlite3.Error as e:
        raise RuntimeError(f"Error inserting output into the database: {e}")

def check_existing_sa(PhysiCellModel, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, dic_samples, qois_dic, db_file):
    """
    Check if the database file exists and if all simulations have been completed.
    Parameters:
    - PhysiCellModel: The PhysiCell model instance.
    - SA_type: The type of sensitivity analysis.
    - SA_method: The method of sensitivity analysis.
    - SA_sampler: The sampler used for sensitivity analysis.
    - param_names: List of parameter names.
    - ref_values: List of reference values for the parameters.
    - bounds: List of bounds for the parameters.
    - perturbations: List of perturbations for the parameters.
    - dic_samples: Dictionary of the dictionaries of samples
    - qois_dic: Dictionary of QoIs (keys as names, values as lambda functions or strings) - If empty store all data as a list of mcds.
    - db_file: Path to the database file.
    Return:
    - exist_db: True if the database exists and all simulations are completed, False otherwise.
    - parameters_missing: List of dictionaries of missing parameters.
    - samples_missing: List of missing samples.
    - replicates_missing: List of missing replicates.
    """

    parameters_missing, samples_missing, replicates_missing = [], [], []
    if not os.path.exists(db_file):
        return False, parameters_missing, samples_missing, replicates_missing

    try:
       
        # Load the database structure
        df_metadata, df_inputs, df_results = load_db_structure(db_file)

        # Convert parameters to strings for comparison
        qoi_keys = list(qois_dic.keys()) if qois_dic else None
        qois_func = list(qois_dic.values()) if qois_dic else None
        param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str = convert_to_str(
            param_names, bounds, ref_values, perturbations, qoi_keys, qois_func
        )
        metadata_checks = {
            "SA_Type": SA_type,
            "SA_Method": SA_method,
            "SA_Sampler": SA_sampler,
            "Num_Samples": len(dic_samples),
            "Param_Names": param_names_str,
            "Bounds": bounds_str,
            "Reference_Values": ref_values_str,
            "Perturbations": pert_str,
            "QoIs": qois_str,
            "QoIs_Functions": qois_fun_str,
        }
        for key, expected in metadata_checks.items():
            # Do not check Bounds because it can differ according to the samples seed
            if key == "Bounds": continue
            # print(f"Checking {key}: Expected: {expected}, Found: {df_metadata[key].iloc[0]}")
            if df_metadata[key].iloc[0] != expected:
                raise ValueError(f"{key} mismatch. Expected: {expected}, Found: {df_metadata[key].iloc[0]}.")

        # Check for missing samples
        samples_db = df_inputs.pivot(index="SampleID", columns="ParamName", values="ParamValue").reindex(columns=param_names).to_dict(orient="index")
        missing_samples = [sample_id for sample_id in dic_samples.keys() if sample_id not in set(samples_db.keys())]
        print(f"Missing samples: {missing_samples}")
        
        # Add missing samples to the database
        if missing_samples:
            print(f"Adding {len(missing_samples)} missing samples to the database.")
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            insert_inputs(cursor, dic_samples, missing_samples)
            conn.commit()
            conn.close()

        # Check for missing replicates
        completed_replicates = df_results.groupby("SampleID")["ReplicateID"].apply(set).to_dict()
        # print(f"Completed replicates: {completed_replicates}")
        for sample_id in dic_samples.keys():
            missing_replicates = set(range(PhysiCellModel.numReplicates)) - completed_replicates.get(sample_id, set())
            if missing_replicates:
                # Add from dic_samples if missing samples - Meaning that extra samples were added
                # else add from db_samples to avoid duplicates
                if missing_samples:
                    parameters_missing.extend(dic_samples[sample_id] for _ in missing_replicates)
                else:
                    parameters_missing.extend(samples_db[sample_id] for _ in missing_replicates)
                samples_missing.extend(sample_id for _ in missing_replicates)
                replicates_missing.extend(missing_replicates)

        if replicates_missing:
            print(f"Missing {len(replicates_missing)} simulations.")
        # print(len(parameters_missing), samples_missing, replicates_missing)
    except Exception as e:
        raise ValueError(f"Error while checking the database: {e}")

    return True, parameters_missing, samples_missing, replicates_missing

def load_db_structure(db_file):
    """
    Load the database structure and return the dataframes for metadata, inputs, and results.
    Parameters:
    - db_file: Path to the database file.
    Return:
    - df_metadata: DataFrame containing metadata information.
    - df_inputs: DataFrame containing input parameters.
    - df_results: DataFrame containing simulation results.
    """
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Load Metadata
    cursor.execute('SELECT * FROM Metadata')
    metadata = cursor.fetchall()         
    df_metadata = pd.DataFrame(metadata, columns=['SA_Type', 'SA_Method', 'SA_Sampler', 'Num_Samples', 'Param_Names', 'Bounds', 'Reference_Values', 'Perturbations', 'QoIs', 'QoIs_Functions', 'Ini_File_Path', 'StructureName'])
    # Load Inputs
    cursor.execute('SELECT * FROM Inputs')
    inputs = cursor.fetchall()
    df_inputs = pd.DataFrame(inputs, columns=['SampleID', 'ParamName', 'ParamValue'])
    # Load Results
    cursor.execute('SELECT * FROM Results')
    results = cursor.fetchall()
    conn.close()
    df_results = pd.DataFrame(results, columns=['SampleID', 'ReplicateID', 'Data'])
    # Deserialize the Data column
    df_results['Data'] = df_results['Data'].apply(pickle.loads)
    # If QoIs is None - all data was stored as a list of mcds
    if df_metadata['QoIs'].values[0] == "None": 
        return df_metadata, df_inputs, df_results
    else: # If QoIs are not None - converts df_results['Data'] to qois columns
        # Convert Data column to qois
        df_results_modified = df_results.copy()
        df_results_modified.drop(columns=['Data'], inplace=True)
        for qoi in df_metadata['QoIs'].values[0].split(', '):
            for i in range(df_results.shape[0]):
                # print(f"index: {i} Extracting {qoi} from Data for SampleID: {df_results.at[i, 'SampleID']}, ReplicateID: {df_results.at[i, 'ReplicateID']} Results: {df_results.at[i, 'Data'][qoi].to_numpy()}")
                qoi_values = df_results.at[i, 'Data'][qoi].to_numpy()
                time_values = df_results.at[i, 'Data']['time'].to_numpy()
                # Save time series data
                for id in range(len(qoi_values)):
                    df_results_modified.at[i, f"{qoi}_{id}"] = qoi_values[id]
                    df_results_modified.at[i, f"time_{id}"] = time_values[id]
    return df_metadata, df_inputs, df_results_modified

def OAT_analyze(dic_samples, dic_qoi):
    """
    Perform OAT analysis on the results.
    Parameters:
    - dic_samples: dictionary of the dictionaries of samples
    - dic_qoi: dictionary of QoIs
    Return:
    - dic_results: dictionary of the results
    """
    # Remove unused variables ref_pars and qoi_ref
    # Extract parameter samples and QoI samples
    par_samples = np.array([list(sample.values()) for sample in dic_samples.values()])
    # Normalize the parameter samples
    par_samples = (par_samples - par_samples.min(axis=0)) / (par_samples.max(axis=0) - par_samples.min(axis=0))
    # Sample 0 is the reference sample
    ref_pars = par_samples[0]; par_samples = par_samples[1:]
    qoi_ref = dic_qoi[0]
    # Extract QoI samples, excluding the reference sample (SampleID different of 0)
    qoi_samples = np.array([qoi for sample_id, qoi in dic_qoi.items() if sample_id != 0])
    # Initialize the results dictionary
    dic_results = {}
    for id, par in enumerate(dic_samples[0].keys()):
        # Calculate the mean and std deviation of the QoIs for each parameter
        par_var = np.abs(par_samples[:, id] - ref_pars[id])
        non_zero_indices = np.where(par_var != 0)[0]
        dic_results[par] = np.abs(qoi_samples[non_zero_indices] - qoi_ref) / par_var[non_zero_indices]  # Compute SI without skipping

    return dic_results

def extract_qoi_from_db(db_file, qoi_functions):
    """
    Extracts the QoI values from the database and returns them as a DataFrame.
    Parameters:
    - db_file: Path to the database file.
    - qoi_functions: Dictionary of QoI functions (keys as names, values as lambda functions or strings).
    Return:
    - df_qois: DataFrame containing the extracted QoI values.
    """
    # Load the database structure
    df_metadata, df_samples, df_output = load_db_structure(db_file)
    # Recreate QoI functions from their string representations
    recreated_qoi_funcs = {
        qoi_name: create_named_function_from_string(qoi_value, qoi_name)
        for qoi_name, qoi_value in qoi_functions.items()
    }
    df_qois = pd.DataFrame()
    for SampleID in df_output['SampleID'].unique():
        df_sample = df_output[df_output['SampleID'] == SampleID]
        for ReplicateID in df_sample['ReplicateID'].unique():
            mcds_ts_list = df_sample[df_sample['ReplicateID'] == ReplicateID]['Data'].values[0]
            # print(f"SampleID: {SampleID}, ReplicateID: {ReplicateID} - mcds_ts_list: {mcds_ts_list}")
            data = {'SampleID': SampleID, 'ReplicateID': ReplicateID}
            for id_time, mcds in enumerate(mcds_ts_list):
                data[f"time_{id_time}"] = mcds.get_time()
                # cell dataframe
                df_cell = mcds.get_cell_df()
                try: 
                    for qoi_name, qoi_func in recreated_qoi_funcs.items():
                        # Store the QoI value in the data dictionary
                        data[f"{qoi_name}_{id_time}"] =  qoi_func(df_cell)
                except Exception as e:
                    raise RuntimeError(f"Error computing QoIs in extract_qoi_from_db function for SampleID: {SampleID}, ReplicateID: {ReplicateID} - QoI: {qoi_name}_{id_time}: {e}")
            # Store the data in a DataFrame
            df_qoi_replicate = pd.DataFrame({key: [value] for key, value in data.items()})
            df_qois = pd.concat([df_qois, df_qoi_replicate], ignore_index=True)
    df_qois = df_qois.reset_index(drop=True)
    return df_qois

def reshape_expanded_data(expanded_data, qoi_columns):
    try:
        # Ensure QoI columns are numeric
        for qoi in qoi_columns:
            expanded_data[qoi] = pd.to_numeric(expanded_data[qoi], errors='coerce')

        # Create a unique time_id for each time step
        expanded_data['time_id'] = expanded_data.groupby(['SampleID', 'ReplicateID']).cumcount()

        # Pivot the DataFrame to create columns for each QoI and time_id
        reshaped_data = expanded_data.pivot_table(
            index=['SampleID', 'ReplicateID'],
            columns='time_id',
            values=qoi_columns + ['time']
        )

        # Flatten the multi-index columns
        reshaped_data.columns = [
            f"{col[0]}_{int(col[1])}" if col[0] != 'time' else f"time_{int(col[1])}"
            for col in reshaped_data.columns
        ]
        reshaped_data.reset_index(inplace=True)

        return reshaped_data
    except Exception as e:
        raise ValueError(f"Error reshaping expanded data: {e}")