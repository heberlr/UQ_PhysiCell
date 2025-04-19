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
    """
    func_name = f"named_{qoi_name}"
    if func_name not in globals():
        exec(
            f"def {func_name}(*args, **kwargs):\n"
            f"    return eval({repr(func_str)}, {{'len': len, 'pd': pd, 'np': np}})(*args, **kwargs)",
            globals()
        )
    return globals()[func_name]

def summary_function(outputPath, summaryFile, dic_params, SampleID, ReplicateID, qoi_functions):
    """
    A standalone function to encapsulate the summary function logic.
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
            RemoveFolder=True
        )
    except Exception as e:
        raise RuntimeError(f"Error in summary function: {e}")

def run_replicate(PhysiCellModel, sample_id, replicate_id, ParametersXML, ParametersRules, qoi_functions):
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
        SummaryFunction=lambda *args: summary_function(*args, qoi_functions=recreated_qoi_funcs)
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
    """
    if os.path.exists(db_file):
        os.remove(db_file)
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

def convert_to_str(param_names: list, bounds: Union[list, None], ref_values: list, pert: list, qois: Union[list, None], qois_fun: Union[list, None]) -> tuple:
    """
    Convert lists to strings for database storage.
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
    """
    # Check if qois_dic is empty
    qoi_keys = list(qois_dic.keys()) if qois_dic else None
    qois_func = list(qois_dic.values()) if qois_dic else None
    # Convert the values to strings for database storage
    try: param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str = convert_to_str(param_names, bounds, ref_values, pert, qoi_keys, qois_func)
    except Exception as e:
        raise ValueError(f"Error converting values to strings: {e}")
    
    cursor.execute('''
        INSERT INTO Metadata (SA_Type, SA_Method, SA_Sampler, Num_Samples, Param_Names, Bounds, Reference_Values, Perturbations, QoIs, QoIs_Functions, Ini_File_Path, StructureName)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (SA_type, SA_method, SA_sampler, num_samples, param_names_str, bounds_str, ref_values_str, pert_str, qois_str, qois_fun_str, ini_file_path, strucName))

def insert_inputs(cursor, samples, PhysiCellModel, sample_ids=None):
    """
    Insert input parameters into the Inputs table.
    """
    # Check if sample_ids is None
    if sample_ids is None:
        sample_ids = range(samples.shape[0])
    param_names = list(PhysiCellModel.XML_parameters_variable.values()) + list(PhysiCellModel.parameters_rules_variable.values())
    for sample_id, sample in zip(sample_ids, samples):
        for param_name, param_value in zip(param_names, sample):
            cursor.execute('INSERT INTO Inputs (SampleID, ParamName, ParamValue) VALUES (?, ?, ?)',
                           (sample_id, param_name, param_value))

def insert_output(cursor, sample_id, replicate_id, result_data):
    """
    Insert simulation results into the Results table.
    """
    cursor.execute('INSERT INTO Results (SampleID, ReplicateID, Data) VALUES (?, ?, ?)',
                   (sample_id, int(replicate_id), sqlite3.Binary(result_data)))

def check_existing_sa(PhysiCellModel, SA_type, SA_method, SA_sampler, param_names, ref_values, bounds, perturbations, samples, qois_dic, db_file):
    """
    Check if the database file exists and if all simulations have been completed.
    """
    if not os.path.exists(db_file):
        return False, [], [], []

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
            "Num_Samples": len(samples),
            "Param_Names": param_names_str,
            "Bounds": bounds_str,
            "Reference_Values": ref_values_str,
            "Perturbations": pert_str,
            "QoIs": qois_str,
            "QoIs_Functions": qois_fun_str,
        }
        for key, expected in metadata_checks.items():
            if df_metadata[key].iloc[0] != expected:
                raise ValueError(f"{key} mismatch. Expected: {expected}, Found: {df_metadata[key].iloc[0]}.")

        # Check for missing samples
        samples_db = df_inputs.pivot(index="SampleID", columns="ParamName", values="ParamValue").to_numpy()
        missing_samples = [i for i, sample in enumerate(samples) if not any(np.array_equal(sample, db_sample) for db_sample in samples_db)]

        # Check for missing replicates
        completed_replicates = df_results.groupby("SampleID")["ReplicateID"].apply(set).to_dict()
        parameters_missing, samples_missing, replicates_missing = [], [], []
        for sample_id in range(len(samples)):
            missing_replicates = set(range(PhysiCellModel.numReplicates)) - completed_replicates.get(sample_id, set())
            if missing_replicates:
                parameters_missing.extend([samples[sample_id]] * len(missing_replicates))
                samples_missing.append([sample_id] * len(missing_replicates))
                replicates_missing.extend(missing_replicates)

        # Add missing samples to the database
        if missing_samples:
            print(f"Adding {len(missing_samples)} missing samples to the database.")
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            insert_inputs(cursor, samples, PhysiCellModel, missing_samples)
            conn.commit()
            conn.close()

        if replicates_missing:
            print(f"Missing {len(replicates_missing)} simulations.")

    except Exception as e:
        ValueError(f"Error while checking the database: {e}")

    return True, parameters_missing, samples_missing, replicates_missing

def load_db_structure(db_file):
    """
    Load the database structure and return the dataframes for metadata, inputs, and results.
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

def OAT_analyze(param_names, samples, qoi_array):
    """
    Perform OAT analysis on the results.
    """
    # Sample 0 is the reference sample
    ref_pars = samples[0]; qoi_ref = qoi_array[0]
    par_samples = samples[1:]; qoi_samples = qoi_array[1:]
    # Initialize the results dictionary
    dic_results = {}
    for id, par in enumerate(param_names):
        # Calculate the mean and std deviation of the QoIs for each parameter
        par_var = np.abs(par_samples[:, id] - ref_pars[id])
        non_zero_indices = np.where(par_var != 0)[0]
        dic_results[par] = np.abs(qoi_samples[non_zero_indices] - qoi_ref) / par_var[non_zero_indices]  # Compute SI without skipping

    return dic_results

def extract_qoi_from_db(db_file, qoi_functions):
    df_metadata, df_samples, df_output = load_db_structure(db_file)
    # Recreate QoI functions from their string representations
    recreated_qoi_funcs = {
        qoi_name: create_named_function_from_string(qoi_value, qoi_name)
        for qoi_name, qoi_value in qoi_functions.items()
    }
    PhysiCellModel = PhysiCell_Model(df_metadata['Ini_File_Path'][0], df_metadata['StructureName'][0])
    df_qois = pd.DataFrame()
    for SampleID in df_output['SampleID'].unique():
        for ReplicateID in df_output['ReplicateID'].unique():
            mcds_ts_list = df_output[(df_output['SampleID'] == SampleID) & (df_output['ReplicateID'] == ReplicateID)]['Data'].values[0]
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