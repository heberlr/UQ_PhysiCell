import pathlib
import urllib.request
import zipfile
import re
import os
import subprocess
from typing import Union
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import random
import configparser # read config *.ini file
import ast # string to literal
import shutil
import time
import copy
import datetime

class PhysiCell_Model:
    """ A class to manage PhysiCell model configurations and executions.
    
    This class handles the setup of PhysiCell models, including reading configuration files,
    generating XML files, and running simulations with specified parameters.
    
    Parameters:
        configFilePath (str): Path to the configuration file (INI format).
        keyModel (str): Key in the configuration file to identify the model.
        verbose (bool): If True, prints detailed information during execution.
    """
    def __init__(self, configFilePath: str, keyModel: str, verbose: bool = False) -> None:
        self.configFilePath = configFilePath
        self.keyModel = keyModel
        self.verbose = verbose
        self._load_config(configFilePath, keyModel)
        self._load_xml_reference()
        self._load_rules_reference()
        # Dictionary to track all active processes
        self.active_processes = {}
        
    def _load_config(self, configFilePath: str, keyModel: str) -> None:
        configFile = configparser.ConfigParser()
        if self.verbose:
            print(f"\t> Constructor PhysiCell_Model: {keyModel} at {configFilePath}...")
            print(f"\t\t>> Reading config file: {configFilePath} ...")
        with open(configFilePath) as fd:
            configFile.read_file(fd)
        
        # Mandatory variables
        self.XML_RefPath = configFile[keyModel]['configFile_ref']
        if not os.path.exists(self.XML_RefPath): raise ValueError(f"Error! XML file {self.XML_RefPath} not found!")
        self.numReplicates = int(configFile[keyModel]['numReplicates'])

        # PhysiCell executable
        self.PC_executable = configFile[keyModel]['executable']
        if self.verbose:
            print(f"\t\t>> Checking executable format ...")
        if os.name == 'nt':
            self.PC_executable = self.PC_executable.replace(os.altsep, os.sep) + '.exe'
        if not os.path.exists(self.PC_executable): 
            try:
                # If PhysiCell path is not found, assume it is in the same folder as the executable
                PC_path = configFile[keyModel]['PhysiCell_path'] if 'PhysiCell_path' in configFile[keyModel] else os.path.dirname(self.PC_executable)
                # If Model path is not found, assume it is in the same folder of reference XML file Model_path/config/XML_RefFile
                Model_path = configFile[keyModel]['Model_path'] if 'Model_path' in configFile[keyModel] else pathlib.Path(self.XML_RefPath).parent.parent
                compile_physicell(PC_path, Model_path, executable_path=self.PC_executable)
            except ValueError as e:
                raise ValueError(f"Error! Executable {self.PC_executable} not found and cannot be compiled! {e}")
        # Path of XML file of reference

        #### Optional variables
        self.projName = configFile[keyModel].get('projName', fallback=keyModel) # project name
        self.XML_name = configFile[keyModel].get('configFile_name', fallback="config_S%06d_R%03d.xml") # config files structure
        self.input_folder = configFile[keyModel].get("configFile_folder", fallback="UQ_PC_InputFolder/") # folder to store input files (.xmls, .csv, and .txt)
        self.output_folder = configFile[keyModel].get('outputs_folder', fallback="UQ_PC_OutputFolder/") # folder to store the output folders
        self.outputs_folder_name = configFile[keyModel].get('outputs_folder_name', fallback="output_S%06d_R%03d/") # structure of output folders
        self.timeout = configFile[keyModel].get('timeout', fallback=60) # timeout for waiting to write files
        self.generate_summary_File = configFile[keyModel].get('generate_summary_file', fallback=False) # generate csv file
        self.output_summary_Path = self.output_folder+'SummaryFile_%06d_%02d.csv' # path of the summary file
        # Rules files and folder
        self.RULES_RefPath = configFile[keyModel].get('rulesFile_ref', fallback=None)
        self.RULES_name = configFile[keyModel].get('rulesFile_name', fallback="rules_S%06d_R%03d.csv")
        self.omp_num_threads = configFile[keyModel].get('omp_num_threads', fallback=None)
        self.XML_parameters = configFile[keyModel].get('parameters', fallback=dict())
        if self.XML_parameters:
            self.XML_parameters = ast.literal_eval(self.XML_parameters)
            # if number of omp_threads defined in fixed parameters overwrite the one in the .ini file
            if ".//parallel/omp_num_threads" in list(self.XML_parameters.keys()): self.omp_num_threads = self.XML_parameters['.//parallel/omp_num_threads']
        self.XML_parameters_variable = {k: v[1] for k, v in self.XML_parameters.items() if isinstance(v, list)}
        self.XML_parameters_fixed = {k: v for k, v in self.XML_parameters.items() if not isinstance(v, list)}
        self.parameters_rules = configFile[keyModel].get('parameters_rules', fallback=dict())
        if self.parameters_rules:
            self.parameters_rules = ast.literal_eval(self.parameters_rules)
        self.parameters_rules_variable = {k: v[1] for k, v in self.parameters_rules.items() if isinstance(v, list)}
        self.parameters_rules_fixed = {k: v for k, v in self.parameters_rules.items() if not isinstance(v, list)}

    def _load_xml_reference(self) -> None:
        tree = ET.parse(self.XML_RefPath)
        self.xml_ref_root = tree.getroot()
        if self.verbose:
            print(f"\t\t>> Checking parameters in XML file ...")
        # if not in .ini file, get from XML
        if not self.omp_num_threads: 
            self.omp_num_threads = _get_xml_element_value(self.xml_ref_root, './/parallel/omp_num_threads')
        for param_key in self.XML_parameters.keys():
            try:
                _get_xml_element_value(self.xml_ref_root, param_key)
            except ValueError as e:
                raise ValueError(f"Error in parameters_xml. {e}")

    def _load_rules_reference(self) -> None:
        self.default_rules = _get_rules(self.RULES_RefPath) if self.RULES_RefPath else None
        if self.verbose:
            print(f"\t\t>> Checking parameters in RULES file ...")
        for param_key in self.parameters_rules.keys():
            try:
                _get_rule_index_in_csv(self.default_rules, param_key)
            except ValueError as e:
                raise ValueError(f"Error in parameters_rules. {e}")

    def _get_xml_path(self, sampleID: int, replicateID: int) -> str:
        filePath = self.input_folder + self.XML_name % (sampleID, replicateID)
        os.makedirs(os.path.dirname(self.input_folder), exist_ok=True)
        return filePath

    def _get_output_path(self, sampleID: int, replicateID: int) -> str:
        folderPath = self.output_folder + self.outputs_folder_name % (sampleID, replicateID)
        os.makedirs(os.path.dirname(folderPath), exist_ok=True)
        return folderPath

    def _get_rules_fileName(self, sampleID: int, replicateID: int) -> str:
        return self.RULES_name % (sampleID, replicateID)

    def info(self) -> None:
        """ 
        Print model configuration information. 
        """
        # Use relative paths for privacy (avoid exposing personal directory structure)
        rel_executable = os.path.relpath(self.PC_executable, os.getcwd())
        rel_xml_ref = os.path.relpath(self.XML_RefPath, os.getcwd())
        rel_rules_ref = os.path.relpath(self.RULES_RefPath, os.getcwd()) if self.RULES_RefPath else None
        
        print(f"""
        Project name: {self.projName} 
        Executable: {rel_executable}
        Number of replicates for each parameter set: {self.numReplicates} 
        Config. file of reference: {rel_xml_ref}
        Folder to save config. files: {self.input_folder} 
        Folder to save output folders: {self.output_folder}
        Rules file of reference: {rel_rules_ref}
        Name of output folders: {self.outputs_folder_name}
        Number of omp threads for each simulation: {self.omp_num_threads}
        Number of parameters for sampling in XML: {len(self.XML_parameters_variable)}
        Parameters in XML: { [param_name for param_name in self.XML_parameters_variable.values()] }
        Number of parameters for sampling in RULES: {len(self.parameters_rules_variable)}
        Parameters in RULES: { [param_name for param_name in self.parameters_rules_variable.values()] }
        """)

    def _copy(self):
        return copy.deepcopy(self)

    def RunModel(self, SampleID: int, ReplicateID: int, Parameters: Union[np.ndarray, dict] = dict(), ParametersRules: Union[np.ndarray, dict] = dict(), RemoveConfigFile: bool = True, SummaryFunction: Union[None, str] = None) -> Union[None, pd.DataFrame]:
        """ 
        Run a single simulation with specified parameters.
        
        Args:
            SampleID (int): Identifier for the parameter sample
            ReplicateID (int): Identifier for the simulation replicate
            Parameters (np.ndarray or dict, optional): Parameter values for XML configuration
            ParametersRules (np.ndarray or dict, optional): Parameter values for RULES configuration
            RemoveConfigFile (bool, optional): If True, removes the generated XML and RULES files after simulation
            SummaryFunction (function, optional): Function to summarize simulation output
        """
        return _run_model(self, SampleID, ReplicateID, Parameters, ParametersRules, RemoveConfigFile, SummaryFunction)
    def run_simulation_subprocess(self, XMLFile, sample_id=None, replicate_id=None):
        """
        Start the simulation as a subprocess and return the process handle.
        
        Args:
            XMLFile (str): Path to the XML configuration file for the simulation
            sample_id (int, optional): Identifier for the parameter sample
            replicate_id (int, optional): Identifier for the simulation replicate
            
        Returns:
            subprocess.Popen: Process handle for the running simulation
        """
        callingModel = [self.PC_executable, XMLFile]
        process = subprocess.Popen(callingModel, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        # Register the process if sample_id and replicate_id are provided
        if sample_id is not None and replicate_id is not None:
            process_id = f"{sample_id}_{replicate_id}"
            self.active_processes[process_id] = {
                'process': process,
                'pid': process.pid,
                'sample_id': sample_id,
                'replicate_id': replicate_id,
                'xml_file': XMLFile,
                'start_time': datetime.datetime.now()
            }
            if self.verbose:
                print(f"Registered process {process_id} with PID {process.pid} - running simulation ...")
        
        return process

    def _terminate_simulation(self, process=None, process_id=None, sample_id=None, replicate_id=None):
        """
        Terminate a running simulation subprocess.
        
        Can identify the process by:
        - Direct process handle
        - Process ID (in format "{sample_id}_{replicate_id}")
        - Sample ID and replicate ID pair
        
        Args:
            process (subprocess.Popen, optional): Process handle to terminate
            process_id (str, optional): Process ID in format "{sample_id}_{replicate_id}"
            sample_id (int, optional): Sample ID to identify process
            replicate_id (int, optional): Replicate ID to identify process
            
        Returns:
            int: Return code from the terminated process, or None if process not found
        """
        # Resolve the process to terminate
        if process is None:
            if process_id is not None:
                if process_id in self.active_processes:
                    process = self.active_processes[process_id]['process']
                else:
                    return None
            elif sample_id is not None and replicate_id is not None:
                process_id = f"{sample_id}_{replicate_id}"
                if process_id in self.active_processes:
                    process = self.active_processes[process_id]['process']
                else:
                    return None
            else:
                return None
        
        # Terminate the process if it's still running
        return_code = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=2)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    if self.verbose:
                        print(f"Failed to kill process {process.pid}")
        
        # Remove process from active_processes if it was tracked
        if process_id in self.active_processes:
            del self.active_processes[process_id]
        elif sample_id is not None and replicate_id is not None:
            process_id = f"{sample_id}_{replicate_id}"
            if process_id in self.active_processes:
                del self.active_processes[process_id]
                
        return return_code
        
    def terminate_all_simulations(self):
        """
        Terminate all active simulation processes.
        
        Returns:
            dict: Dictionary of process IDs and their termination return codes
        """
        results = {}
        process_ids = list(self.active_processes.keys())  # Create a copy of keys to avoid dict size change during iteration
        
        for process_id in process_ids:
            if self.verbose:
                print(f"Terminating process {process_id}")
            return_code = self._terminate_simulation(process_id=process_id)
            results[process_id] = return_code
            
        return results
        
    def get_active_processes_info(self):
        """
        Get information about all active processes.
        
        Returns:
            dict: Dictionary containing information about all active processes
        """
        # Update status of all processes
        for process_id, info in self.active_processes.items():
            process = info['process']
            if process.poll() is not None:
                info['status'] = 'finished'
                info['return_code'] = process.returncode
            else:
                info['status'] = 'running'
                info['runtime'] = (datetime.datetime.now() - info['start_time']).total_seconds()
        
        return self.active_processes 
    
    def remove_io_folders(self):
        if os.path.exists(self.input_folder):
            shutil.rmtree(self.input_folder)
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

def _check_parameters_input(model: PhysiCell_Model, parameters_input_xml: Union[np.ndarray, dict], parameters_input_rules: Union[np.ndarray, dict]) -> None:
    if model.verbose:
        print(f"\t\t\t\t>>>> Checking XML parameters input {type(parameters_input_xml)} ...")
    if isinstance(parameters_input_xml, dict):
        if set(model.XML_parameters_variable.values()) != set(parameters_input_xml.keys()):
            raise ValueError(f"""Error: XML parameters defined in dictionary:
                             {parameters_input_xml.keys()} 
                             are different from keys defined in {model.configFilePath}: 
                             {model.XML_parameters_variable.values()}.""")
    elif isinstance(parameters_input_xml, np.ndarray):
        if len(model.XML_parameters_variable.keys()) != parameters_input_xml.shape[0]:
            raise ValueError(f"Error: number of XML parameters defined input: {parameters_input_xml.shape[0]} are different from keys defined in {model.configFilePath}: {len(model.XML_parameters_variable.keys())}.")
    else:
        raise ValueError("Error: XML parameters input need to be a numpy array 1D or a dictionary.")
    if model.verbose:
        print(f"\t\t\t\t>>>> Checking RULES parameters type {type(parameters_input_rules)} ...")
    if isinstance(parameters_input_rules, dict):
        if set(model.parameters_rules_variable.values()) != set(parameters_input_rules.keys()):
            raise ValueError(f"""Error: RULES parameters defined in dictionary:
                             {parameters_input_rules.keys()} 
                             are different from keys defined in {model.configFilePath}: 
                             {model.parameters_rules_variable.values()}.""")
    elif isinstance(parameters_input_rules, np.ndarray):
        if len(model.parameters_rules_variable.keys()) != parameters_input_rules.shape[0]:
            raise ValueError(f"Error: number of RULES parameters defined input: {parameters_input_rules.shape[0]} are different from keys defined in {model.configFilePath}: {len(model.parameters_rules_variable.keys())}.")
    else:
        raise ValueError("Error: RULES parameters input need to be a numpy array 1D or a dictionary.")

def _setup_model_input(model: PhysiCell_Model, SampleID: int, ReplicateID: int, parameters_input: Union[np.ndarray, dict], parameters_rules_input: Union[np.ndarray, dict]) -> None:
    try:
        if model.verbose:
            print(f"\t\t\t>>> Checking parameters input ...")
        _check_parameters_input(model, parameters_input, parameters_rules_input)
    except ValueError as e:
        raise ValueError(f"Error in parameters input!\n{e}")

    dic_xml_parameters = model.XML_parameters.copy()
    XMLFile = model._get_xml_path(SampleID, ReplicateID)
    if model.parameters_rules:
        if model.verbose:
            print(f"\t\t\t>>> Setting up rules input ...")
        dic_xml_parameters['.//cell_rules/rulesets/ruleset/folder'] = model.input_folder
        dic_xml_parameters['.//cell_rules/rulesets/ruleset/filename'] = model._get_rules_fileName(SampleID, ReplicateID)
        csvFile_out = model.input_folder + model._get_rules_fileName(SampleID, ReplicateID)
        dic_rules_temp = {}
        for idx, param_key, param_name in zip(range(len(model.parameters_rules_variable)), model.parameters_rules_variable.keys(), model.parameters_rules_variable.values()):
            single_param_rule = param_key.split(",")[-1]
            if isinstance(parameters_rules_input, dict):
                dic_rules_temp[idx] = [parameters_rules_input[param_name], param_key, single_param_rule]
            else:
                dic_rules_temp[idx] = [parameters_rules_input[idx], param_key, single_param_rule]
        for idx, param_key in enumerate(model.parameters_rules_fixed.keys()):
            single_param_rule = param_key.split(",")[-1]
            dic_rules_temp[idx + len(model.parameters_rules_variable)] = [float(model.parameters_rules_fixed[param_key]), param_key, single_param_rule]
        if model.verbose:
            # Use relative path for privacy
            rel_csvFile_out = os.path.relpath(csvFile_out, os.getcwd())
            print(f"\t\t\t>>> Generating rules file {rel_csvFile_out} ...")
        try:
            _generate_csv_file(model.default_rules, csvFile_out, dic_rules_temp)
        except ValueError as e:
            raise ValueError(f"Error in generating rules file! {e}")

    if model.verbose:
        print(f"\t\t\t>>> Setting up XML input...")
    dic_xml_parameters['.//save/folder'] = model._get_output_path(SampleID, ReplicateID)
    dic_xml_parameters['./parallel/omp_num_threads'] = model.omp_num_threads
    
    # If not defined in .ini file, set initial condition folder to the folder of the reference XML file
    if (_get_xml_element_value(model.xml_ref_root, './/initial_conditions/cell_positions[@enabled]') == "true") and ('.//initial_conditions/cell_positions/folder' not in list(model.XML_parameters.keys())):
        dic_xml_parameters['.//initial_conditions/cell_positions/folder'] = os.path.dirname(model.XML_RefPath)
    
    try:
        _get_xml_element_value(model.xml_ref_root, './/options/random_seed')
        dic_xml_parameters['.//options/random_seed'] = "system_clock"
    except ValueError:
        try:
            _get_xml_element_value(model.xml_ref_root, './/user_parameters/random_seed')
            dic_xml_parameters['.//user_parameters/random_seed'] = random.randint(0, 4294967295)
        except ValueError as e:
            raise ValueError(f"Error in setting random seed. {e}")
    for idx, param_key, param_name in zip(range(len(model.XML_parameters_variable)), model.XML_parameters_variable.keys(), model.XML_parameters_variable.values()):
        if isinstance(parameters_input, dict):
            dic_xml_parameters[param_key] = parameters_input[param_name]
        else:
            dic_xml_parameters[param_key] = parameters_input[idx]
    if model.verbose:
        # Use relative path for privacy
        rel_XMLFile = os.path.relpath(XMLFile, os.getcwd())
        print(f"\t\t\t>>> Generating XML file {rel_XMLFile} ...")
    try:
        _generate_xml_file(pathlib.Path(model.XML_RefPath), pathlib.Path(XMLFile), dic_xml_parameters, model.timeout)
    except ValueError as e:
        raise ValueError(f"Error in generating XML file! {e}")

def _run_model(model: PhysiCell_Model, SampleID: int, ReplicateID: int, Parameters: Union[np.ndarray, dict] = dict(), ParametersRules: Union[np.ndarray, dict] = dict(), RemoveConfigFile: bool = True, SummaryFunction: Union[None, str] = None) -> Union[None, pd.DataFrame]:
    if model.verbose:
        print(f"\t> Running - Sample:{SampleID}, Replicate: {ReplicateID}, Parameters XML: {Parameters}, Parameters rules: {ParametersRules}...")
    try:
        try:
            if model.verbose:
                print(f"\t\t>> Setting up model input ...")
            _setup_model_input(model, SampleID, ReplicateID, Parameters, parameters_rules_input=ParametersRules)
        except ValueError as e:
            raise ValueError(f"Error in setup_model_input! (Sample: {SampleID} and Replicate: {ReplicateID}).\n{e}")

        XMLFile = model._get_xml_path(SampleID, ReplicateID)

        if model.verbose:
            print(f"\t\t>> Running model ...")
        # Use run_simulation_subprocess with tracking
        process = model.run_simulation_subprocess(XMLFile, SampleID, ReplicateID)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise ValueError(f"""Error: model output error! 
            Executable: {model.PC_executable} XML File: {XMLFile}. returned: {str(process.returncode)}
            Last 1000 characters of the PhysiCell output:
            {stdout[-1000:]}""")
        elif model.verbose:
            print("\t\t>> Simulation completed successfully!")

        if RemoveConfigFile:
            if model.verbose:
                print("\t\t>> Removing config files ...")
            try:
                os.remove(pathlib.Path(XMLFile))
                if model.parameters_rules:
                    filenameRule = model._get_rules_fileName(SampleID, ReplicateID)
                    os.remove(pathlib.Path(model.input_folder + filenameRule))
            except OSError as e:
                print(f"Error removing files: {e}")

        if SummaryFunction:
            if model.verbose:
                print(f"\t\t>> Running summary function: {SummaryFunction} ...")
            OutputFolder = model._get_output_path(SampleID, ReplicateID)
            dic_params = {}
            if isinstance(Parameters, dict):
                dic_params = {param_name: Parameters[param_name] for param_name in model.XML_parameters_variable.values()}
            else:
                dic_params = {param_name: Parameters[i] for i, param_name in enumerate(model.XML_parameters_variable.values())}

            if model.parameters_rules:
                if isinstance(ParametersRules, dict):
                    for param_key in model.parameters_rules_variable.values():
                        dic_params[param_key] = ParametersRules[param_key]
                else:
                    for i, param_key in enumerate(model.parameters_rules_variable.values()):
                        dic_params[param_key] = ParametersRules[i]
            try:
                if model.generate_summary_File:
                    SummaryFile = model.output_summary_Path % (SampleID, ReplicateID)
                    if model.verbose:
                        print(f"\t\t\t>>> Generating summary file {SummaryFile}...")
                else:
                    if model.verbose:
                        print(f"\t\t\t>>> Returning summary data ...")
                    SummaryFile = None
                return SummaryFunction(OutputFolder, SummaryFile, dic_params, SampleID, ReplicateID)
            except (OSError, SystemError) as e:
                raise ValueError(f"Error in SummaryFunction! (Sample: {SampleID} and Replicate: {ReplicateID}).\n\t{e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred in RunModel: {e}")

_attr_name_re = re.compile(r"\[@([A-Za-z_:][\w:.\-]*)")
def _get_xml_element_value(xml_root: ET.Element, key: str) -> str:
    elems = xml_root.findall(key)
    if len(elems) != 1:
        raise ValueError(f""" Error in getting XML element!
                Multiples occurrences or none found to this key: {key}, occurrences: {[pos.text for pos in elems]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    # Get last segment of the key to extract the attribute name
    last_seg = key.rsplit("/", 1)[-1]
    # Check and extract the attribute name using regex
    attribute = _attr_name_re.search(last_seg)
    elem = elems[0]
    if attribute:
        return elem.get(attribute.group(1))
    return elem.text

def _set_xml_element_value(xml_root: ET.Element, key: str, val: Union[str, int, float]) -> None:
    elems = xml_root.findall(key)
    if len(elems) != 1:
        raise ValueError(f""" Error in setting XML element!
                Multiples occurrences or none found to this key: {key}, occurrences: {[pos.text for pos in elems]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    # Get last segment of the key to extract the attribute name
    last_seg = key.rsplit("/", 1)[-1]
    # Check and extract the attribute name using regex
    attribute = _attr_name_re.search(last_seg)
    elem = elems[0]
    if attribute:
        elem.set(attribute.group(1), str(val))
    else:
        elem.text = str(val)

def _generate_xml_file(xml_file_in: str, xml_file_out: str, dic_parameters: dict, max_wait_time: float) -> None:
    shutil.copyfile(xml_file_in, xml_file_out)
    start_time = time.time()
    while not os.path.exists(xml_file_out) or os.path.getsize(xml_file_out) != os.path.getsize(xml_file_in):
        if time.time() - start_time > max_wait_time:
            raise TimeoutError(f"Timeout: Waiting for {xml_file_out} to be fully written exceeded {max_wait_time} seconds.")
        time.sleep(0.1)
    tree = ET.parse(xml_file_out)
    xml_root = tree.getroot()
    for key in dic_parameters.keys():
        val = dic_parameters[key]
        try:
            _set_xml_element_value(xml_root, key, val)
        except ValueError as e:
            raise ValueError(f"Error in setting parameter {key} with value {val}. {e}")
        # print(key,val)
    tree.write(xml_file_out)

def _get_rules(filename: str) -> list:
    default_rules = []
    try:
        with open(filename, newline='') as csvfile:
            fieldNames = ['cell_type', 'signal', 'direction', 'behavior', 'saturation', 'half_max', 'hill_power', 'dead']
            reader = csv.DictReader(csvfile, fieldnames=fieldNames, delimiter=',')
            for row in reader:
                default_rules.append(row)
        return default_rules
    except FileNotFoundError:
        raise ValueError(f"Error! File {filename} not found!")

def _get_rule_index_in_csv(rules: list, key_rules: str) -> int:
    try:
        cell_type, signal, direction, behavior, parameter = key_rules.split(",")
    except ValueError:
        raise ValueError("Error in rule format: Provide 'cell_type,signal,direction,behavior,parameter' where parameter can be 'saturation', 'half_max', 'hill_power', 'dead', or 'inactive'.")
    try:
        for idx, rule in enumerate(rules):
            if ( (cell_type == rule['cell_type']) and (signal == rule['signal']) and 
                (direction == rule['direction']) and (behavior == rule['behavior']) and
                (parameter in ['saturation', 'half_max', 'hill_power', 'dead', 'inactive']) ):
                return idx
    except ValueError:
        raise ValueError(f"Error! Rule {cell_type},{signal},{direction},{behavior} not found or parameter {parameter} does not exist!")

def _generate_csv_file(rules: list, csv_file_out: str, dic_parameters_rules: dict) -> None:
    try:
        with open(csv_file_out, 'w', newline='') as csvfile:
            fieldNames = ['cell_type', 'signal', 'direction', 'behavior', 'saturation', 'half_max', 'hill_power', 'dead']
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames, delimiter=',')
            rules_temp = rules.copy()
            rules_inactived = []
            for parameterID in dic_parameters_rules.keys():
                value, key_rule, param_name = dic_parameters_rules[parameterID]
                index_rule = _get_rule_index_in_csv(rules, key_rule)
                if (param_name not in fieldNames) and (param_name != "inactive"):
                    raise ValueError(f"Error! Parameter {param_name} not found in RULE: {key_rule}. Available parameters: {fieldNames} and inactive option.")
                # if the parameter is 'inactive', remove the rule or ignore it
                if (param_name == "inactive"):
                    if ( value == 1 or value == 'true' or value == 'True' or value == True ): rules_inactived.append(index_rule)
                    continue
                else:
                    rules_temp[index_rule][param_name] = value
            for id, rule in enumerate(rules_temp):
                # if the rule is not inactived, write it to the csv file
                if id not in rules_inactived:
                    writer.writerow(rule)
    except:
        raise ValueError(f"Error generating csv file.")

def get_physicell(target_dir: str, force_download=False, interactive=False):
    """
    Download PhysiCell from GitHub and extract it to the target directory.
    
    Args:
        target_dir (str): Directory where PhysiCell will be downloaded and extracted.
        force_download (bool): If True, download even if PhysiCell already exists.
                              If False, check for existing installation first.
        interactive (bool): If True, prompt user for action when PhysiCell exists.
                           If False, skip download when PhysiCell exists (unless force_download=True).
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Define the target directory
        target_path = pathlib.Path(target_dir)
        
        # Create the target directory if it doesn't exist
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Check if PhysiCell already exists
        physicell_dir = target_path / "PhysiCell-master"
        if physicell_dir.exists():
            # Show path relative to current working directory for privacy
            rel_path = os.path.relpath(str(physicell_dir.resolve()), os.getcwd())
            print(f"PhysiCell already exists at: {rel_path}")
            
            if not force_download:
                if interactive:
                    while True:
                        choice = input("Do you want to (r)eplace, (s)kip, or (c)ancel? [r/s/c]: ").lower().strip()
                        if choice in ['r', 'replace']:
                            print("Removing existing PhysiCell installation...")
                            shutil.rmtree(physicell_dir)
                            break
                        elif choice in ['s', 'skip']:
                            print("Skipping download. Using existing PhysiCell installation.")
                            return True
                        elif choice in ['c', 'cancel']:
                            print("Download cancelled by user.")
                            return False
                        else:
                            print("Please enter 'r' for replace, 's' for skip, or 'c' for cancel.")
                else:
                    print("Skipping download. Use force_download=True to override.")
                    return True
            else:
                print("Force download enabled. Removing existing installation...")
                shutil.rmtree(physicell_dir)
        
        # Navigate to the target directory (change working directory)
        original_cwd = os.getcwd()
        os.chdir(target_path)
        
        print(f"Downloading PhysiCell to {target_path.resolve()}...")
        
        # Download the PhysiCell zip file
        url = "https://github.com/MathCancer/PhysiCell/archive/refs/heads/master.zip"
        zip_filename = "PhysiCell.zip"
        
        urllib.request.urlretrieve(url, zip_filename)
        print("Download completed.")
        
        # Unzip the downloaded file - this will create a new directory called PhysiCell-master
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction completed.")
        
        # Remove the zip file
        os.remove(zip_filename)
        print("Temporary zip file removed.")
        
        print("PhysiCell has been downloaded into the examples folder.")
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        return True
        
    except Exception as e:
        print(f"Error downloading PhysiCell: {e}")
        # Restore original working directory in case of error
        try:
            os.chdir(original_cwd)
        except:
            pass
        return False

def get_executable_name_from_makefile(model_path):
    """
    Extract the executable name from the Makefile in the model directory.
    
    Args:
        model_path (str): Path to the model directory containing the Makefile
    
    Returns:
        str or None: The executable name if found, None otherwise
    """
    makefile_path = os.path.join(model_path, "Makefile")
    
    if not os.path.exists(makefile_path):
        print(f"Makefile not found at: {makefile_path}")
        return None
    
    try:
        with open(makefile_path, 'r') as f:
            content = f.read()
        
        # Look for PROGRAM_NAME := <name> pattern
        match = re.search(r'^PROGRAM_NAME\s*:=\s*(\w+)', content, re.MULTILINE)
        if match:
            executable_name = match.group(1)
            print(f"Found executable name in Makefile: {executable_name}")
            return executable_name
        else:
            print("Could not find PROGRAM_NAME in Makefile")
            return None
            
    except Exception as e:
        print(f"Error reading Makefile: {e}")
        return None


def compile_physicell(pc_path, model_path, executable_path=None, force_compile=False):
    """
    Compile PhysiCell and return the executable name from the Makefile.
    Actual compilation steps would depend on the user's environment and requirements.
    Args:
        pc_path (str): Path to the PhysiCell directory
        model_path (str): Path to the model directory containing the Makefile
        executable_path (str, optional): Path to store the compiled executable. If None, uses model_path.
        force_compile (bool): If True, forces recompilation even if executable exists
    """
    if not force_compile and os.path.exists(executable_path):
        print(f"Executable already exists in {executable_path}. Skipping compilation.")
        return
    
    executable2move = None
    # If executable_path is not provided or is executable path is equal to model_path, move it to model_path
    if executable_path is None or pathlib.Path(executable_path).parent == pathlib.Path(model_path):
        # Get executable name from Makefile before compilation
        executable2move = get_executable_name_from_makefile(model_path)
        executable_path = os.path.join(model_path, executable2move)

    print(f"Compiling PhysiCell model: {executable_path}...")
    # Check if pc_path exists
    if not os.path.exists(pc_path):
        print(f"PhysiCell path does not exist ... Downloading it now.")
        root_dir = os.path.dirname(pc_path)
        get_physicell(target_dir=root_dir)
    
    original_cwd = os.getcwd()
    
    # Calculate the relative path from PhysiCell directory to model directory
    # Convert both paths to absolute paths BEFORE changing working directory
    abs_pc_path = os.path.abspath(pc_path)
    abs_model_path = os.path.abspath(model_path)
    
    # Calculate relative path from PhysiCell directory to model directory
    rel_model_path = os.path.relpath(abs_model_path, abs_pc_path)
    
    # Use relative paths for privacy (avoid exposing personal directory structure)
    rel_pc_path = os.path.relpath(abs_pc_path, original_cwd)
    rel_model_path_display = os.path.relpath(abs_model_path, original_cwd)
    
    print(f"PhysiCell path: {rel_pc_path}")
    print(f"Model path: {rel_model_path_display}")
    print(f"Relative path for PROJ: {rel_model_path}")
    
    try:
        # Change to the PhysiCell directory
        os.chdir(pc_path)
        
        # Compile the model using the calculated relative path
        result = os.system(f"make load PROJ=../{rel_model_path} && make")
        
        if result != 0:
            raise ValueError(f"Make command failed with return code: {result}")
        
        print("Compilation completed.")
        
        # Move executable to model directory if it exists
        if executable2move:
            # Use the relative path we calculated earlier
            target_path = os.path.join(rel_model_path, executable2move)
            shutil.move(executable2move, target_path)
            print(f"Moved executable {executable2move} to {target_path}")
        
    except Exception as e:
        raise ValueError(f"Error during compilation: {e}")

    finally:
        # Restore original working directory
        os.chdir(original_cwd)