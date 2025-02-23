import pathlib
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
from shutil import copyfile
import time
import copy

class PhysiCell_Model:
    def __init__(self, configFilePath: str, keyModel: str, verbose: bool = False) -> None:
        self.configFilePath = configFilePath
        self.verbose = verbose
        self._load_config(configFilePath, keyModel)
        self._load_xml_reference()
        self._load_rules_reference()

    def _load_config(self, configFilePath: str, keyModel: str) -> None:
        configFile = configparser.ConfigParser()
        if self.verbose:
            print(f"\t> Constructor PhysiCell_Model: {keyModel} at {configFilePath}...")
            print(f"\t\t>> Reading config file: {configFilePath} ...")
        with open(configFilePath) as fd:
            configFile.read_file(fd)
        # PhysiCell executable
        self.PC_executable = configFile[keyModel]['executable']
        # Path of XML file of reference
        self.XML_RefPath = configFile[keyModel]['configFile_ref']
        self.numReplicates = int(configFile[keyModel]['numReplicates'])

        #### Optional variables
        self.projName = configFile[keyModel].get('projName', fallback=keyModel) # project name
        self.XML_name = configFile[keyModel].get('configFile_name', fallback="config_S%06d_R%03d.xml") # config files structure
        self.input_folder = configFile[keyModel].get("configFile_folder", fallback="UQ_PC_InputFolder/") # folder to store input files (.xmls, .csv, and .txt)
        self.output_folder = configFile[keyModel].get('outputs_folder', fallback="UQ_PC_OutputFolder/") # folder to store the output folders
        self.outputs_folder_name = configFile[keyModel].get('outputs_folder_name', fallback="output_S%06d_R%03d/") # structure of output folders
        self.timeout = configFile[keyModel].get('timeout', fallback=60) # timeout for waiting to write files
        self.output_summary_Path = self.output_folder+'SummaryFile_%06d_%02d.csv'
        # Rules files and folder
        self.RULES_RefPath = configFile[keyModel].get('rulesFile_ref', fallback=None)
        self.RULES_name = configFile[keyModel].get('rulesFile_name', fallback="rules_S%06d_R%03d.csv")
        self.omp_num_threads = configFile[keyModel].get('omp_num_threads', fallback=None)
        self.XML_parameters = configFile[keyModel].get('parameters', fallback=dict())
        if self.XML_parameters:
            self.XML_parameters = ast.literal_eval(self.XML_parameters)
        self.XML_parameters_variable = {k: v[1] for k, v in self.XML_parameters.items() if isinstance(v, list)}
        self.XML_parameters_fixed = {k: v for k, v in self.XML_parameters.items() if not isinstance(v, list)}
        self.parameters_rules = configFile[keyModel].get('parameters_rules', fallback=dict())
        if self.parameters_rules:
            self.parameters_rules = ast.literal_eval(self.parameters_rules)
        self.parameters_rules_variable = {k: v[1] for k, v in self.parameters_rules.items() if isinstance(v, list)}
        self.parameters_rules_fixed = {k: v for k, v in self.parameters_rules.items() if not isinstance(v, list)}
        if self.verbose:
            print(f"\t\t>> Checking executable format ...")
        if os.name == 'nt':
            self.PC_executable = self.PC_executable.replace(os.altsep, os.sep) + '.exe'

    def _load_xml_reference(self) -> None:
        tree = ET.parse(self.XML_RefPath)
        self.xml_ref_root = tree.getroot()
        if self.verbose:
            print(f"\t\t>> Checking parameters in XML file ...")
        if not self.omp_num_threads: self.omp_num_threads = get_xml_element_value(self.xml_ref_root, './/parallel/omp_num_threads')
        for param_key in self.XML_parameters.keys():
            try:
                get_xml_element_value(self.xml_ref_root, param_key)
            except ValueError as e:
                raise ValueError(f"Error in parameters_xml. {e}")

    def _load_rules_reference(self) -> None:
        self.default_rules = get_rules(self.RULES_RefPath) if self.RULES_RefPath else None
        if self.verbose:
            print(f"\t\t>> Checking parameters in RULES file ...")
        for param_key in self.parameters_rules.keys():
            try:
                get_rule_index_in_csv(self.default_rules, param_key)
            except ValueError as e:
                raise ValueError(f"Error in parameters_rules. {e}")

    def get_XML_Path(self, sampleID: int, replicateID: int) -> str:
        filePath = self.input_folder + self.XML_name % (sampleID, replicateID)
        os.makedirs(os.path.dirname(self.input_folder), exist_ok=True)
        return filePath

    def get_output_Path(self, sampleID: int, replicateID: int) -> str:
        folderPath = self.output_folder + self.outputs_folder_name % (sampleID, replicateID)
        os.makedirs(os.path.dirname(folderPath), exist_ok=True)
        return folderPath

    def get_RULES_FileName(self, sampleID: int, replicateID: int) -> str:
        return self.RULES_name % (sampleID, replicateID)

    def info(self) -> None:
        print(f"""
        Project name: {self.projName} 
        Executable: {self.PC_executable}
        Number of replicates for each parameter set: {self.numReplicates} 
        Config. file of reference: {self.XML_RefPath}
        Folder to save config. files: {self.input_folder} 
        Folder to save output folders: {self.output_folder}
        Rules file of reference: {self.RULES_RefPath}
        Name of output folders: {self.outputs_folder_name}
        Number of omp threads for each simulation: {self.omp_num_threads}
        Number of parameters for sampling in XML: {len(self.XML_parameters_variable)}
        Parameters in XML: { [param_name for param_name in self.XML_parameters_variable.values()] }
        Number of parameters for sampling in RULES: {len(self.parameters_rules_variable)}
        Parameters in RULES: { [param_name for param_name in self.parameters_rules_variable.values()] }
        """)

    def copy(self):
        return copy.deepcopy(self)

    def RunModel(self, SampleID: int, ReplicateID: int, Parameters: Union[np.ndarray, dict] = dict(), ParametersRules: Union[np.ndarray, dict] = dict(), RemoveConfigFile: bool = True, SummaryFunction: Union[None, str] = None) -> Union[None, pd.DataFrame]:
        return RunModel(self, SampleID, ReplicateID, Parameters, ParametersRules, RemoveConfigFile, SummaryFunction)

def check_parameters_input(model: PhysiCell_Model, parameters_input_xml: Union[np.ndarray, dict], parameters_input_rules: Union[np.ndarray, dict]) -> None:
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

def setup_model_input(model: PhysiCell_Model, SampleID: int, ReplicateID: int, parameters_input: Union[np.ndarray, dict], parameters_rules_input: Union[np.ndarray, dict]) -> None:
    try:
        if model.verbose:
            print(f"\t\t\t>>> Checking parameters input ...")
        check_parameters_input(model, parameters_input, parameters_rules_input)
    except ValueError as e:
        raise ValueError(f"Error in parameters input!\n{e}")

    dic_xml_parameters = model.XML_parameters.copy()
    XMLFile = model.get_XML_Path(SampleID, ReplicateID)
    if model.parameters_rules:
        if model.verbose:
            print(f"\t\t\t>>> Setting up rules input ...")
        dic_xml_parameters['.//cell_rules/rulesets/ruleset/folder'] = model.input_folder
        dic_xml_parameters['.//cell_rules/rulesets/ruleset/filename'] = model.get_RULES_FileName(SampleID, ReplicateID)
        csvFile_out = model.input_folder + model.get_RULES_FileName(SampleID, ReplicateID)
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
            print(f"\t\t\t>>> Generating rules file {csvFile_out} ...")
        try:
            generate_csv_file(model.default_rules, csvFile_out, dic_rules_temp)
        except ValueError as e:
            raise ValueError(f"Error in generating rules file! {e}")

    if model.verbose:
        print(f"\t\t\t>>> Setting up XML input...")
    dic_xml_parameters['.//save/folder'] = model.get_output_Path(SampleID, ReplicateID)
    dic_xml_parameters['./parallel/omp_num_threads'] = model.omp_num_threads
    try:
        dic_xml_parameters['.//initial_conditions/cell_positions/folder'] = os.path.dirname(model.XML_RefPath)
    except ValueError:
        pass
    try:
        get_xml_element_value(model.xml_ref_root, './/options/random_seed')
        dic_xml_parameters['.//options/random_seed'] = "system_clock"
    except ValueError:
        try:
            get_xml_element_value(model.xml_ref_root, './/user_parameters/random_seed')
            dic_xml_parameters['.//user_parameters/random_seed'] = random.randint(0, 4294967295)
        except ValueError as e:
            raise ValueError(f"Error in setting random seed. {e}")
    for idx, param_key, param_name in zip(range(len(model.XML_parameters_variable)), model.XML_parameters_variable.keys(), model.XML_parameters_variable.values()):
        if isinstance(parameters_input, dict):
            dic_xml_parameters[param_key] = parameters_input[param_name]
        else:
            dic_xml_parameters[param_key] = parameters_input[idx]
    if model.verbose:
        print(f"\t\t\t>>> Generating XML file {XMLFile} ...")
    try:
        generate_xml_file(pathlib.Path(model.XML_RefPath), pathlib.Path(XMLFile), dic_xml_parameters, model.timeout)
    except ValueError as e:
        raise ValueError(f"Error in generating XML file! {e}")

def RunModel(model: PhysiCell_Model, SampleID: int, ReplicateID: int, Parameters: Union[np.ndarray, dict] = dict(), ParametersRules: Union[np.ndarray, dict] = dict(), RemoveConfigFile: bool = True, SummaryFunction: Union[None, str] = None) -> Union[None, pd.DataFrame]:
    if model.verbose:
        print(f"\t> RunModel - Sample:{SampleID}, Replicate: {ReplicateID}, Parameters XML: {Parameters}, Parameters rules: {ParametersRules}...")
    try:
        try:
            if model.verbose:
                print(f"\t\t>> Setting up model input ...")
            setup_model_input(model, SampleID, ReplicateID, Parameters, parameters_rules_input=ParametersRules)
        except ValueError as e:
            raise ValueError(f"Error in setup_model_input! (Sample: {SampleID} and Replicate: {ReplicateID}).\n{e}")

        XMLFile = model.get_XML_Path(SampleID, ReplicateID)

        if model.verbose:
            print(f"\t\t>> Running model ...")
        callingModel = [model.PC_executable, XMLFile]
        cache = subprocess.run(callingModel, universal_newlines=True, capture_output=True)
        if cache.returncode != 0:
            raise ValueError(f"""Error: model output error! 
            Executable: {model.PC_executable} XML File: {XMLFile}. returned: {str(cache.returncode)}
            Last 1000 characters of the PhysiCell output:
            {cache.stdout[-1000:]}""")

        if RemoveConfigFile:
            if model.verbose:
                print(f"\t\t>> Removing config files ...")
            try:
                os.remove(pathlib.Path(XMLFile))
                if model.parameters_rules:
                    filenameRule = model.get_RULES_FileName(SampleID, ReplicateID)
                    os.remove(pathlib.Path(model.input_folder + filenameRule))
            except OSError as e:
                print(f"Error removing files: {e}")

        if SummaryFunction:
            if model.verbose:
                print(f"\t\t>> Running summary function: {SummaryFunction} ...")
            OutputFolder = model.get_output_Path(SampleID, ReplicateID)
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
                if model.output_summary_Path:
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

def get_xml_element_value(xml_root: ET.Element, key: str) -> str:
    elem = xml_root.findall(key)
    if len(elem) != 1:
        raise ValueError(f""" Error in getting XML element!
                Multiples occurrences or none found to this key: {key}, occurrences: {[pos.text for pos in elem]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    return elem[0].text

def set_xml_element_value(xml_root: ET.Element, key: str, val: Union[str, int, float]) -> None:
    elem = xml_root.findall(key)
    if len(elem) != 1:
        raise ValueError(f""" Error in setting XML element!
                Multiples occurrences or none found to this key: {key}, occurrences: {[pos.text for pos in elem]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    else:
        elem[0].text = str(val)

def generate_xml_file(xml_file_in: str, xml_file_out: str, dic_parameters: dict, max_wait_time: float) -> None:
    copyfile(xml_file_in, xml_file_out)
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
            set_xml_element_value(xml_root, key, val)
        except ValueError as e:
            raise ValueError(f"Error in setting parameter {key} with value {val}. {e}")
        # print(key,val)
    tree.write(xml_file_out)

def get_rules(filename: str) -> list:
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

def get_rule_index_in_csv(rules: list, key_rules: str) -> int:
    try:
        cell_type, signal, direction, behavior, parameter = key_rules.split(",")
    except ValueError:
        raise ValueError("Error in rule format: Provide 'cell_type,signal,direction,behavior,parameter' where parameter can be 'saturation', 'half_max', 'hill_power', or 'dead'.")
    try:
        for idx, rule in enumerate(rules):
            if ( (cell_type == rule['cell_type']) and (signal == rule['signal']) and 
                (direction == rule['direction']) and (behavior == rule['behavior']) and
                (parameter in ['saturation', 'half_max', 'hill_power', 'dead']) ):
                return idx
    except ValueError:
        raise ValueError(f"Error! Rule {cell_type},{signal},{direction},{behavior} not found or parameter {parameter} does not exist!")

def generate_csv_file(rules: list, csv_file_out: str, dic_parameters_rules: dict) -> None:
    try:
        with open(csv_file_out, 'w', newline='') as csvfile:
            fieldNames = ['cell_type', 'signal', 'direction', 'behavior', 'saturation', 'half_max', 'hill_power', 'dead']
            writer = csv.DictWriter(csvfile, fieldnames=fieldNames, delimiter=',')
            rules_temp = rules.copy()
            for parameterID in dic_parameters_rules.keys():
                value, key_rule, param_name = dic_parameters_rules[parameterID]
                index_rule = get_rule_index_in_csv(rules, key_rule)
                rules_temp[index_rule][param_name] = value
            for rule in rules_temp: writer.writerow(rule)
    except:
        raise ValueError(f"Error generating csv file.")
