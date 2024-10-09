import pathlib, os, sys, subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import csv
import random
import configparser # read config *.ini file
import ast # string to literal
from shutil import copyfile
    
class PhysiCell_Model:
    def __init__(self, fileName, keyModel):
        self.ModelfileName = fileName
        configFile = configparser.ConfigParser()
        with open(fileName) as fd:
            configFile.read_file(fd)
        self.projName = configFile[keyModel]['projName']
        self.executable = configFile[keyModel]['executable']
        if (os.name == 'nt'): self.executable = self.executable.replace(os.altsep, os.sep)+'.exe' # change path according to windows


        self.configFile_ref = configFile[keyModel]['configFile_ref']
        self.configFile_name = configFile[keyModel]['configFile_name'] # config files structure
        self.configFile_folder = configFile[keyModel].get("configFile_folder", fallback=None) # folder to storage the config files (.xmls) - if it is none then uses outputs_folder

        # Parameters for XML files
        self.outputs_folder = configFile[keyModel]['outputs_folder'] # folder to storage the output folders
        self.outputs_folder_name = configFile[keyModel]['outputs_folder_name'] # prefix of output folders
        self.omp_num_threads = int(configFile[keyModel]['omp_num_threads']) # number of threads omp for PhysiCell simulation
        self.numReplicates = int(configFile[keyModel]['numReplicates']) # number of replicates for each simualtion

        # dictionary with parameters to change in the xml, parameters == None will be replace accordingly.
        self.parameters = ast.literal_eval(configFile[keyModel]['parameters']) # read dictionary of parameters, parameters that will change is defined as a list [None, Name] it will preserves the order of parameters to change. 
        self.keys_variable_params = [k for k, v in self.parameters.items() if (type(v) == list) ] # list with the order of parameters to change. The parameters that will change is a list [None, Name].
        for param_key in self.keys_variable_params: 
            if ( (self.parameters[param_key][0]) or (type(self.parameters[param_key][1]) != str) ):
                sys.exit(f"Error parameters format: Parameters to be explored needs to be a list of [None, Name] and not {self.parameters[param_key]}! Check the file: {self.ModelfileName} and key {keyModel}.")
            check_parameter_in_xml(self.configFile_ref, param_key)
        
        # dictionary with parameters of RULES to change the rules, parameters == None will be replace accordingly.
        self.parameters_rules = configFile[keyModel].get("parameters_rules", fallback=None)
        if ( self.parameters_rules ):
            self.rulesFile_ref = configFile[keyModel]['rulesFile_ref']
            self.rulesFile_name = configFile[keyModel]['rulesFile_name']
            self.rules = get_rules(self.rulesFile_ref)
            self.parameters_rules = ast.literal_eval(self.parameters_rules) # convert string to dictionary
            self.keys_variable_params_rules = [k for k, v in self.parameters_rules.items() if (type(v) == list) ] # list with the order of parameters to change. The parameters that will change is a list [None, Name].
            for param_key in self.keys_variable_params_rules: 
                if ( (self.parameters_rules[param_key][0]) or (type(self.parameters_rules[param_key][1]) != str) ):
                    sys.exit(f"Error rules parameter format: Rules parameters to be explored needs to be a list of [None, Name] and not {self.parameters_rules[param_key]}! Check the file: {self.ModelfileName} and key {keyModel}.")
                index_rule = get_rule_index_in_csv(self.rules, param_key)
    
    def get_configFilePath(self,sampleID, replicateID):
        if (self.configFile_folder): # Storage the new files in confidFile_folder
            os.makedirs(os.path.dirname(self.configFile_folder), exist_ok=True)
            return self.configFile_folder+self.configFile_name%(sampleID,replicateID)
        else: # If there is no confidFile_folder storage in output folder
            folder = self.get_outputPath(sampleID, replicateID)
            return folder+self.configFile_name%(sampleID,replicateID)
        
    def get_outputPath(self,sampleID, replicateID):
        folder = self.outputs_folder+self.outputs_folder_name%(sampleID,replicateID)
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        return folder
    
    def get_rulesFilePath(self,sampleID, replicateID):
        if (self.configFile_folder): # Storage the new files in confidFile_folder
            os.makedirs(os.path.dirname(self.configFile_folder), exist_ok=True)
            return self.configFile_folder, self.rulesFile_name%(sampleID,replicateID)
        else: # If there is no confidFile_folder storage in output folder
            folder = self.get_outputPath(sampleID, replicateID)
            return folder, self.rulesFile_name%(sampleID,replicateID)
    
    def info(self):
        if (self.parameters_rules): 
            NumParRules = len(self.keys_variable_params_rules)
            ParRules = [self.parameters_rules[param_key][1] for param_key in self.keys_variable_params_rules]
        else: NumParRules = 0; ParRules = []
        print(f"""
        Project name: {self.projName} 
        Executable: {self.executable}
        Config. file of reference: {self.configFile_ref} 
        Config. file names: {self.configFile_name}
        Folder to save config. files: {self.configFile_folder} 
        Folder to save output folders: {self.outputs_folder}
        Name of output folders: {self.outputs_folder_name}
        Number of omp threads for each simulation: {self.omp_num_threads}
        Number of parameters for sampling in XML: {len(self.keys_variable_params)}
        Parameters in XML: { [self.parameters[param_key][1] for param_key in self.keys_variable_params] }
        Number of parameters for sampling in RULES: {NumParRules}
        Parameters in RULES: { ParRules }
        Number of replicates for each parameter set: {self.numReplicates} 
        """)

    def createXMLs(self, parameters_input, SampleID, ReplicateID, parameters_rules_input = None ): # Give a array with parameters samples generate the xml file associated
        if( len(self.keys_variable_params) != parameters_input.shape[0]): # Check if parameter matrix numpy array 1D is compatible with .ini file
            if (self.parameters_rules): # sum of parameters in rules and parameters in xml
                if ((len(self.keys_variable_params_rules) + len(self.keys_variable_params)) != parameters_input.shape[0]):
                    sys.exit(f"Error: number of parameters defined 'None' in {self.ModelfileName} = {len(self.keys_variable_params)} is different of samples from parameters = {parameters_input.shape[0]-len(self.keys_variable_params_rules)}.")
            else: # only parameters in xml
                sys.exit(f"Error: number of parameters defined 'None' in {self.ModelfileName} = {len(self.keys_variable_params)} is different of samples from parameters = {parameters_input.shape[0]}.")
        dic_parameters = self.parameters.copy() # copy of dictionary of parameters
        # Config file (.xml)
        ConfigFile = self.get_configFilePath(SampleID,ReplicateID)
        # Rule file (.csv)
        if (self.parameters_rules): # If there is changes in parameter of rules
            dic_parameters['.//cell_rules/rulesets/ruleset/folder'], dic_parameters['.//cell_rules/rulesets/ruleset/filename'] = self.get_rulesFilePath(SampleID, ReplicateID)
            if( len(self.keys_variable_params_rules)+len(self.keys_variable_params) != parameters_rules_input.shape[0]+parameters_input.shape[0]): # Check if parameter rule matrix numpy array 1D is compatible with .ini file
                sys.exit(f"Error: number of parameters rules defined 'None' in {self.ModelfileName} = {len(self.keys_variable_params_rules)} is different of samples from parameters_rules = {parameters_rules_input.shape[0]}.")
            dic_parameters_rules = {}
            for idx, param_key in enumerate(self.keys_variable_params_rules): 
                single_param_rule = param_key.split(",")[-1] # last item is the parameter of rule
                dic_parameters_rules[idx] = [ parameters_rules_input[idx], param_key, single_param_rule ] # preserve the order id_rule:[value, rule key, parameter key]                                
            csvFile_out = dic_parameters['.//cell_rules/rulesets/ruleset/folder']+dic_parameters['.//cell_rules/rulesets/ruleset/filename']           
            generate_csv_file(self.rules, csvFile_out, dic_parameters_rules)
        # Output folder (.mat, .xml, .svg)
        if (self.outputs_folder): dic_parameters['.//save/folder'] = self.get_outputPath(SampleID, ReplicateID) # else save in folder of reference config file (util if there is a custom type of output)
        dic_parameters['.//omp_num_threads'] = self.omp_num_threads # number of threads omp for PhysiCell simulation
        dic_parameters['.//user_parameters/random_seed'] = random.randint(0,4294967295) # random seed for each simulation
        # update the values of parameter from None to the sampled
        for idx, param_key in enumerate(self.keys_variable_params): dic_parameters[param_key] = parameters_input[idx] # preserve the order
        generate_xml_file(pathlib.Path(self.configFile_ref), pathlib.Path(ConfigFile), dic_parameters)

    def RunModel(self, SampleID, ReplicateID, Parameters, ParametersRules = None, RemoveConfigFile = True, SummaryFunction=None):
        try:
            # Generate XML file for this simulation
            self.createXMLs(Parameters, SampleID, ReplicateID, parameters_rules_input=ParametersRules)
            
            # XML path
            ConfigFile = self.get_configFilePath(SampleID, ReplicateID)
            
            # Write input for simulation & execute
            callingModel = [self.executable, ConfigFile]
            cache = subprocess.run( callingModel,universal_newlines=True, capture_output=True)
            
            if ( cache.returncode != 0):
                print(f"Error: model output error! Executable: {self.executable} ConfigFile {ConfigFile}. returned: \n{str(cache.returncode)}")
                print(cache.stdout[-200])
                return -1
           
            # Remove config file XML and rule file CSV
            if (RemoveConfigFile):
                try:
                    os.remove( pathlib.Path(ConfigFile) )
                    if (self.parameters_rules):
                        folderRule, filenameRule = self.get_rulesFilePath(SampleID, ReplicateID)
                        os.remove( pathlib.Path(folderRule+filenameRule) )
                except OSError as e:
                    print(f"Error removing files: {e}")
            
            # Write the stats in a file and remove the folder
            if (SummaryFunction):
                OutputFolder = self.get_outputPath(SampleID, ReplicateID)
                SummaryFile = self.outputs_folder+'SummaryFile_%06d_%02d.csv'%(SampleID,ReplicateID)
                ParamNames = [self.parameters[param_key][1] for param_key in self.keys_variable_params]
                dic_params = {ParamNames[i]: Parameters[i] for i in range(len(Parameters))}
                
                if (self.parameters_rules):
                    ParamNamesRules = [self.parameters_rules[param_key][1] for param_key in self.keys_variable_params_rules]
                    for i in range(len(ParametersRules)): dic_params[ParamNamesRules[i]] = ParametersRules[i]
                try: 
                    result_summary = SummaryFunction(OutputFolder,SummaryFile, dic_params,  SampleID, ReplicateID)
                except (OSError, SystemError) as error: 
                    print(f"\t{error}\n\tError in SummaryFunction! (Sample: {SampleID} and Replicate: {ReplicateID}).")
               
            return 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return -1

def get_xml_element_value(xml_root, key):
    elem = xml_root.findall(key)
    if (len(elem) != 1):
        raise ValueError(f"""
                Multiples occurrences or none found to this key: {key}, occurences: {[pos.text for pos in elem]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    return elem[0].text

def set_xml_element_value(xml_root, key, val):
    elem = xml_root.findall(key)
    if (len(elem) != 1):
        raise ValueError(f"""
                Multiples occurrences or none found to this key: {key}, occurences: {[pos.text for pos in elem]}.
                Key examples:
                # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                # key parameter example: ".//random_seed"
                """)
    else: elem[0].text = str(val)

# Give a xml input create xml file output with parameters changes (verify this function for multiple cell types)
def generate_xml_file(xml_file_in, xml_file_out, dic_parameters):
    copyfile(xml_file_in,xml_file_out)
    tree = ET.parse(xml_file_out)
    xml_root = tree.getroot()
    # Loop in all parameters
    for key in dic_parameters.keys():
        val = dic_parameters[key]
        try: set_xml_element_value(xml_root, key, val)
        except ValueError as e:
            print(f"Error in Parameters definition: {e}")
            sys.exit(1)
        # print(key,val)
    tree.write(xml_file_out)

def check_parameter_in_xml(xml_file_in, key_parameter):
    tree = ET.parse(xml_file_in)
    xml_root = tree.getroot()
    try: text_elem = get_xml_element_value(xml_root, key_parameter)
    except ValueError as e:
        print(f"Error in Parameters definition: {e}")
        sys.exit(1)

def get_rules(filename):
    default_rules = []
    try: 
        with open(filename, newline='') as csvfile:
            fieldNames = ['cell_type', 'signal', 'direction', 'behavior', 'saturation', 'half_max', 'hill_power', 'dead']
            reader = csv.DictReader(csvfile, fieldnames=fieldNames, delimiter=',')
            for row in reader:
                default_rules.append(row)
        return default_rules
    except FileNotFoundError:
        print(f"Error! File {filename} not found!")
        sys.exit(1)

def get_rule_index_in_csv(rules, key_rules):
    # rule: cell_type, signal, direction, behavior
    # parameters: saturation, half_max, hill_power, dead
    try: cell_type, signal, direction, behavior, parameter = key_rules.split(",")
    except ValueError:
        print("Error in rule format: Provide 'cell_type,signal,direction,behavior,parameter' where parameter can be 'saturation', 'half_max', 'hill_power', or 'dead'.")
        sys.exit(1)
    for idx, rule in enumerate(rules):
        if ( (cell_type == rule['cell_type']) and (signal == rule['signal']) and 
            (direction == rule['direction']) and (behavior == rule['behavior']) and
            (parameter in ['saturation', 'half_max', 'hill_power', 'dead']) ):
            return idx
    print(f"Error! Rule {cell_type},{signal},{direction},{behavior} not found or parameter {parameter} does not exist!")
    sys.exit(1)

def generate_csv_file(rules, csv_file_out, dic_parameters_rules):
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
        print(f"Error generating csv file.")
        sys.exit(1)