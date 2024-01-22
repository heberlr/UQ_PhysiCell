import pathlib, os, sys
from shutil import copyfile
import xml.etree.ElementTree as ET
import numpy as np
import random
import configparser # read config *.ini file
import ast # string to literal
    
class PhysiCell_Model:
    def __init__(self, fileName, keyModel):
        configFile = configparser.ConfigParser()
        with open(fileName) as fd:
            configFile.read_file(fd)
        self.projName = configFile[keyModel]['projName']
        self.executable = configFile[keyModel]['executable']
        
        self.configFile_ref = configFile[keyModel]['configFile_ref']
        self.configFile_name = configFile[keyModel]['configFile_name'] # config files structure
        self.configFile_folder = configFile[keyModel].get("configFile_folder", fallback=None) # folder to storage the config files (.xmls) - if it is none then uses outputs_folder

        # Parameters for XML files
        self.outputs_folder = configFile[keyModel]['outputs_folder'] # folder to storage the output folders
        self.outputs_folder_name = configFile[keyModel]['outputs_folder_name'] # prefix of output folders
        self.omp_num_threads = configFile[keyModel]['omp_num_threads'] # number of threads omp for PhysiCell simulation
        self.numReplicates = int(configFile[keyModel]['numReplicates']) # number of replicates for each simualtion

        # dictionary with parameters to change in the xml, parameters == None will be fill in with self.parameterSamples
        self.parameters = ast.literal_eval(configFile[keyModel]['parameters']) # read dictionary of parameters, parameters that will change is defined as None, it will preserves the order of parameters to change. 
        self.keys_variable_params = [k for k, v in self.parameters.items() if not v ] # list with the order of parameters to change. The parameters that will change is None.
        self.parameterSamples = np.load(configFile[keyModel]['parametersSamplesFile']) # parameters samples numpy array 2D [sample_idx, parameter_idx]
        if( len(self.keys_variable_params) != self.parameterSamples.shape[1]):
            sys.exit(f"Error: number of parameters defined 'None' in {fileName} = {len(self.keys_variable_params)} is different of samples from {configFile[keyModel]['parametersSamplesFile']} = {self.parameterSamples.shape[1]}.")

    def get_configFilePath(self,sampleID, replicateID):
        if (self.configFile_folder): 
            os.makedirs(os.path.dirname(self.configFile_folder), exist_ok=True)
            return self.configFile_folder+self.configFile_name%(sampleID,replicateID)
        else:
            folder = self.get_outputPath(sampleID, replicateID)
            return folder+self.configFile_name%(sampleID,replicateID)
    def get_outputPath(self,sampleID, replicateID):
        folder = self.outputs_folder+self.outputs_folder_name%(sampleID,replicateID)
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        return folder
    def info(self):
        print(f"""
        Project name: {self.projName} 
        Executable: {self.executable}
        Config. file of reference: {self.configFile_ref} 
        Config. file names: {self.configFile_name}
        Folder to save config. files: {self.configFile_ref} 
        Folder to save output folders: {self.outputs_folder}
        Name of output folders: {self.outputs_folder_name}
        Number of omp threads for each simulation: {self.omp_num_threads}
        Number of parameters for sampling: {self.parameterSamples.shape[1]} 
        Number of samples: {self.parameterSamples.shape[0]} 
        Number of replicates for each parameter set: {self.numReplicates} 
        Parameters: {self.keys_variable_params}
        """)
    def createXMLs(self): # Give a array with parameters samples generate the xml files for each simulation
        for sampleID in range(self.parameterSamples.shape[0]):
            for replicateID in range(self.numReplicates):
                ConfigFile = self.get_configFilePath(sampleID,replicateID)
                if (self.outputs_folder): self.parameters['.//save/folder'] = self.get_outputPath(sampleID, replicateID) # else save in folder of reference config file (util if there is a custom type of output)
                self.parameters['.//omp_num_threads'] = self.omp_num_threads # number of threads omp for PhysiCell simulation
                self.parameters['.//random_seed'] = random.randint(0,4294967295) # random seed for each simulation
                # update the values of parameter from None to the sampled
                for idx, param_key in enumerate(self.keys_variable_params): self.parameters[param_key] = self.parameterSamples[sampleID, idx]
                generate_xml_file(pathlib.Path(self.configFile_ref), pathlib.Path(ConfigFile), self.parameters)

# Give a xml input create xml file output with parameters changes (verify this function for multiple cell types)
def generate_xml_file(xml_file_in, xml_file_out, parameters):
    copyfile(xml_file_in,xml_file_out)
    tree = ET.parse(xml_file_out)
    xml_root = tree.getroot()
    # Loop in all parameters
    for key in parameters.keys():
        val = parameters[key]
        # print(key,val)
        elem = xml_root.findall(key)
        if (len(elem) != 1):
            sys.exit(f"""
                     Error: multiples occurrences or none found to this key: {key}, occurences: {[pos.text for pos in elem]}.
                     Key examples:
                     # key cell cycle example: ".//*[@name='CD8 Tcell']/phenotype/cycle/phase_transition_rates/rate[4]"
                     # key substrates example: ".//*[@name='TNF']/physical_parameter_set/diffusion_coefficient"
                     # key parameter example: ".//random_seed"
                    """)
        elem[0].text = str(val)
    tree.write(xml_file_out)
