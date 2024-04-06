import pathlib, os, sys, subprocess
import xml.etree.ElementTree as ET
import pandas as pd
import random
import configparser # read config *.ini file
import ast # string to literal
from shutil import copyfile, rmtree
import pcdl
    
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
                sys.exit(f"Error: parameters to be explored needs to be a list of [None, Name] and not {self.parameters[param_key]}! Check the file: {self.ModelfileName} and key {keyModel}.")

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
        Folder to save config. files: {self.configFile_folder} 
        Folder to save output folders: {self.outputs_folder}
        Name of output folders: {self.outputs_folder_name}
        Number of omp threads for each simulation: {self.omp_num_threads}
        Number of parameters for sampling: {len(self.keys_variable_params)}
        Parameters: { [self.parameters[param_key][1] for param_key in self.keys_variable_params] }
        Number of replicates for each parameter set: {self.numReplicates} 
        """)

    def createXMLs(self, parameters, SampleID, ReplicateID): # Give a array with parameters samples generate the xml file associated
        if( len(self.keys_variable_params) != parameters.shape[0]): # Check if parameter matrix numpy array 1D is compatible with .ini file
            sys.exit(f"Error: number of parameters defined 'None' in {self.ModelfileName} = {len(self.keys_variable_params)} is different of samples from parameters = {parameters.shape[0]}.")
        dic_parameters = self.parameters.copy() # copy of dictionary of parameters
        ConfigFile = self.get_configFilePath(SampleID,ReplicateID)
        if (self.outputs_folder): dic_parameters['.//save/folder'] = self.get_outputPath(SampleID, ReplicateID) # else save in folder of reference config file (util if there is a custom type of output)
        dic_parameters['.//omp_num_threads'] = self.omp_num_threads # number of threads omp for PhysiCell simulation
        dic_parameters['.//random_seed'] = random.randint(0,4294967295) # random seed for each simulation
        # update the values of parameter from None to the sampled
        for idx, param_key in enumerate(self.keys_variable_params): dic_parameters[param_key] = parameters[idx] # preserve the order
        generate_xml_file(pathlib.Path(self.configFile_ref), pathlib.Path(ConfigFile), dic_parameters)

    def RunModel(self, SampleID, ReplicateID, Parameters, RemoveConfigFile = True, SummaryFunction=None):
        # Generate XML file for this simulation
        self.createXMLs(Parameters, SampleID, ReplicateID)
        # XML path
        ConfigFile = self.get_configFilePath(SampleID, ReplicateID)
        # Write input for simulation & execute
        callingModel = [self.executable, ConfigFile]
        cache = subprocess.run( callingModel,universal_newlines=True, capture_output=True)
        if ( cache.returncode != 0):
            print(f"Error: model output error! Executable: {self.executable} ConfigFile {ConfigFile}. returned: \n{str(cache.returncode)}")
            print(cache.stdout[-200])
            return -1
        else:
            # remove config file XML
            if (RemoveConfigFile): os.remove( pathlib.Path(ConfigFile) )
            # write the stats in a file and remove the folder
            if (SummaryFunction):
                OutputFolder = self.get_outputPath(SampleID, ReplicateID)
                SummaryFile = self.outputs_folder+'SummaryFile_%06d_%02d.csv'%(SampleID,ReplicateID)
                ParamNames = [self.parameters[param_key][1] for param_key in self.keys_variable_params]
                dic_params = {ParamNames[i]: Parameters[i] for i in range(len(Parameters))}
                try: result_summary = SummaryFunction(OutputFolder,SummaryFile, dic_params,  SampleID, ReplicateID)
                except OSError as error: print(f"\t{error}\n\tError in SummaryFunction! (Sample: {SampleID} and Replicate: {ReplicateID}).")
                except SystemError as error: print(f"\t{error}\n\tError in SummaryFunction! (Sample: {SampleID} and Replicate: {ReplicateID}).")
            return 0

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

def summ_func(OutputFolder,SummaryFile, dic_params, SampleID, ReplicateID):
    mcds = pcdl.TimeStep('final.xml',OutputFolder, microenv=False, graph=False, settingxml=None, verbose=False)
    df_cell = mcds.get_cell_df() 
    live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
    dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( OutputFolder )
    df.to_csv(SummaryFile, sep='\t', encoding='utf-8')