from PhysiCellModel import PhysiCell_Model
from HPC_exploration import create_JOB

if __name__ == '__main__':
    fileName = "SampleModel.ini"
    key_model = "physicell_model"
    key_HPC_params = "hpc_parameters"
    
    # Create the structure of model exploration
    PhysiCellModel1 = PhysiCell_Model(fileName, key_model)
    
    # Print information of exploration
    PhysiCellModel1.info()
    
    # Generate XML files
    PhysiCellModel1.createXMLs() # Generate XML files for each simulation
    
    # Create bash script to run on cluster
    create_JOB(key_HPC_params, ID_Job = 0, args = [fileName, key_model, "sequential", '0', '2'])
    create_JOB(key_HPC_params, ID_Job = 1, args = [fileName, key_model, "samples", '1'])
    create_JOB(key_HPC_params, ID_Job = 2, args = [fileName, key_model, "individual", '0', '1', '1', '0'])
