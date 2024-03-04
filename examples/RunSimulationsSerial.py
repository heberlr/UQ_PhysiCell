from PhysiCellModel import PhysiCell_Model
import numpy as np
from HPC_exploration import model

if __name__ == '__main__':
    fileName = "SampleModel.ini"
    key_model = "physicell_model_2"
    key_HPC_params = "hpc_parameters"
    
    # Create the structure of model exploration
    PhysiCellModel1 = PhysiCell_Model(fileName, key_model)
    
    # Sample parameters 
    Parameters= np.array([[0, 1],[2, 3]])
    SamplesID= np.arange(Parameters.shape[0])
    PhysiCellModel1.set_parameters( Parameters, SamplesID )

    # Print information of exploration
    PhysiCellModel1.info()

    # Generate XML files for these parameters and replicates
    PhysiCellModel1.createXMLs()

    NumSimulations = Parameters.shape[0]*PhysiCellModel1.numReplicates
    print(f"Total number of simulations: {NumSimulations}")

    for sampleID in SamplesID:
        for replicateID in np.arange(PhysiCellModel1.numReplicates):
            print(', Sample: ', sampleID,', Replicate: ', replicateID)
            print(PhysiCellModel1.get_configFilePath(sampleID, replicateID), PhysiCellModel1.executable)
            model(PhysiCellModel1.get_configFilePath(sampleID, replicateID), PhysiCellModel1.executable)
