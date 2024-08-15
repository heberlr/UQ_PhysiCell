from uq_physicell.uq_physicell import PhysiCell_Model
from uq_physicell.sumstats import summ_func_FinalPopLiveDead
import numpy as np

if __name__ == '__main__':
    fileName = "examples/SampleModel.ini"
    key_model = "physicell_model_2"
    
    # Create the structure of model exploration
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    
    # Sample parameters 
    Parameters_dic = {1: np.array([0.75,0.5]), 2: np.array([0.80,0.55])}
    print(f"Total number of simulations: {len(Parameters_dic)*PhysiCellModel.numReplicates}")

    for sampleID, par_value in Parameters_dic.items():
        for replicateID in np.arange(PhysiCellModel.numReplicates):
            print(', Sample: ', sampleID,', Replicate: ', replicateID)
            PhysiCellModel.RunModel(SampleID=sampleID, ReplicateID=replicateID,Parameters=par_value,SummaryFunction=summ_func_FinalPopLiveDead)
