from uq_physicell.uq_physicell import PhysiCell_Model
from uq_physicell.sumstats import summ_func
import numpy as np

if __name__ == '__main__':
    fileName = "test/SampleModel.ini"
    key_model = "physicell_model_2"
    
    # Create the structure of model exploration
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    # Print information of exploration
    PhysiCellModel.RunModel(SampleID=0, ReplicateID=0,Parameters=np.array([0.75,0.5]),SummaryFunction=summ_func)
    PhysiCellModel.RunModel(SampleID=0, ReplicateID=1,Parameters=np.array([0.75,0.5]),RemoveConfigFile=False)
    PhysiCellModel.RunModel(SampleID=1, ReplicateID=0,Parameters=np.array([0.80,0.55]), SummaryFunction=summ_func)

