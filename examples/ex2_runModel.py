from uq_physicell import PhysiCell_Model
from uq_physicell import summ_func_FinalPopLiveDead
import numpy as np

if __name__ == '__main__':
    fileName = "examples/SampleModel.ini"
    key_model = "physicell_model_2"

    # Create the structure of model exploration
    print("Generate the model exploration structure")
    PhysiCellModel = PhysiCell_Model(fileName, key_model, verbose=True)
    # Print information of exploration
    print("First simulation using the summary function summ_func_FinalPopLiveDead.")
    PhysiCellModel.RunModel(SampleID=0, ReplicateID=0,Parameters=np.array([0.75,0.5]),SummaryFunction=summ_func_FinalPopLiveDead)
    print("Second simulation does not summarize the output and does not remove the configuration file.")
    PhysiCellModel.RunModel(SampleID=0, ReplicateID=1,Parameters=np.array([0.75,0.5]),RemoveConfigFile=False)
    print("Third simulation using the summary function summ_func_FinalPopLiveDead but does not generate the summary file.")
    # Turn off summary file generation - return a data frame
    PhysiCellModel.output_summary_Path = None
    df = PhysiCellModel.RunModel(SampleID=1, ReplicateID=0,Parameters=np.array([0.80,0.55]), SummaryFunction=summ_func_FinalPopLiveDead)
    print(df.head())

