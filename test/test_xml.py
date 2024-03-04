from uq_physicell.uq_physicell import PhysiCell_Model
import numpy as np

if __name__ == '__main__':
    fileName = "test/SampleModel.ini"
    key_model = "physicell_model"
    key_model2 = "physicell_model_2"
    
    # Create the structure of model exploration
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    # Create a xml
    PhysiCellModel.createXMLs(parameters=np.array([0.75,0.5,50,5,0.5]), SampleID=1, ReplicateID=2)

    # Create the structure of model exploration
    PhysiCellModel2 = PhysiCell_Model(fileName, key_model2)
    # Create a xml
    PhysiCellModel2.createXMLs(parameters=np.array([0.25,0.0]), SampleID=2, ReplicateID=3)
