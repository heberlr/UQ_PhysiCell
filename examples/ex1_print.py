from uq_physicell.uq_physicell import PhysiCell_Model

if __name__ == '__main__':
    fileName = "examples/SampleModel.ini"
    key_model = "physicell_model"
    key_model2 = "physicell_model_2"
    
    # Create the structure of model exploration
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    # Print information of exploration
    PhysiCellModel.info()

    # Create the structure of model exploration
    PhysiCellModel2 = PhysiCell_Model(fileName, key_model2)
    # Print information of exploration
    PhysiCellModel2.info()
