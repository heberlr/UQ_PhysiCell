from uq_physicell import PhysiCell_Model

if __name__ == '__main__':
    fileName = "examples/SampleModel.ini"
    key_model = "physicell_model"
    key_model2 = "physicell_model_2"

    # Create the structure of model exploration
    print("Generate the model exploration structure 1.")
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    # Print information of exploration
    PhysiCellModel.info()

    print("Generate the model exploration structure 2 with verbose.")
    # Create the structure of model exploration
    PhysiCellModel2 = PhysiCell_Model(fileName, key_model2, verbose=True)
    # Print information of exploration
    PhysiCellModel2.info()
