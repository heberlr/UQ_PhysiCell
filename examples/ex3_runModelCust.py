from uq_physicell import PhysiCell_Model
import numpy as np
import pandas as pd
from shutil import rmtree
import pcdl
from typing import Union

def custom_summary_func(OutputFolder:str,SummaryFile:Union[str,None], dic_params:dict, SampleID:int, ReplicateID:int)-> Union[pd.DataFrame,None]:
    mcds_ts = pcdl.TimeSeries(OutputFolder, microenv=False, graph=False, settingxml=None, verbose=False)
    for mcds in mcds_ts.get_mcds_list():
        df_cell = mcds.get_cell_df()
        live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
        dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
        data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
        data_conc = {**data,**dic_params} # concatenate output data and parameters
        if ( mcds.get_time() == 0 ): df = pd.DataFrame([data_conc]) # create the dataframe
        else: df.loc[len(df)] = data_conc # append the dictionary to the dataframe
   # remove replicate output folder
    rmtree( OutputFolder )
    if (SummaryFile):
        df.to_csv(SummaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

if __name__ == '__main__':
    fileName = "examples/SampleModel.ini"
    key_model = "physicell_model_2"

    # Create the structure of model exploration
    print("Generate the model exploration structure")
    PhysiCellModel = PhysiCell_Model(fileName, key_model)
    # Print information of exploration
    print("First simulation using the custom summary function custom_summary_func.")
    PhysiCellModel.info()
    PhysiCellModel.RunModel(SampleID=0, ReplicateID=0,Parameters=np.array([0.75,0.5]), SummaryFunction=custom_summary_func)
    print("Second simulation using the custom summary function, difference omp_threads, and does not generate the summary file.")
    # Activate verbose mode
    PhysiCellModel.verbose = True
    # Change the number of threads
    PhysiCellModel.omp_num_threads = 4
    # Turn off summary file generation - return a data frame
    PhysiCellModel.output_summary_Path = None
    PhysiCellModel.info()
    df = PhysiCellModel.RunModel(SampleID=1, ReplicateID=0,Parameters=np.array([0.80,0.55]), SummaryFunction=custom_summary_func)
    print(df.head())

