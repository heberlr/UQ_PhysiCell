import pcdl
import pandas as pd
from shutil import rmtree

'''
A library of summary statistics to be applied to PhysiCell simulation output files.

Each method takes arguments
1) outputPath: a PhysiCell output directory 
2) summaryFile: a file to storage the summary of the PhysiCell simulation 
3) dic_params: a dictionary with the parameters used in the simulation
4) SampleID: a unique identifier for the sample
5) ReplicateID: a unique identifier for the replicate

Each method will return the appropriate data frame summary.
'''
# Final population of live and dead cells
def summ_func_FinalPopLiveDead(outputPath,summaryFile, dic_params, SampleID, ReplicateID):
    # read the last file
    mcds = pcdl.TimeStep('final.xml',outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    # dataframe of cells
    df_cell = mcds.get_cell_df() 
    # population stats live and dead cells
    live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
    dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
    # dataframe structure
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( outputPath )
    df.to_csv(summaryFile, sep='\t', encoding='utf-8')

# Population over time of live and dead cells
def summ_func_TimeSeriesPopLiveDead(outputPath,summaryFile, dic_params, SampleID, ReplicateID):
    mcds_ts = pcdl.TimeSeries(outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    for mcds in mcds_ts.get_mcds_list():
        df_cell = mcds.get_cell_df() 
        live_cells = len(df_cell[ (df_cell['dead'] == False) ] )
        dead_cells = len(df_cell[ (df_cell['dead'] == True) ] )
        data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'live_cells': live_cells, 'dead_cells': dead_cells, 'run_time_sec': mcds.get_runtime()}
        data_conc = {**data,**dic_params} # concatenate output data and parameters
        if ( mcds.get_time() == 0 ): df = pd.DataFrame([data_conc]) # create the dataframe
        else: df.loc[len(df)] = data_conc # append the dictionary to the dataframe
    # remove replicate output folder
    rmtree( outputPath )
    df.to_csv(summaryFile, sep='\t', encoding='utf-8')
