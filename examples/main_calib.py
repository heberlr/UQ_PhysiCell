from uq_physicell.uq_physicell import PhysiCell_Model
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import os, shutil
import pcdl

fileName = "SampleModel.ini"
key_model = "physicell_model_2"

# Create the structure of model exploration
PhysiCellModel = PhysiCell_Model(fileName, key_model)
NumCoresPYMC = 4 # If NumCoresPYMC = 1 is serial, else in parallel

def sum_stat(output_folders):
    dic_output = {'liveCells': [], 'deadCells': []}
    for outputFolder in output_folders:
        mcds = pcdl.TimeStep('output00000004.xml',outputFolder, microenv=False, graph=False, settingxml=None, verbose=False)
        df_cell = mcds.get_cell_df()
        dic_output['liveCells'].append(len(df_cell[ (df_cell['dead'] == False) ] ))
        dic_output['deadCells'].append(len(df_cell[ (df_cell['dead'] == True) ] ))
        shutil.rmtree( outputFolder ) # remove output folder
    # print(dic_output)
    return np.array([ np.average(dic_output['liveCells']), np.average(dic_output['deadCells'])]) # average of replicates


def RunPhysiCellModel(rng, *args, size=None):
    # Set the parameters in PhysiCell class
    SampleID = os.getpid() # process number
    Parameters =  [args[:-1]] # Exclude the last arg that is size
    # print(f" Parameters: {Parameters}, Process ID: {SampleID}") 
    # Run the replicates
    output_folders = []
    for replicateID in range(PhysiCellModel.numReplicates):
        PhysiCellModel.RunModel(SampleID, replicateID, Parameters, RemoveConfigFile=True)
        output_folders.append(PhysiCellModel.get_outputPath(SampleID, replicateID))
    # Return the sum stats of replicates
    # print(f"... running PhysiCell")
    return sum_stat(output_folders)

def Run_CalibrationSMC(observedData, FileNameInfData):
    with pm.Model() as model_lv:
        par1 = pm.HalfNormal("viral_replication_rate", 0.15) # Positive values
        par2 = pm.Uniform("min_virion_count", lower = 0, upper = 100) # Positive values

        sim = pm.Simulator("sim", RunPhysiCellModel, params=[par1, par2], epsilon=100, observed=observedData)

        idata_lv = pm.sample_smc(draws=10, chains=3, cores=NumCoresPYMC, idata_kwargs={'log_likelihood':True} ) # return inference data (return_inferencedata=True is default pymc3 > 4.0)
        idata_lv.to_netcdf(FileNameInfData) # save the inferencedata

def Plot_Calibration(FileNameInfData):
    idata_lv = az.from_netcdf(FileNameInfData)
    # print(idata_lv['posterior'], idata_lv['sample_stats'], idata_lv['observed_data'])
    # print( f"""Chain: {idata_lv['sample_stats']['log_marginal_likelihood']['chain']})
    #       Draw: {idata_lv['sample_stats']['log_marginal_likelihood']['draw']}""")
    # print( idata_lv['sample_stats']['beta'])
    # print( idata_lv['sample_stats']['accept_rate'])
    az.plot_trace(idata_lv, kind="rank_vlines")
    az.plot_posterior(idata_lv)
    plt.show()

if __name__ == '__main__':
    print(f"Running on PyMC v{pm.__version__}")
    # call the rng number + parameters
    # PhysiCell_model(0, 10, 100 ) # Test of function

    with pm.Model() as model_lv:
        par1 = pm.HalfNormal("par1", 1.0)

        sim = pm.Simulator("sim", PhysiCell_model, params=[par1], epsilon=0.1, observed=10)

        idata_lv = pm.sample_smc(draws=1000, chains=4, cores=4)
        
        az.plot_trace(idata_lv, kind="rank_vlines")
        az.plot_posterior(idata_lv)
        plt.show()