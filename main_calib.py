from PhysiCellModel import PhysiCell_Model
from HPC_exploration import create_JOB
from HPC_exploration import model
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pathlib, os, shutil
import pcdl

fileName = "SampleModel.ini"
key_model = "physicell_model_2"
key_HPC_params = "hpc_parameters"
# Create the structure of model exploration
PhysiCellModel1 = PhysiCell_Model(fileName, key_model)
NumCoresPYMC = 2 # If NumCoresPYMC = 1 is serial, else in parallel

def sum_stat(output_folders):
    return np.array([999.0, 11])
    dic_output = {'liveCells': [], 'deadCells': []}
    for outputFolder in output_folders:
        mcds = pcdl.TimeStep('output00000048.xml',outputFolder, microenv=False, graph=False, settingxml=None)
        df_cell = mcds.get_cell_df()
        dic_output['liveCells'].append(len(df_cell[ (df_cell['dead'] == False) ] ))
        dic_output['deadCells'].append(len(df_cell[ (df_cell['dead'] == True) ] ))
        shutil.rmtree( outputFolder ) # remove output folder
    # print(dic_output)
    return np.array([ np.average(dic_output['liveCells']), np.average(dic_output['deadCells'])]) # average of replicates


def PhysiCell_model(rng, *args, size=None):
    # Set the parameters in PhysiCell class
    SampleID = os.getpid() # process number
    # print(f" Parameters: {np.array([args[:-1]])}, Process ID: {SampleID}") 
    PhysiCellModel1.set_parameters( np.array([args[:-1]]), [SampleID] ) # Exclude the last arg that is size - SampleID needs to be a list
    # Generate XML files for this simulation (all replicates)
    PhysiCellModel1.createXMLs()

    # Run the replicates
    output_folders = []
    for replicateID in range(PhysiCellModel1.numReplicates):
        # model(PhysiCellModel1.get_configFilePath(SampleID, replicateID), PhysiCellModel1.executable)
        os.remove( pathlib.Path(PhysiCellModel1.get_configFilePath(SampleID, replicateID)) ) # remove config file XML
        output_folders.append(PhysiCellModel1.get_outputPath(SampleID, replicateID))

    # Return the sum stats of replicates
    # print(f"... running PhysiCell")
    return sum_stat(output_folders)


if __name__ == '__main__':
    print(f"Running on PyMC v{pm.__version__}")
    # To load data: 
    idata_lv = az.from_netcdf("InfData_.nc")
    print(idata_lv['posterior'], idata_lv['sample_stats'], idata_lv['observed_data'])
    print( f"Chain: {idata_lv['sample_stats']['log_marginal_likelihood']['chain']}, Draw: {idata_lv['sample_stats']['log_marginal_likelihood']['draw']}")
    # print( idata_lv['sample_stats']['beta'])
    # print( idata_lv['sample_stats']['accept_rate'])
    exit(0)
    data_avg_live_dead = np.array([1000, 10]) # observational data
    # call the rng number + parameters
    # PhysiCell_model(0, 1.2, 2.1, 1) # Test of function
    with pm.Model() as model_lv:
        par1 = pm.HalfNormal("viral_replication_rate", 1.0) # Positive values
        par2 = pm.HalfNormal("min_virion_count", 5.0) # Positive values

        sim = pm.Simulator("sim", PhysiCell_model, params=[par1, par2], epsilon=100, observed=data_avg_live_dead)

        idata_lv = pm.sample_smc(draws=1000, chains=2, cores=NumCoresPYMC, idata_kwargs={'log_likelihood':True} ) # return inference data (return_inferencedata=True is default pymc3 > 4.0)
        idata_lv.to_netcdf("InfData_.nc") # save the inferencedata

        az.plot_trace(idata_lv, kind="rank_vlines")
        az.plot_posterior(idata_lv)
        plt.show()