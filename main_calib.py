from PhysiCellModel import PhysiCell_Model
from HPC_exploration import create_JOB
from HPC_exploration import model
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pathlib, os, shutil
from PhysiCell_sumstats import live_pop
import subprocess
import pcdl

fileName = "SampleModel.ini"
key_model = "pancreatic_drug_1_model"
key_HPC_params = "hpc_parameters"
# Create the structure of model exploration
PhysiCellModel1 = PhysiCell_Model(fileName, key_model)
NumCoresPYMC = 4 # If NumCoresPYMC = 1 is serial, else in parallel

def sum_stat(output_set):
    for output_config_pair in output_set:
        dic_output = live_pop(output_config_pair[0], output_config_pair[1], time = 'final')

    # print(np.average(dic_output['live_cells']))
    # print([np.average(dic_output['live_cells']) / np.std(dic_output['live_cells'])])

    return [np.average(dic_output)]


def PhysiCell_model(rng, *args, size=None):
    return np.random.normal(290, 1, size = 1)
    # print('called PhysiCell model')
    # Set the parameters in PhysiCell class
    SampleID = os.getpid() # process number
    # print(f" Parameters: {np.array([args[:-1]])}, Process ID: {SampleID}")
    PhysiCellModel1.set_parameters( np.array([args[:-1]]), [SampleID] ) # Exclude the last arg that is size - SampleID needs to be a list
    # print('Set PhysiCell parameters successfully')
    # Generate XML files for this simulation (all replicates)
    PhysiCellModel1.createXMLs()
    # print('Created PhysiCell XMLs successfully')
    # Run the replicates
    output_config_pairs = []
    for replicateID in range(PhysiCellModel1.numReplicates):
        model(PhysiCellModel1.get_configFilePath(SampleID, replicateID), PhysiCellModel1.executable)
        output_config_pairs.append([
            PhysiCellModel1.get_outputPath(SampleID, replicateID),
            PhysiCellModel1.get_configFilePath(SampleID, replicateID)
        ])
    # print(f' Initialized output pair array: {output_config_pairs}')
    # Return the sum stats of replicates
    return sum_stat(output_config_pairs)

if __name__ == '__main__':
    print(f"Running on PyMC v{pm.__version__}")
    # # To load data:
    # idata_lv = az.from_netcdf("InfData_.nc")
    # print(idata_lv['posterior'], idata_lv['sample_stats'], idata_lv['observed_data'])
    # print( f"Chain: {idata_lv['sample_stats']['log_marginal_likelihood']['chain']}, Draw: {idata_lv['sample_stats']['log_marginal_likelihood']['draw']}")
    # # print( idata_lv['sample_stats']['beta'])
    # # print( idata_lv['sample_stats']['accept_rate'])
    # # exit(0)
    obs = np.array([290]) # observational data
    # # call the rng number + parameters
    # PhysiCell_model(0, 1.2, 0.0001, 1) # Test of function
    with pm.Model() as model_lv:
        factor_max_apoptosis_drug1 = pm.Uniform("factor_max_apoptosis_drug1", lower = 0, upper = 200) # Positive values
        dna_damage_apop_half_max = pm.Uniform("dna_damage_apop_half_max", lower = 0, upper = 10)
        sim = pm.Simulator("sim", PhysiCell_model, params=[factor_max_apoptosis_drug1, dna_damage_apop_half_max], epsilon=100, observed=obs)

        idata_lv = pm.sample_smc(draws=1000, chains=4, cores=NumCoresPYMC, idata_kwargs={'log_likelihood':True} ) # return inference data (return_inferencedata=True is default pymc3 > 4.0)
        idata_lv.to_netcdf("InfData_Fig2_.nc") # save the inferencedata

        az.plot_trace(idata_lv, kind="rank_vlines")
        az.plot_posterior(idata_lv)
        plt.show()