from PhysiCellModel import PhysiCell_Model
from HPC_exploration import create_JOB
from HPC_exploration import model
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import subprocess
import PhysiCell_sumstats as sumstat
import os
from datetime import datetime
from PhysiCell_sumstats import final_population

fileName = "SampleModel.ini"
key_model = "pancreatic_model"
# key_HPC_params = "hpc_parameters"
# Create the structure of model exploration
PhysiCellModel1 = PhysiCell_Model(fileName, key_model)

def sum_stat(args):
    fx =  args[0] + args[1]*2
    # print(args[0], args[1], fx)
    return fx


def PhysiCell_model(rng, *args, size=None):
    # Set the parameters in PhysiCell class
    SampleID = [int(
        str(os.getpid())+
        str(datetime.now().hour)+
        str(datetime.now().minute)+
        str(datetime.now().second)
    )] # Needs to be a list
    print(np.array([args[:-1]]), SampleID)
    PhysiCellModel1.set_parameters( np.array([args[:-1]]), SampleID ) # Exclude the last arg that is size
    # Generate XML files for this simulation (all replicates)
    PhysiCellModel1.createXMLs()
    # Run the model (local test)
    print("... running PhysiCell.")
    for ReplicateID in range(1, PhysiCellModel1.numReplicates):
        subprocess(model(PhysiCellModel1.get_configFilePath(SampleID, ReplicateID), PhysiCellModel1.executable))
    print(f"... successfully ran PhysiCell sample {SampleID}.")
    # Run the model (HPC)
    # print("... running PhysiCell")
    return sum_stat(args)
# serial parallelizes
# run 8 replicates oeach on own thread with different
# run 8-threaded physicell looped 8 times with 8 different seeds

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