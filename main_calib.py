from PhysiCellModel import PhysiCell_Model
from HPC_exploration import create_JOB
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

fileName = "SampleModel.ini"
key_model = "physicell_model_2"
key_HPC_params = "hpc_parameters"
# Create the structure of model exploration
PhysiCellModel1 = PhysiCell_Model(fileName, key_model)

def sum_stat(args):
    fx =  args[0] + args[1]*2
    # print(args[0], args[1], fx)
    return fx


def PhysiCell_model(rng, *args, size=None):
    # Set the parameters in PhysiCell class
    PhysiCellModel1.set_parameters( np.array([args[:-1]]) ) # Exclude the last arg that is size
    # Generate XML files for each simulation
    # PhysiCellModel1.createXMLs()
    # Run the model
    # print("... running PhysiCell")
    return sum_stat(args)


if __name__ == '__main__':
    print(f"Running on PyMC v{pm.__version__}")
    # call the rng number + parameters
    # PhysiCell_model(0, 10, 100 ) # Test of function
    with pm.Model() as model_lv:
        par1 = pm.HalfNormal("par1", 1.0)
        par2 = pm.HalfNormal("par2", 5.0)

        sim = pm.Simulator("sim", PhysiCell_model, params=[par1, par2], epsilon=0.1, observed=10)

        idata_lv = pm.sample_smc(draws=1000, chains=4, cores=1)
        
        az.plot_trace(idata_lv, kind="rank_vlines")
        az.plot_posterior(idata_lv)
        plt.show()