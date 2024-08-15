from mpi4py import MPI 
import numpy as np
from uq_physicell.uq_physicell import PhysiCell_Model
from uq_physicell.sumstats import summ_func_FinalPopLiveDead

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':
    PhysiCellModel = PhysiCell_Model("examples/SampleModel.ini", 'physicell_model_2')
    
    # Sample parameters 
    Parameters_dic = {1: np.array([0.75,0.5]), 2: np.array([0.80,0.55])}
    NumSimulations = len(Parameters_dic)*PhysiCellModel.numReplicates
    NumSimulationsPerRank  = int(NumSimulations/size)
    
    # Generate a three list with size NumSimulations
    Parameters = []; Samples = []; Replicates = []
    for sampleID, par_value in Parameters.items():
        for replicateID in np.arange(PhysiCellModel.numReplicates):
            Parameters.append(par_value)
            Samples.append(sampleID)
            Replicates.append(replicateID)
    
    SplitIndexes = np.array_split(np.arange(NumSimulations),size, axis=0) # split [0,1,...,NumSimulations-1] in size arrays equally (or +1 in some ranks)
    
    print(f"Total number of simulations: {NumSimulations} Simulations per rank: {NumSimulationsPerRank}")

    for ind_sim in SplitIndexes[rank]:
        print('Rank: ',rank, ', Simulation: ', ind_sim, ', Sample: ', Samples[ind_sim],', Replicate: ', Replicates[ind_sim])
        PhysiCellModel.RunModel(Samples[ind_sim], Replicates[ind_sim],Parameters[ind_sim],SummaryFunction=summ_func_FinalPopLiveDead)

    MPI.Finalize()
