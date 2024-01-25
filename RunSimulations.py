from mpi4py import MPI 
import numpy as np
import sys
from HPC_exploration import model, args_run_simulations

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':
    PhysiCell_Model, Samples, Replicates = args_run_simulations(sys.argv[1:])
    NumSimulations = len(Samples)
    NumSimulationsPerRank  = int(NumSimulations/size)
    
    print(f"Total number of simulations: {NumSimulations} Simulations per rank: {NumSimulationsPerRank}")

    data = np.array_split(np.arange(NumSimulations),size, axis=0) # split [0,1,...,NumSimulations-1] in size arrays equally (or +1 in some ranks)

    for ind_sim in data[rank]:
        sampleID = Samples[ind_sim]
        replicateID = Replicates[ind_sim]
        print('Rank: ',rank, ', Simulation: ', ind_sim, ', Sample: ', sampleID,', Replicate: ', replicateID)
        model(PhysiCell_Model.get_configFilePath(sampleID, replicateID), PhysiCell_Model.executable)
