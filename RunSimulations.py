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
    mod = NumSimulations%size
    
    if ( mod != 0): NumSimulationsPerRank = NumSimulationsPerRank + 1 # if there is no equal split on the nodes, add a ghost simulation
    data = None

    if rank == 0:
        print(f"Total number of simulations: {NumSimulations} Simulations per rank: {NumSimulationsPerRank}")
        data = np.arange(NumSimulations) # [0,1,...,NumSimulations-1]
        if ( mod != 0):
          add = -1*np.ones((size-mod),dtype='d') # Add -1 in the ghost simulations
          data = np.concatenate((data,add),axis=None, dtype='d') # now len(data) % size == 0 

    recvbuf = np.empty(NumSimulationsPerRank, dtype='d') # These are the indexes of samples NOT the samples number
    comm.Scatter(data, recvbuf, root=0)
    for ind_sim in recvbuf:
        if ( ind_sim < 0 ): continue # If recvbuf is negative (-1) do NOT execute the model
        sampleID = Samples[int(ind_sim)]
        replicateID = Replicates[int(ind_sim)]
        print('Rank: ',rank, ', Simulation: ', ind_sim, ', Sample: ', sampleID,', Replicate: ', replicateID)
        # model(PhysiCell_Model.get_configFilePath(sampleID, replicateID), PhysiCell_Model.executable)
