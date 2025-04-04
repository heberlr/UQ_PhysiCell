from mpi4py import MPI
from uq_physicell import PhysiCell_Model
from uq_physicell import summ_func_FinalPopLiveDead
import numpy as np
import os

from SA_ex4_ex5 import SA_problem, SA_analyze, plot_analysis

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':
    PhysiCellModel = PhysiCell_Model("examples/SampleModel.ini", 'physicell_model_2')
    if rank == 0:
        PhysiCellModel.info()
        print("Sensitivity analysis - Sobol using MPI")
        # Remove the output folder
        if os.path.exists(PhysiCellModel.output_folder): os.system('rm -rf '+PhysiCellModel.output_folder)

    # Number of parameters expected in the XML and rules
    num_params_xml = len(PhysiCellModel.XML_parameters_variable)
    num_params_rules = len(PhysiCellModel.parameters_rules_variable)

    # Sensitivity analysis - Sobol
    problem, sa_sobol = SA_problem(PhysiCellModel)

    if rank == 0:
        print(f"SA problem +/- 20% of reference value: \n{sa_sobol}")
        print("Bounds parameters: ", problem['bounds'])

    # Generate a three list with size NumSimulations
    Parameters = []; Samples = []; Replicates = []
    for sampleID in range(sa_sobol.samples.shape[0]):
        for replicateID in np.arange(PhysiCellModel.numReplicates):
            # check if the file already exists - allows to resume the simulation since that we have a fixed seed
            if ( os.path.isfile(PhysiCellModel.output_folder+'SummaryFile_%06d_%02d.csv'%(sampleID,replicateID)) ) : continue
            Parameters.append(sa_sobol.samples[sampleID])
            Samples.append(sampleID)
            Replicates.append(replicateID)

    # Split simulations into ranks
    SplitIndexes = np.array_split(np.arange(len(Samples)),size, axis=0) # split [0,1,...,NumSimulations-1] in size arrays equally (or +1 in some ranks)
    if rank==0: print(f"Total number of simulations: {len(Samples)} Simulations per rank: {int(len(Samples)/size)}")

    # Run simulations
    for ind_sim in SplitIndexes[rank]:
        if (num_params_xml > 0): ParametersXML = Parameters[ind_sim][:num_params_xml]
        else: ParametersXML = np.array([])
        if (num_params_rules > 0): ParametersRules = Parameters[ind_sim][num_params_xml:]
        else: ParametersRules = np.array([])
        print('Simulation: ', ind_sim, ', Sample: ', Samples[ind_sim],', Replicate: ', Replicates[ind_sim], 'Parameters XML: ', ParametersXML, 'Parameters rules: ', ParametersRules)
        PhysiCellModel.RunModel(Samples[ind_sim], Replicates[ind_sim],ParametersXML, ParametersRules = ParametersRules,SummaryFunction=summ_func_FinalPopLiveDead)

    if rank == 0:
        # Analyze the results
        dic_analyzes = SA_analyze(PhysiCellModel, problem, sa_sobol)
        # Plot the analysis
        plot_analysis(problem, dic_analyzes)

    MPI.Finalize()
