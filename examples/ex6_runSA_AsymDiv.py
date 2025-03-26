from uq_physicell import PhysiCell_Model

from mpi4py import MPI
import pcdl
import numpy as np
import pandas as pd
from shutil import rmtree
import os
import glob
from SALib import ProblemSpec
from SALib.analyze import sobol
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def summ_func_FinalFraction(outputPath,summaryFile, dic_params, SampleID, ReplicateID):
    # read the last file
    mcds = pcdl.TimeStep('final.xml',outputPath, microenv=False, graph=False, settingxml=None, verbose=False)
    # dataframe of cells
    df_cell = mcds.get_cell_df()
    # population stats live and dead cells
    proj1_cells = len(df_cell[ (df_cell['cell_type'] == 'progenitor_1') ] )
    proj2_cells = len(df_cell[ (df_cell['cell_type'] == 'progenitor_2') ] )
    # dataframe structure
    data = {'time': mcds.get_time(), 'sampleID': SampleID, 'replicateID': ReplicateID, 'frac_proj1_cells': proj1_cells/len(df_cell), 'frac_proj2_cells': proj2_cells/len(df_cell), 'run_time_sec': mcds.get_runtime()}
    data_conc = {**data,**dic_params} # concatenate output data and parameters
    df = pd.DataFrame([data_conc])
    # remove replicate output folder
    rmtree( outputPath )
    df.to_csv(summaryFile, sep='\t', encoding='utf-8')

def SA_problem(PhysiCellModel:PhysiCell_Model) -> tuple:
    # Define parameters range +/- 20% of the reference value
    dic_ref_value = {'cycle_duration_stem_cell': 1440.0, 'asym_div_to_prog_1_sat': 0.1}
    # Define parameters range +/- 20% of the reference value
    names_parameters = []
    bounds_parameters = []
     # Parameters from xml
    for param_name in PhysiCellModel.XML_parameters_variable.values():
        if not ( param_name in dic_ref_value.keys() ): continue
        names_parameters.append(param_name)
        bounds_parameters.append([dic_ref_value[param_name]*0.8, dic_ref_value[param_name]*1.2])
    # Parameters from rules
    if (PhysiCellModel.parameters_rules):
        for param_name in PhysiCellModel.parameters_rules_variable.values():
            if not ( param_name in dic_ref_value.keys() ): continue
            names_parameters.append(param_name)
            bounds_parameters.append([dic_ref_value[param_name]*0.8, dic_ref_value[param_name]*1.2])
    # Define SA problem
    problem = {'names': names_parameters, 'bounds': bounds_parameters}
    sa_sobol = ProblemSpec(problem)
     # Sample parameters - fixed seed
    sa_sobol.sample_sobol(2**2, calc_second_order=True, seed=42) # Recommend power of 2 - calculate second order = False: N*(D+2) samples OR True: N*(2D+2) samples
    return problem, sa_sobol

def SA_analyze(PhysiCellModel:PhysiCell_Model, problem:dict, sa_sobol:ProblemSpec) -> dict:
    # Read the output of all simulations
    data_files = glob.glob(PhysiCellModel.output_folder+'SummaryFile_*.csv')
    df_all = pd.concat((pd.read_csv(file, sep='\t', encoding='utf-8') for file in data_files), ignore_index=True)
    # Take the mean of replicates in each sample and time
    df_samples_mean = df_all.groupby(['sampleID', 'time'], as_index=False).mean()

    dic_analyzes = {}
    # Analyze for 2 QoIs: frac_proj1_cells and frac_proj2_cells
    for qoi in ['frac_proj1_cells', 'frac_proj2_cells']:
        sa_sobol.set_results(df_samples_mean[qoi].to_numpy())
        # Perform Sobol analysis on the QoI
        Si = sobol.analyze(problem, sa_sobol.results, calc_second_order=True)#, print_to_console=True)
        dic_analyzes[qoi] = {'S1': Si['S1'], 'ST': Si['ST']}
    return dic_analyzes

def plot_analysis(problem:dict, dic_analyzes: dict) -> None:
    # Plot the analysis
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(problem['names'], dic_analyzes['frac_proj1_cells']['S1'], label='First order')
    axes[0].bar(problem['names'], dic_analyzes['frac_proj1_cells']['ST'], label='Total effect')
    axes[0].set_title('Fraction of projenitor 1 cells')
    axes[0].set_ylabel('Sobol index')
    axes[1].bar(problem['names'], dic_analyzes['frac_proj2_cells']['S1'], label='First order')
    axes[1].bar(problem['names'], dic_analyzes['frac_proj2_cells']['ST'], label='Total effect')
    axes[1].set_title('Fraction of projenitor 2 cells')
    axes[1].set_ylabel('Sobol index')
    fig.subplots_adjust(wspace=0.5)
    axes[0].legend()
    axes[1].legend()
    plt.show()



if __name__ == '__main__':
    PhysiCellModel = PhysiCell_Model("examples/SampleModel.ini", 'asymmetric_division')
    if rank == 0:
        PhysiCellModel.info()
        print("Sensitivity analysis - Sobol using MPI")

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
    if rank ==0 : print(f"Total number of simulations: {len(Samples)} Simulations per rank: {int(len(Samples)/size)}\n Running simulations ...")

    # Run simulations
    for ind_sim in SplitIndexes[rank]:
        if (num_params_xml > 0): ParametersXML = Parameters[ind_sim][:num_params_xml]
        else: ParametersXML = np.array([])
        if (num_params_rules > 0): ParametersRules = Parameters[ind_sim][num_params_xml:]
        else: ParametersRules = np.array([])
        # Note that the PhysiCellModel structure is expecting 3 parameters, it is missing the third parameter: asym_div_to_prog_2_sat
        # Let's pass asym_div_to_prog_2_sat = 1.0 - asym_div_to_prog_1_sat
        ParametersRules = np.append(ParametersRules, 1.0 - ParametersRules[0])
        print('Simulation: ', ind_sim, ', Sample: ', Samples[ind_sim],', Replicate: ', Replicates[ind_sim], 'Parameters XML: ', ParametersXML, 'Parameters rules: ', ParametersRules)
        try: PhysiCellModel.RunModel(Samples[ind_sim], Replicates[ind_sim],ParametersXML, ParametersRules = ParametersRules,SummaryFunction=summ_func_FinalFraction, RemoveConfigFile=False)
        except Exception as e:
            print("Error in rank: ", rank, " Simulation: ", ind_sim, " Error: ", e)
            # kill all processes
            comm.Abort()

    if rank == 0:
        print("Analyzing the results ...")
        # Analyze the results
        dic_analyzes = SA_analyze(PhysiCellModel, problem, sa_sobol)
        # Plot the analysis
        plot_analysis(problem, dic_analyzes)

    MPI.Finalize()
