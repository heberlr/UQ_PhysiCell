from uq_physicell import PhysiCell_Model, get_rule_index_in_csv
import glob
import pandas as pd
from SALib import ProblemSpec
from SALib.analyze import sobol
import matplotlib.pyplot as plt

def SA_problem(PhysiCellModel:PhysiCell_Model) -> tuple:
    # Define parameters range +/- 20% of the reference value
    dic_ref_value = {'viral_replication_rate': 0.75, 'min_virion_count': 0.5}
    # Define parameters range +/- 20% of the reference value
    names_parameters = []
    bounds_parameters = []
     # Parameters from xml
    for param_name in PhysiCellModel.XML_parameters_variable.values():
        names_parameters.append(param_name)
        bounds_parameters.append([dic_ref_value[param_name]*0.8, dic_ref_value[param_name]*1.2])
    # Parameters from rules
    if (PhysiCellModel.parameters_rules):
        for param_name in PhysiCellModel.parameters_rules_variable.values():
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
    # Analyze for 1 QoI: live_cells and dead_cells
    for qoi in ['live_cells', 'dead_cells']:
        sa_sobol.set_results(df_samples_mean[qoi].to_numpy())
        # Perform Sobol analysis on the QoI
        Si = sobol.analyze(problem, sa_sobol.results, calc_second_order=True)#, print_to_console=True)
        dic_analyzes[qoi] = {'S1': Si['S1'], 'ST': Si['ST']}
    return dic_analyzes

def plot_analysis(problem:dict, dic_analyzes: dict) -> None:
    # Plot the analysis
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(problem['names'], dic_analyzes['live_cells']['S1'], label='First order')
    axes[0].bar(problem['names'], dic_analyzes['live_cells']['ST'], label='Total effect')
    axes[0].set_title('Live cells')
    axes[0].set_ylabel('Sobol index')
    axes[1].bar(problem['names'], dic_analyzes['dead_cells']['S1'], label='First order')
    axes[1].bar(problem['names'], dic_analyzes['dead_cells']['ST'], label='Total effect')
    axes[1].set_title('Dead cells')
    axes[1].set_ylabel('Sobol index')
    fig.subplots_adjust(wspace=0.5)
    axes[0].legend()
    axes[1].legend()
    plt.show()
