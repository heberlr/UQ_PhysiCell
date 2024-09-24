from uq_physicell.uq_physicell import PhysiCell_Model, get_rule_index_in_csv
import glob
import pandas as pd
from SALib import ProblemSpec
from SALib.analyze import sobol
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create the structure of model exploration
    PhysiCellModel = PhysiCell_Model("examples/SampleModel.ini", "physicell_model_2")
    
     # Sensitivity analysis - Sobol
    dic_ref_value = {'viral_replication_rate': 0.75, 'min_virion_count': 0.5}
    # Define parameters range +/- 20% of the reference value
    names_parameters = []
    bounds_parameters = []
    # Parameters from xml
    for key_xml in PhysiCellModel.keys_variable_params:
        names_parameters.append(PhysiCellModel.parameters[key_xml][1])
        bounds_parameters.append([float(dic_ref_value[names_parameters[-1]])*0.8, float(dic_ref_value[names_parameters[-1]])*1.2])
    num_params_xml = len(names_parameters)
    # Parameters from rules
    if (PhysiCellModel.parameters_rules):
        for key_rule, list_rule in PhysiCellModel.parameters_rules.items():
            id_rule = get_rule_index_in_csv(PhysiCellModel.rules, key_rule)
            parameter_rule = key_rule.split(',')[-1]
            names_parameters.append(list_rule[1])
            bounds_parameters.append([float(PhysiCellModel.rules[id_rule][parameter_rule])*0.8, float(PhysiCellModel.rules[id_rule][parameter_rule])*1.2])
    num_params_rules = len(names_parameters) - num_params_xml
    # Define SA problem
    problem = {'names': names_parameters, 'bounds': bounds_parameters}
    sa_sobol = ProblemSpec(problem)
    # Sample parameters - fixed seed
    sa_sobol.sample_sobol(2**2, calc_second_order=True, seed=42) # Recommend power of 2 - calculate second order = False: N*(D+2) samples OR True: N*(2D+2) samples

    # Read the output of all simulations
    data_files = glob.glob(PhysiCellModel.outputs_folder+'SummaryFile_*.csv')
    df_all = pd.concat((pd.read_csv(file, sep='\t', encoding='utf-8') for file in data_files), ignore_index=True)
    # Take the mean of replicates in each sample and time
    df_samples_mean = df_all.groupby(['sampleID', 'time'], as_index=False).mean() 
    
    dic_analyzes = {}
    # Analyze for 2 QoIS: live_cells and dead_cells
    for qoi in ['live_cells', 'dead_cells']:
        sa_sobol.set_results(df_samples_mean[qoi].to_numpy())
        # Perform Sobol analysis on the QoI
        Si = sobol.analyze(problem, sa_sobol.results, calc_second_order=True)#, print_to_console=True)
        dic_analyzes[qoi] = {'S1': Si['S1'], 'ST': Si['ST']}

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