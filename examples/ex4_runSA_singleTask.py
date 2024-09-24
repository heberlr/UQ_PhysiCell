from uq_physicell.uq_physicell import PhysiCell_Model, get_rule_index_in_csv
from uq_physicell.sumstats import summ_func_FinalPopLiveDead
import numpy as np
import os
from SALib import ProblemSpec

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
    sa_sobol = ProblemSpec({'names': names_parameters, 'bounds': bounds_parameters})

    # Sample parameters - fixed seed
    sa_sobol.sample_sobol(2**2, calc_second_order=True, seed=42) # Recommend power of 2 - calculate second order = False: N*(D+2) samples OR True: N*(2D+2) samples
    print(sa_sobol)
    
    # Generate a three list with size NumSimulations
    Parameters = []; Samples = []; Replicates = []
    for sampleID in range(sa_sobol.samples.shape[0]):
        for replicateID in np.arange(PhysiCellModel.numReplicates):
            # check if the file already exists - allows to resume the simulation since that we have a fixed seed
            if ( os.path.isfile(PhysiCellModel.outputs_folder+'SummaryFile_%06d_%02d.csv'%(sampleID,replicateID)) ) : continue
            Parameters.append(sa_sobol.samples[sampleID])
            Samples.append(sampleID)
            Replicates.append(replicateID)
    
    # Run simulations
    for ind_sim in range(len(Samples)):
        if (num_params_xml > 0): ParametersXML = Parameters[ind_sim][:num_params_xml]
        else: ParametersXML = np.array([])
        if (num_params_rules > 0): ParametersRules = Parameters[ind_sim][num_params_xml:]
        else: ParametersRules = np.array([])
        print('Simulation: ', ind_sim, ', Sample: ', Samples[ind_sim],', Replicate: ', Replicates[ind_sim], 'Parameters XML: ', ParametersXML, 'Parameters rules: ', ParametersRules)
        PhysiCellModel.RunModel(Samples[ind_sim], Replicates[ind_sim], ParametersXML, ParametersRules = ParametersRules,SummaryFunction=summ_func_FinalPopLiveDead)