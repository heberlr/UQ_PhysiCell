import traceback
from uq_physicell import PhysiCell_Model
from uq_physicell.model_analysis import ModelAnalysisContext, run_simulations
from mpi4py import MPI
from pathlib import Path

def create_model_analysis_context(struc_name):
    db_path = f"uq_Simulations_OAT_{struc_name}.db"  # Path to the database file
    model_config = {"ini_path": "uq_config.ini", "struc_name": struc_name} # Example model configuration
    sampler = 'OAT'  # Example sampler
    params_info = { # Example parameters information
        'IC_file': {'ref_value': None, 'lower_bound': None, 'upper_bound': None, 'type': 'string'},
        'epi_normal_rule_pressureDcellcycle_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'epi_normal_rule_ecmImesenc_normal_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_normal_rule_ecmDspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_normal_rule_ecmIspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_normal_rule_infsignalDepi_normal_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'fib_rule_ecmDspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'fib_rule_ecmIspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'epi_tumor_rule_pressureDcellcycle_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'epi_tumor_rule_ecmImesenc_tumor_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_tumor_rule_ecmDspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_tumor_rule_ecmIspeed_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'},
        'mesenc_tumor_rule_infsignalDepi_tumor_inactive': {'ref_value': 0, 'lower_bound': 0, 'upper_bound': 1, 'perturbation': 100.0, 'type': 'bool'}
    }  
    qois_info = {}  # Example QoIs information (empty for this example) - Simulation output will be a vector of mcds (MultiCellular Data Standard) objects   
    # Create a PhysiCell model structure just for checking the settings
    model = PhysiCell_Model(model_config['ini_path'], model_config['struc_name'])
    # Create the model analysis context
    return ModelAnalysisContext(db_path, model_config, sampler, params_info, qois_info, parallel_method='inter-node')

def my_custom_sampler(params_dict: dict, sampler: str = 'OAT') -> dict:
    """Generate parameter samples using local sampling methods for sensitivity analysis.
    
    This function creates parameter samples using local sampling strategies,
    particularly the One-At-a-Time (OAT) method, where parameters are perturbed
    individually around reference values.
    
    Args:
        params_dict (dict): Dictionary containing parameter definitions. Each
            parameter should have:
            - 'ref_value': Reference value for the parameter
            - 'perturbation': Single value or list of perturbation percentages
        sampler (str, optional): Local sampling method to use. Currently only
            'OAT' (One-At-a-Time) is supported. Defaults to 'OAT'
    """
    from uq_physicell.model_analysis.samplers import run_local_sampler
    IC_files = ['cells_1_to_1.csv', 'cells_1_to_2.csv', 'cells_1_to_5.csv', 'cells_1_to_10.csv', 'cells_2_to_1.csv', 'cells_5_to_1.csv', 'cells_10_to_1.csv']
    # Remove IC_file parameter from params_dict for local sampling
    dic_samples_rules = params_dict.copy()
    dic_samples_rules.pop('IC_file')
    # Generate samples for the other parameters
    dic_samples_rules = run_local_sampler(dic_samples_rules, sampler=sampler)
    # Expand samples for each IC file
    updated_samples = {}
    for params in dic_samples_rules.values():
        for IC_path in IC_files:
            sample_id = len(updated_samples)
            updated_samples[sample_id] = params.copy()
            updated_samples[sample_id]['IC_file'] = IC_path
    return updated_samples

if __name__ == "__main__":
    # Get MPI communicator and rank information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 handles initialization
    if rank == 0:
        # Run global sampling
        context = create_model_analysis_context('CoCulture')
        # Generate samples using the specified sampler
        try:
            # context.generate_samples() Can't be used here because of IC_path parameter
            context.dic_samples = my_custom_sampler(context.params_dict, sampler=context.dic_metadata['Sampler'])
            print(f"Generated {len(context.dic_samples)} samples using {context.dic_metadata['Sampler']} for CoCulture.")
        except Exception as e:  
            print(f"Error generating samples: {e}")
            traceback.print_exc()
            raise
    else:
        # Non-root ranks just need a context placeholder for run_simulations
        context = None
    
    # Broadcast the context to all ranks (run_simulations will handle MPI distribution)
    context = comm.bcast(context, root=0)

    # All ranks must participate in MPI simulation
    run_simulations(context)
    if rank == 0:
        print(f"Simulations completed and results stored in the database: {context.db_path}.")