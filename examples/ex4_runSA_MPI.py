from uq_physicell import PhysiCell_Model, get_physicell
from uq_physicell.model_analysis import ModelAnalysisContext, run_simulations
from uq_physicell.model_analysis.utils import calculate_qoi_statistics
from uq_physicell.database.ma_db import load_structure

from shutil import rmtree
import pcdl
import pandas as pd
from mpi4py import MPI

def summ_func_FinalFraction(OutputFolder:str,SummaryFile:str, dic_params:dict, SampleID:int, ReplicateID:int):
    # read the last file
    mcds = pcdl.TimeStep('final.xml',OutputFolder, microenv=False, graph=False, settingxml=None, verbose=False)
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
    rmtree( OutputFolder )
    if (SummaryFile):
        df.to_csv(SummaryFile, sep='\t', encoding='utf-8')
        return None
    else: return df

if __name__ == "__main__":
    # Get MPI communicator and rank information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 handles initialization
    if rank == 0:
        print(f"Initializing MPI job with {size} processes...")
        
        # Database name to save all simulations
        db_path = "ex4_PhysiCell_SA_MPI.db"
        model_config  = {"ini_path": "Model_Struct.ini", "struc_name": 'asymmetric_division'}
        # Define parameter space
        params_info = {
            "cycle_duration_stem_cell": {"lower_bound": 1152.0, "upper_bound": 1728.0, "ref_value":1440.0 },
            "asym_div_to_prog_1_sat": {"lower_bound": 0.0, "upper_bound": 1.0, "ref_value": 0.0}
        }
        qoi_funcs = {'frac_proj1_cells':None, 'frac_proj2_cells':None, 'run_time_sec':None}
        # Define the name of sampler
        sampler = "Sobol"

        # Get PhysiCell, if not exist, it will be downloaded - Alternatively, you can change the paths in the Model_Struct.ini file to a PhysiCell folder on your system.
        get_physicell(target_dir=".")

        # Create a PhysiCell model structure just for checking the settings
        model = PhysiCell_Model(model_config['ini_path'], model_config['struc_name'])
        
        # Setup the context and number of workers to run simulations in MPI (inter-node parallel)
        context = ModelAnalysisContext(db_path, model_config, sampler, params_info, qois_info=qoi_funcs, num_workers=size, parallel_method="inter-node") 
        # Assign custom summary function in ModelAnalysisContext
        context.summary_function = summ_func_FinalFraction
        # Generate the samples using the specified sampler
        context.generate_samples(N=8)
        # Add to the samplers the parameter asym_div_to_prog_2_sat = 1 - asym_div_to_prog_1_sat
        for sample_id, params in context.dic_samples.items():
            params['asym_div_to_prog_2_sat'] = 1 - params['asym_div_to_prog_1_sat']
        print(f"Generated {len(context.dic_samples)} samples using {sampler}")
    else:
        # Non-root ranks just need a context placeholder for run_simulations
        context = None
    
    # Broadcast the context to all ranks (run_simulations will handle MPI distribution)
    context = comm.bcast(context, root=0)
    
    # All ranks participate in running simulations
    run_simulations(context)
    
    # Only rank 0 prints completion message
    if rank == 0:
        print(f"Simulations completed and results stored in the database: {context.db_path}.")