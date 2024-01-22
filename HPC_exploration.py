import numpy as np
import os, sys, subprocess
from PhysiCellModel import PhysiCell_Model
import configparser # read config *.ini file

# Define the PhysiCell execution            
def model(ConfigFile, Executable):
    # Write input for simulation & execute
    callingModel = [Executable, ConfigFile]
    cache = subprocess.run( callingModel,universal_newlines=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ( cache.returncode != 0):
        print(f"Error: model output error! Sample: {Sample} Replicate {Replicate}. returned: \n{str(cache.returncode)}")
        os._exit(1)

# Check if the arguments are compatible with PhysiCell simulations structure
def args_run_simulations(args):
    configFile_name = args[0]
    pc_key = args[1] 
    mode = args[2]

    PhysiCellModel = PhysiCell_Model(configFile_name, pc_key)
    num_replicates = PhysiCellModel.numReplicates # number of replicates for each simulation

    try:
        if( mode == 'sequential' ): # [ConfigFile_name] [PhysiCell_key] sequential [Initial ID sample] [Final ID sample]
            initial_sample = int(args[3])
            final_sample = int(args[4])
            num_samples = final_sample - initial_sample
            if ( (initial_sample >= PhysiCellModel.parameterSamples.shape[0]) or (final_sample > PhysiCellModel.parameterSamples.shape[0]) or ( initial_sample >= final_sample) or  (initial_sample < 0) ):
                print(f"Error: Sample range unexpected! Interval [{initial_sample} , {final_sample}) doesn't match with samples in [{pc_key}] from {configFile_name} .")
                sys.exit(1)
            Samples = np.arange(initial_sample,final_sample)# initial_sample, initial_sample+1, ..., final_sample-1
            Replicates = np.arange(num_replicates) # 0,1,...,num_replicates-1
            Samples = np.repeat(Samples, num_replicates)
            Replicates = np.tile(Replicates, num_samples)
        elif ( mode == 'samples' ): # [ConfigFile_name] [PhysiCell_key] samples [ ID_sample_1  ID_sample_2 ... ID_sample_n]
            num_samples = len(args)-3
            Samples = np.zeros(shape=(num_samples), dtype='d')
            Replicates = np.arange(num_replicates) # 0,1,...,num_replicates-1
            for sample_index in range(len(Samples)):
                sample_id = int(args[sample_index+3])
                if ( (sample_id >= PhysiCellModel.parameterSamples.shape[0]) or  (sample_id < 0) ):
                    print(f"Error: Sample {sample_id} unexpected! It doesn't match with samples size = {PhysiCellModel.parameterSamples.shape[0]} in [{pc_key}] from {configFile_name}.")
                    sys.exit(1)
                Samples[sample_index] = sample_id # samples starts in args[3]
            Samples = np.repeat(Samples, num_replicates)
            Replicates = np.tile(Replicates, num_samples)
        elif ( mode == 'individual' ): # [ConfigFile_name] [PhysiCell_key] individual [ ID_sample_1  Replicate_ID_sample_1 ... ID_sample_n Replicate_ID_sample_n]
            num_samples = (len(args)-3)*0.5 # pair sample and replicate  
            if ( num_samples % 1 != 0):
                print("Error: Number of args for individual runs!\n Please provide the sample and replicate pair(s).")
                sys.exit(1)
            num_samples = int(num_samples)
            Samples = np.zeros(shape=(num_samples), dtype='d')
            Replicates = np.zeros(shape=(num_samples), dtype='d')
            for sample_index in range(len(Samples)):
                sample_id = int(args[2*sample_index+3]) # args[3], args[5], args[7] ...
                replicate_id = int(args[2*sample_index+4]) # args[4], args[6], args[8] ..
                if ( (sample_id >= PhysiCellModel.parameterSamples.shape[0]) or (sample_id < 0) or (replicate_id < 0) or (replicate_id >= num_replicates) ):
                    print(f"Error: Sample: {sample_id} and replicate: {replicate_id} unexpected! It doesn't match with PhysiCell in [{pc_key}] from {configFile_name}].")
                    sys.exit(1)
                Samples[sample_index] =  sample_id
                Replicates[sample_index] =  replicate_id
        else:
            print(f"Error: The mode {mode} unexpected!\n Please provide a valid mode: sequential, samples, or individual.")
            sys.exit(1)
    except:
        print("Error: Unexpected syntax!")
        print("option 1)  [ConfigFile_name] [PhysiCell_key] sequential [Initial ID sample] [Final ID sample]  # it doesn't include Final ID sample")
        print("option 2)  [ConfigFile_name] [PhysiCell_key] samples [ ID_sample_1  ID_sample_2 ... ID_sample_n] # run all replicates for each sample")
        print("option 3)  [ConfigFile_name] [PhysiCell_key] individual [ ID_sample_1  Replicate_ID_sample_1 ... ID_sample_n Replicate_ID_sample_n] # run individual replicate for each sample")
        sys.exit(1)
    
    return PhysiCellModel, Samples, Replicates

# Create a slurm script job
def create_JOB(key_params, ID_Job, args): 
    configFile_name = args[0]
    PhysiCellModel = args_run_simulations(args)[0] # Checking args

    # Read the config file with the HPC parameters
    configFile = configparser.ConfigParser()
    with open(configFile_name) as fd:
        configFile.read_file(fd)

    email = configFile[key_params].get("email", fallback=None) # receive email if the job fail
    slurm_account_name = configFile[key_params]["slurm_account_name"] # account name for cluster simulation
    numNodes = configFile[key_params]['numNodes'] # number of nodes
    numTaskPerNode = configFile[key_params]["numTaskPerNode"] # number of processors on each node - avaliability: max(omp_num_threads * numTaskPerNode)
    memory = configFile[key_params].get("memory", fallback=None) # memory allocation to simulations
    Maxtime = configFile[key_params]["Maxtime"] # max time of simulation - format of time day-hour:minute:second ("0-00:00:00")
    projName = PhysiCellModel.projName # Name of project
    omp_num_threads = PhysiCellModel.omp_num_threads # number of threads omp for PhysiCell simulation

    # Creating the script file
    original_stdout = sys.stdout
    sys.stdout = open('job_'+str("%06d"%ID_Job)+'.sh','w')
    print ("#!/bin/bash\n")
    if (email):
        print ("#SBATCH --mail-user="+email)
        print ("#SBATCH --mail-type=FAIL")
    print ("#SBATCH --job-name="+PhysiCellModel.projName+str("%06d"%ID_Job))
    print ("#SBATCH -p general")
    print ("#SBATCH -o "+PhysiCellModel.projName+"_%j.txt")
    print ("#SBATCH -e "+PhysiCellModel.projName+"_%j.err")
    print ("#SBATCH --nodes="+numNodes)
    print ("#SBATCH --ntasks-per-node="+numTaskPerNode)
    print ("#SBATCH --cpus-per-task="+PhysiCellModel.omp_num_threads)
    print ("#SBATCH --time="+Maxtime)
    print("#SBATCH -A "+slurm_account_name)
    if (memory):
        print ("#SBATCH --mem="+memory)
    print ("\nmodule load python/3.9.8")
    print ("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n")
    run = 'python hpc/RunSimulations.py'
    for arg in args:
        run += ' '+arg
    print ("srun --cpu-bind=sockets "+run)
    sys.stdout = original_stdout