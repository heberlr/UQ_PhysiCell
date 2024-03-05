import numpy as np
import os, sys, subprocess
from uq_physicell.uq_physicell import PhysiCell_Model
import configparser # read config *.ini file

# Create a slurm script job
def create_JOB(key_params, ID_Job, args): 
    configFile_name = args[0]
    PhysiCellModel = args_run_simulations(args)[0] # Checking args

    # Read the config file with the HPC parameters
    configFile = configparser.ConfigParser()
    with open(configFile_name) as fd:
        configFile.read_file(fd)
    path_run_script = configFile[key_params]['path_run_script'] # the path of RunSimulations script
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
    run = 'python '+ path_run_script +'/RunSimulations.py'
    for arg in args:
        run += ' '+arg
    print ("srun --cpu-bind=sockets "+run)
    sys.stdout = original_stdout