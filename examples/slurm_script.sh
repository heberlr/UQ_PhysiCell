#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --mail-type=FAIL,END
#SBATCH --mem=64G
#SBATCH --mail-user=your_email@example.com
#SBATCH --output=example_job_%j.log
#SBATCH --error=example_job_%j.err
#SBATCH --time=0-12:00:00
#SBATCH -A your_account
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=8

# Explanation:
# OpenMP threads are used to parallelize individual runs of the simulation.
# The number of threads is set using the OMP_NUM_THREADS environment variable, 
# which is derived from the --cpus-per-task option.
# MPI tasks are used to run multiple simulations in parallel, with the number of tasks 
# specified by the --ntasks option.

# Load Python environment - install uq_physicell in your environment: pip install uq_physicell
module load conda
conda activate your_env_name

# Compile the model
make load PROJ=your_project_path && make

# Run simulations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running the script"
srun python ./path_to_your_script/your_script.py
echo "Script finished"