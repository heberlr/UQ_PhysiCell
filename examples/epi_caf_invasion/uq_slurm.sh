#!/bin/bash
#SBATCH --job-name=UQ_epi_caf_invasion
#SBATCH --mail-type=FAIL,END
#SBATCH --mem=64G
#SBATCH --output=UQ_epi_caf_invasion_%j.log
#SBATCH --error=UQ_epi_caf_invasion_%j.err
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=4

# -----------------------------
# How to run this script:
# sbatch --account=$SLURM_ACCOUNT --mail-user=$MY_EMAIL uq_slurm.sh 
# -----------------------------

#load python
module load libfabric
module load cray-mpich-ucx
module load conda
conda activate dask_env

# Run simulations with MPI
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running UQ simulations with MPI"
echo "Number of MPI tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
srun python uq_script.py
echo "Script finished"