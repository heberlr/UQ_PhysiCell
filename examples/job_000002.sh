#!/bin/bash

#SBATCH --mail-user=hlimadar@iu.edu
#SBATCH --mail-type=FAIL
#SBATCH --job-name=Template000002
#SBATCH -p general
#SBATCH -o Template_%j.txt
#SBATCH -e Template_%j.err
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH -A r00241

module load python/3.9.8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --cpu-bind=sockets python ./RunSimulations.py SampleModel.ini physicell_model individual 0 1 1 0
