#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --constraint=knl
#SBATCH --tasks-per-node=68
#SBATCH --qos=regular
#SBATCH --job-name=ae98_inverse_shadowgraphy
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=oods@nersc.gov
#SBATCH --mail-type=ALL

# set up for problem & define any environment variables here
module load python
conda activate shadow_2021

srun -n 1 --cpu-bind=none python invert.py ae98_2022

# perform any cleanup or short post-processing here
