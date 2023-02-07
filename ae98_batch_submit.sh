#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --constraint=knl
#SBATCH --tasks-per-node=68
#SBATCH --qos=regular
#SBATCH --job-name=co2_52,1mJ_delayRail205mm_micro6,0mm_PMQs41mm_h2_31,7psig_input
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=oods@nersc.gov
#SBATCH --mail-type=ALL

# set up for problem & define any environment variables here
module load python
conda activate shadow_2021
export KMP_AFFINITY=disabled

srun -n 1 --cpu-bind=none python3 invert.py co2_52,1mJ_delayRail205mm_micro6,0mm_PMQs41mm_h2_31,7psig_input
