#!/bin/bash
#SBATCH --job-name=GEN
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --time=17:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=1-1000
#SBATCH --hint=nomultithread
#SBATCH --partition=tau

sh t1.sh $(($SLURM_ARRAY_TASK_ID+$1))