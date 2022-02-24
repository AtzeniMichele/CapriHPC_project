#!/bin/bash

#SBATCH --job-name Capri_job
#SBATCH --output results_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --time 10:00
#SBATCH --mail-user michele.atzeni@studenti.unipd.it
#SBATCH --mail-type FAIL,END
#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --mem 16G

cd $SLURM_SUBMIT_DIR

srun singularity exec capri_project_container.sif python3 NNscript.py
