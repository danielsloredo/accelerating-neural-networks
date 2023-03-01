#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
#SBATCH --account=free # The account name for the job.
#SBATCH --gres=gpu:1
#SBATCH -c 1 # The number of cpu cores to use.
#SBATCH --time=6:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.
#SBATCH --output=%x.out
module load anaconda
module load cuda11.2/toolkit
#Command to execute Python program

python curves_no_hsigmoid.py $momentum $rate
#End of script
