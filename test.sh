#!/bin/bash

#SBATCH --mail-user=username@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J guo
#SBATCH --output=slurm_out/out%j.out
#SBATCH --error=slurm_out/err%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A30|V100
#SBATCH -p academic
#SBATCH -t 23:00:00

python3 test.py