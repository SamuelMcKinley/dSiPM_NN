#!/bin/bash
#SBATCH -J batch_workflow_manager
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 2
#SBATCH --mem-per-cpu=16G

export PATH=~/miniconda3/envs/base/bin:$PATH

python3 -u workflow_manager.py "$1"
