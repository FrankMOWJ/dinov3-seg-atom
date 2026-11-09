#!/bin/bash
#SBATCH --job-name=dinov3
#SBATCH --output=logs/{dinov3}_%J.log       # %J is the unique job ID
#SBATCH --mail-type=ALL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition gpu                   # Which partition to run on 
#SBATCH --gpus=1                          # Only if on a GPU partition, how many GPUs
#SBATCH --nodes=1                         # Number of CPU nodes
#SBATCH --ntasks-per-node=1               # Number of MPI tasks per node
#SBATCH --cpus-per-task=1                 # Number of CPUs per task
#SBATCH --mem-per-cpu=32G                 # Memory per CPU
#SBATCH --time=2:00:00                    # Time limit hrs:min:sec


date;hostname;pwd


# Do any setup you need here, like loading a conda environment or setting environment variables
module load miniconda
conda activate pytorch


# whatever you want your script to do here, e.g.
python train/train.py --config train/config.json

