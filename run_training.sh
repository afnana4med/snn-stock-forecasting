#!/bin/bash
#SBATCH --job-name=snn_train
#SBATCH --output=logs/snn_train-%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# Load modules if needed (depends on cluster config)
# module load cuda/11.4

# Activate your conda environment if running on a cluster
# source ~/.bashrc
# conda activate genn5

# Run all three experiments from the project root
cd "$(dirname "$0")"
python -m snn_stock.main --config configs/rate_eprop.yaml
python -m snn_stock.main --config configs/temporal_eprop.yaml
python -m snn_stock.main --config configs/temporal_eventprop.yaml
