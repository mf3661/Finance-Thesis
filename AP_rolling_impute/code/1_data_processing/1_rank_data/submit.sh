#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=rank
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rank.txt

PY=/user/yw4389/.conda/envs/ipca_env/bin/python

"$PY" -u 1_rank_data.py