#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=filter
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=logs/filter.txt

PY=/project/dachxiu/haohu/envs/LLM/env_llm/bin/python

"$PY" -u 1_filter.py --B 10 --depth 4 --N 10 --base_seed 0