#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=simplify
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=2:00:00
#SBATCH --output=logs/combine.txt

PY=/project/dachxiu/haohu/envs/LLM/env_llm/bin/python

"$PY" -u 1_combine_portfolio.py --B 10 --depth 4 --N 10 --base_seed 0