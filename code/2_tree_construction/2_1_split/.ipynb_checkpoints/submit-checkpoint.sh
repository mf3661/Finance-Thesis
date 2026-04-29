#!/bin/bash
#SBATCH --job-name=aptree_split_year
#SBATCH --account=pi-dachxiu
#SBATCH --partition=amd
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --array=1951-2024
#SBATCH --output=logs/aptree_split_%a.txt

YEAR=${SLURM_ARRAY_TASK_ID}
PY=/project/dachxiu/haohu/envs/LLM/env_llm/bin/python

"$PY" -u 1_split.py --year ${YEAR} --B 10 --depth 4 --N 10 --base_seed 0
