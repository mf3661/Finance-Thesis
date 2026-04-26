#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=selection
#SBATCH --array=1-440
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=logs/selection_%a.txt

PY=/project/dachxiu/haohu/envs/LLM/env_llm/bin/python

N_WINDOW=44
TASK_ID=${SLURM_ARRAY_TASK_ID}

REPEAT_IDX=$(( (TASK_ID - 1) / N_WINDOW + 1 ))
WINDOW_IDX=$(( (TASK_ID - 1) % N_WINDOW + 1 ))

"$PY" -u 1_selection.py \
  --repeat_idx ${REPEAT_IDX} \
  --window_idx ${WINDOW_IDX}