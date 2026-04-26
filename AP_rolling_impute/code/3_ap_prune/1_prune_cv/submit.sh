#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --job-name=prune_cv
#SBATCH --array=1-440
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=6:00:00
#SBATCH --output=logs/prune_cv_%a.txt

# PY=/user/yw4389/.conda/envs/ipca_env/bin/python

#N_WINDOW=44
#TASK_ID=${SLURM_ARRAY_TASK_ID}
#
#REPEAT_IDX=$(( (TASK_ID - 1) / N_WINDOW + 1 ))
#WINDOW_IDX=$(( (TASK_ID - 1) % N_WINDOW + 1 ))
#
#"$PY" -u 1_prune_cv.py \
#  --repeat_idx ${REPEAT_IDX} \
#  --window_idx ${WINDOW_IDX}

#!/usr/bin/env bash
set -euo pipefail

PY=/user/yw4389/.conda/envs/ipca_env/bin/python
N_WINDOW=44

TASK_ID="${1:-}"

if [[ -z "$TASK_ID" || ! "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] Invalid TASK_ID: <$TASK_ID>"
  exit 1
fi

REPEAT_IDX=$(( (TASK_ID - 1) / N_WINDOW + 1 ))
WINDOW_IDX=$(( (TASK_ID - 1) % N_WINDOW + 1 ))

echo "[INFO] TASK_ID=$TASK_ID REPEAT_IDX=$REPEAT_IDX WINDOW_IDX=$WINDOW_IDX"

"$PY" -u 1_prune_cv.py \
  --repeat_idx "$REPEAT_IDX" \
  --window_idx "$WINDOW_IDX"