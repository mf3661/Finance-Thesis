#!/bin/bash
#SBATCH --job-name=aptree_split_year
#SBATCH --account=pi-dachxiu
#SBATCH --partition=amd
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --array=1951-2024
#SBATCH --output=logs/aptree_split_%a.txt

# YEAR=${SLURM_ARRAY_TASK_ID}
# PY=/user/yw4389/.conda/envs/ipca_env/bin/python

# "$PY" -u 1_split.py --year ${YEAR} --B 100 --depth 4 --N 10 --base_seed 0
#for yr in $(seq 1950 2024); do
#  echo "Submitting year $yr";
#  grid_run --grid_submit=batch --grid_mem=40G bash submit.sh --year $yr --B 100 --depth 4 --N 10 --base_seed 0;
#  done

#!/usr/bin/env bash
set -euo pipefail

YEAR=""
B=10
DEPTH=4
N=10
BASE_SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --year)
      YEAR="$2"
      shift 2
      ;;
    --B)
      B="$2"
      shift 2
      ;;
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --N)
      N="$2"
      shift 2
      ;;
    --base_seed)
      BASE_SEED="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$YEAR" ]]; then
  echo "[ERROR] Missing --year"
  exit 1
fi

echo "[INFO] YEAR=$YEAR B=$B DEPTH=$DEPTH N=$N BASE_SEED=$BASE_SEED"
echo "[INFO] Host=$(hostname)"
echo "[INFO] PWD=$(pwd)"

PY=/user/yw4389/.conda/envs/ipca_env/bin/python

exec "$PY" -u 1_split.py \
  --year "$YEAR" \
  --B "$B" \
  --depth "$DEPTH" \
  --N "$N" \
  --base_seed "$BASE_SEED"