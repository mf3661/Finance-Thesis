#!/bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -t 1-44
#$ -N ipca_sr_grid

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/user/yw4389/.conda/envs/ipca_env/bin/python}
SCRIPT_DIR=${SCRIPT_DIR:-$PWD}
RESULTS_DIR=${RESULTS_DIR:-$SCRIPT_DIR/Results/grid_runs}
TASK_ID=${SGE_TASK_ID:?SGE_TASK_ID is not set}

mkdir -p "$RESULTS_DIR"
mkdir -p "$SCRIPT_DIR/logs"

cd "$SCRIPT_DIR"

echo "Running task ${TASK_ID} on $(hostname)"
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" "$SCRIPT_DIR/sample_code_sr_grid.py" \
  --task_id "$TASK_ID" \
  --data_path "$SCRIPT_DIR/chars_raw_imputed.feather" \
  --chars_csv_path "$SCRIPT_DIR/chars_summary.csv" \
  --results_dir "$RESULTS_DIR"
