#!/bin/bash
#$ -cwd
#$ -j y
#$ -o logs/final_turnover_$JOB_ID.log

set -euo pipefail

PYTHON_BIN=/user/yw4389/.conda/envs/ipca_env/bin/python
ROOT_PARENT=/user/yw4389/Finance_Thesis/AP_rolling_impute/data/2_tree_construction/1_split/output
STOCK_RET_PATH=/user/yw4389/Finance_Thesis/AP_rolling_impute/data/monthly_stock_returns.parquet
OUT_DIR=/user/yw4389/Finance_Thesis/AP_rolling_impute/data/4_turnover_optionA/output

mkdir -p logs
mkdir -p "$OUT_DIR"

"$PYTHON_BIN" compute_node_turnover_optionA.py \
    --root-parent "$ROOT_PARENT" \
    --stock-ret-path "$STOCK_RET_PATH" \
    --stock-ret-format parquet \
    --out-dir "$OUT_DIR" \
    --leaf-date-col date \
    --leaf-stock-col permno \
    --ret-date-col date \
    --ret-stock-col permno \
    --ret-col ret
