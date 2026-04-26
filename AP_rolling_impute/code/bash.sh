#!/bin/bash
set -euo pipefail

ROOT_DIR="/user/yw4389/Finance_Thesis/AP_rolling_impute/code"

scripts=(
  "$ROOT_DIR/1_data_processing/1_rank_data/submit.sh"
  "$ROOT_DIR/2_tree_construction/1_split/submit.sh"
  "$ROOT_DIR/2_tree_construction/2_combine_portfolio/submit.sh"
  "$ROOT_DIR/2_tree_construction/3_filter/submit.sh"
  "$ROOT_DIR/3_ap_prune/1_prune_cv/submit.sh"
  "$ROOT_DIR/3_ap_prune/2_selection/submit.sh"
)

cd "$ROOT_DIR"

for script in "${scripts[@]}"; do
  echo "Submitting $script"

  if [ ! -f "$script" ]; then
    echo "Missing script: $script"
    exit 1
  fi

  chmod +x "$script"

  output=$(grid_run --grid_submit=batch bash "$script")
  echo "$output"

  job_id=$(echo "$output" | sed -n 's/.*Your job \([0-9][0-9]*\).*/\1/p')

  if [ -z "$job_id" ]; then
    echo "Could not parse job ID for $script"
    exit 1
  fi

  echo "Waiting for job $job_id..."

  while qstat | awk 'NR>2 {print $1}' | grep -q "^${job_id}$"; do
    sleep 10
  done

  err_file="$ROOT_DIR/bash.e${job_id}"
  out_file="$ROOT_DIR/bash.o${job_id}"

  if [ -f "$err_file" ] && [ -s "$err_file" ]; then
    echo "Job $job_id for $script failed. Error log:"
    cat "$err_file"
    exit 1
  fi

  echo "Finished $script"
  if [ -f "$out_file" ] && [ -s "$out_file" ]; then
    echo "Stdout log:"
    cat "$out_file"
  fi
  echo "----------------------------------------"
done

echo "All jobs completed successfully."