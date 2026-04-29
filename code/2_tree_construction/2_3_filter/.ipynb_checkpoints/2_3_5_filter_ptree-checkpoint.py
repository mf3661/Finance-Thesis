import os
import argparse
import shutil
import numpy as np
import pandas as pd

# =========================================================
# paths
# =========================================================
COMBINE_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_2_combine_portfolio/2_2_5_combine_ptree/output"
FILTER_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_3_filter/2_3_5_filter_ptree/output"

# fixed columns
DATE_COL = "date"


def drop_sparse_columns(df: pd.DataFrame, min_valid_frac: float):
    """
    Drop columns whose fraction of non-missing observations is below min_valid_frac.
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]
    n_rows = len(df)

    keep_cols = []
    for c in non_date_cols:
        valid_frac = df[c].notna().sum() / n_rows if n_rows > 0 else 0.0
        if valid_frac >= min_valid_frac:
            keep_cols.append(c)

    out = df[[DATE_COL] + keep_cols].copy()
    return out, keep_cols


def drop_zero_variance_columns(df: pd.DataFrame):
    """
    Keep a column if:
    - it has at least 2 finite observations
    - and std over the finite part is positive
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    keep_cols = []
    for c in non_date_cols:
        x = df[c].to_numpy(dtype=float)
        good = np.isfinite(x)

        if good.sum() < 2:
            continue

        if np.std(x[good], ddof=0) > 0:
            keep_cols.append(c)

    out = df[[DATE_COL] + keep_cols].copy()
    return out, keep_cols


def dedup_columns_by_values(df: pd.DataFrame):
    """
    Deduplicate columns after sparse/variance filtering.
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    if len(non_date_cols) == 0:
        return df[[DATE_COL]].copy(), []

    tmp = df[non_date_cols].T
    keep_mask = ~tmp.duplicated()

    keep_cols = list(tmp.index[keep_mask])
    out = df[[DATE_COL] + keep_cols].copy()

    return out, keep_cols


def filter_one_repeat(input_dir: str, output_dir: str, year: int, repeat_id: int, min_valid_frac: float):
    os.makedirs(output_dir, exist_ok=True)

    ret_path_pkl = os.path.join(input_dir, "level_all_excess_ret_combined.pkl.gz")
    ret_path_parquet = os.path.join(input_dir, "level_all_excess_ret_combined.parquet")
    meta_path_parquet = os.path.join(input_dir, "portfolio_metadata.parquet")
    meta_path_csv = os.path.join(input_dir, "portfolio_metadata.csv")

    if os.path.exists(ret_path_pkl):
        ret_df = pd.read_pickle(ret_path_pkl).copy()
    elif os.path.exists(ret_path_parquet):
        ret_df = pd.read_parquet(ret_path_parquet).copy()
    else:
        raise FileNotFoundError(
            f"Cannot find return file in {input_dir}. "
            f"Tried {ret_path_pkl} and {ret_path_parquet}"
        )

    if os.path.exists(meta_path_parquet):
        metadata = pd.read_parquet(meta_path_parquet).copy()
    elif os.path.exists(meta_path_csv):
        if os.path.getsize(meta_path_csv) == 0:
            metadata = pd.DataFrame(columns=["column_name"])
        else:
            metadata = pd.read_csv(meta_path_csv).copy()
    else:
        raise FileNotFoundError(
            f"Cannot find metadata file in {input_dir}. "
            f"Tried {meta_path_parquet} and {meta_path_csv}"
        )

    if "column_name" not in metadata.columns:
        raise ValueError(f"Metadata in {input_dir} must contain column_name.")

    print("-" * 80)
    print(f"[INFO] OOS year: {year} | repeat_id: {repeat_id}")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Return shape before filtering: {ret_df.shape}")
    print(f"[INFO] Metadata shape before filtering: {metadata.shape}")
    print(f"[INFO] min_valid_frac: {min_valid_frac}")

    missing_before = int(ret_df.drop(columns=[DATE_COL], errors="ignore").isna().sum().sum())
    print(f"[INFO] Missing cells before filtering: {missing_before:,}")

    # =====================================================
    # 1. drop sparse columns
    # =====================================================
    before_cols = ret_df.shape[1]
    ret_df_sparse, _ = drop_sparse_columns(ret_df, min_valid_frac=min_valid_frac)
    after_cols = ret_df_sparse.shape[1]
    print(f"[INFO] Dropped sparse columns: {before_cols - after_cols}")

    # =====================================================
    # 2. drop zero variance columns
    # =====================================================
    before_cols = ret_df_sparse.shape[1]
    ret_df_var, _ = drop_zero_variance_columns(ret_df_sparse)
    after_cols = ret_df_var.shape[1]
    print(f"[INFO] Dropped zero variance columns: {before_cols - after_cols}")

    if ret_df_var.shape[1] <= 1:
        raise ValueError(
            f"After sparse + zero-variance filtering, no portfolio columns remain in {input_dir}. "
            f"Only date column is left."
        )

    # =====================================================
    # 3. dedup columns
    # =====================================================
    before_cols = ret_df_var.shape[1]
    ret_filtered, keep_cols = dedup_columns_by_values(ret_df_var)
    after_cols = ret_filtered.shape[1]
    print(f"[INFO] Dropped duplicated columns: {before_cols - after_cols}")

    # =====================================================
    # 4. filter metadata
    # =====================================================
    keep_col_set = set(keep_cols)
    metadata_filtered = metadata[metadata["column_name"].isin(keep_col_set)].copy()

    metadata_filtered["column_name"] = pd.Categorical(
        metadata_filtered["column_name"],
        categories=keep_cols,
        ordered=True
    )
    metadata_filtered = metadata_filtered.sort_values("column_name").reset_index(drop=True)
    metadata_filtered["column_name"] = metadata_filtered["column_name"].astype(str)

    # =====================================================
    # 5. final alignment check
    # =====================================================
    ret_cols_final = list(ret_filtered.columns[1:])
    meta_cols_final = metadata_filtered["column_name"].tolist()

    if ret_cols_final != meta_cols_final:
        raise ValueError(
            "Filtered return columns and metadata are not aligned.\n"
            f"ret first 5: {ret_cols_final[:5]}\n"
            f"meta first 5: {meta_cols_final[:5]}"
        )

    missing_after = int(ret_filtered.drop(columns=[DATE_COL], errors="ignore").isna().sum().sum())
    print(f"[INFO] Missing cells after filtering: {missing_after:,}")

    # save main outputs
    ret_filtered.to_pickle(os.path.join(output_dir, "level_all_excess_ret_combined_filtered.pkl.gz"))
    ret_filtered.to_csv(os.path.join(output_dir, "level_all_excess_ret_combined_filtered.csv"), index=False)
    ret_filtered.to_parquet(os.path.join(output_dir, "level_all_excess_ret_combined_filtered.parquet"), index=False)

    metadata_filtered.to_csv(os.path.join(output_dir, "portfolio_metadata_filtered.csv"), index=False)
    metadata_filtered.to_parquet(os.path.join(output_dir, "portfolio_metadata_filtered.parquet"), index=False)

    # pass-through optional helpful files
    passthrough_files = [
        "sample_map.csv",
        "sample_map.parquet",
        "adj_w.csv",
        "adj_w.parquet",
        "combo_specs.parquet",
        "split_rules.parquet",
        "combine_summary.csv",
    ]
    for fname in passthrough_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    summary = pd.DataFrame([{
        "year": year,
        "repeat_id": repeat_id,
        "n_rows": ret_filtered.shape[0],
        "n_portfolios": max(ret_filtered.shape[1] - 1, 0),
        "min_valid_frac": float(min_valid_frac),
        "n_missing_cells_before": missing_before,
        "n_missing_cells_after": missing_after,
        "n_metadata_rows": metadata_filtered.shape[0],
    }])
    summary.to_csv(os.path.join(output_dir, "filter_summary.csv"), index=False)

    print(f"[INFO] Return shape after filtering: {ret_filtered.shape}")
    print(f"[INFO] Metadata shape after filtering: {metadata_filtered.shape}")
    print(f"[INFO] Repeat {repeat_id} filter finished successfully")


def main(args):
    year = args.year
    N = args.N
    min_valid_frac = args.min_valid_frac

    if N <= 0:
        raise ValueError("N must be positive.")
    if not (0 < min_valid_frac <= 1):
        raise ValueError("min_valid_frac must be in (0, 1].")

    input_root = os.path.join(COMBINE_OUTPUT_BASE_DIR, f"oos_{year}")
    output_root = os.path.join(FILTER_OUTPUT_BASE_DIR, f"oos_{year}")
    os.makedirs(output_root, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] OOS year: {year}")
    print(f"[INFO] Combine input root: {input_root}")
    print(f"[INFO] Filter output root: {output_root}")
    print(f"[INFO] N repeats to filter: {N}")
    print(f"[INFO] min_valid_frac: {min_valid_frac}")

    for repeat_id in range(1, N + 1):
        input_dir = os.path.join(input_root, f"repeat_{repeat_id:02d}")
        output_dir = os.path.join(output_root, f"repeat_{repeat_id:02d}")

        filter_one_repeat(
            input_dir=input_dir,
            output_dir=output_dir,
            year=year,
            repeat_id=repeat_id,
            min_valid_frac=min_valid_frac,
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--min_valid_frac", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
