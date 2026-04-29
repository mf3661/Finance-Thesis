import os
import argparse
import shutil
import numpy as np
import pandas as pd

# =========================================================
# paths
# =========================================================
SPLIT_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_1_split/2_1_5_split_ptree/output"
COMBINE_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_2_combine_portfolio/2_2_5_combine_ptree/output"
RF_PATH = "../../../data/common/rf.csv"

DATE_COL = "date"


def convert_tb3ms_to_monthly_rf(rf_df: pd.DataFrame):
    rf_df = rf_df.copy()
    rf_df["observation_date"] = pd.to_datetime(rf_df["observation_date"])
    rf_df["month"] = rf_df["observation_date"].dt.to_period("M")
    rf_df["rf_monthly"] = (1.0 + rf_df["TB3MS"] / 100.0) ** (1.0 / 12.0) - 1.0
    rf_df = rf_df[["month", "rf_monthly"]].drop_duplicates(subset=["month"]).sort_values("month")
    return rf_df


def subtract_rf(ret_df: pd.DataFrame, rf_path: str):
    ret_df = ret_df.copy()
    ret_df[DATE_COL] = pd.to_datetime(ret_df[DATE_COL])
    ret_df["month"] = ret_df[DATE_COL].dt.to_period("M")

    rf_raw = pd.read_csv(rf_path)
    rf_monthly = convert_tb3ms_to_monthly_rf(rf_raw)

    ret_df = ret_df.merge(rf_monthly, on="month", how="left")
    ret_cols = [c for c in ret_df.columns if c not in [DATE_COL, "month", "rf_monthly"]]
    ret_df[ret_cols] = ret_df[ret_cols].sub(ret_df["rf_monthly"], axis=0)
    ret_df = ret_df.drop(columns=["month", "rf_monthly"])
    return ret_df


def drop_zero_variance_columns(df: pd.DataFrame):
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
    non_date_cols = [c for c in df.columns if c != DATE_COL]
    if len(non_date_cols) == 0:
        return df[[DATE_COL]].copy(), []

    tmp = df[non_date_cols].T
    keep_mask = ~tmp.duplicated()
    keep_cols = list(tmp.index[keep_mask])
    out = df[[DATE_COL] + keep_cols].copy()
    return out, keep_cols


def combine_one_repeat(
    year: int,
    repeat_id: int,
    split_input_dir: str,
    repeat_output_dir: str,
    keep_nodes: str,
    drop_missing_months: bool,
    enforce_expected_counts: bool,
):
    os.makedirs(repeat_output_dir, exist_ok=True)

    node_ret_path = os.path.join(split_input_dir, "node_returns.parquet")
    meta_path = os.path.join(split_input_dir, "node_metadata.parquet")
    combo_path = os.path.join(split_input_dir, "combo_specs.parquet")
    split_rules_path = os.path.join(split_input_dir, "split_rules.parquet")

    if not os.path.exists(node_ret_path):
        raise FileNotFoundError(f"Cannot find node return file: {node_ret_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find metadata file: {meta_path}")

    node_returns = pd.read_parquet(node_ret_path).copy()
    metadata = pd.read_parquet(meta_path).copy()

    print("-" * 80)
    print(f"[INFO] OOS year: {year} | repeat_id: {repeat_id}")
    print(f"[INFO] node_returns rows before repeat filter: {len(node_returns):,}")
    print(f"[INFO] metadata rows before repeat filter: {len(metadata):,}")

    node_returns = node_returns[node_returns["repeat_id"] == repeat_id].copy()
    metadata = metadata[metadata["repeat_id"] == repeat_id].copy()

    if node_returns.empty:
        raise ValueError(f"No rows found for repeat_id={repeat_id} in {node_ret_path}")
    if metadata.empty:
        raise ValueError(f"No metadata rows found for repeat_id={repeat_id} in {meta_path}")

    print(f"[INFO] node_returns rows after repeat filter: {len(node_returns):,}")
    print(f"[INFO] metadata rows after repeat filter: {len(metadata):,}")
    print(f"[INFO] samples found: {sorted(node_returns['sample'].dropna().unique().tolist())}")

    if keep_nodes == "leaves":
        if "is_final_leaf" not in metadata.columns:
            raise ValueError("Metadata does not contain is_final_leaf, cannot keep leaves only.")
        keep_cols = metadata.loc[metadata["is_final_leaf"] == True, "column_name"].astype(str).tolist()
        metadata = metadata[metadata["column_name"].isin(keep_cols)].copy()
        node_returns = node_returns[node_returns["column_name"].isin(keep_cols)].copy()
        print(f"[INFO] Kept final leaves only. Remaining metadata rows: {len(metadata)}")
    elif keep_nodes == "all":
        print("[INFO] Keeping all nodes.")
    else:
        raise ValueError("--keep_nodes must be one of: all, leaves")

    sample_counts = (
        node_returns[[DATE_COL, "sample"]]
        .drop_duplicates()
        .groupby("sample")[DATE_COL]
        .nunique()
        .to_dict()
    )
    print(f"[INFO] Month counts by sample before checks: {sample_counts}")

    if enforce_expected_counts:
        train_n = int(sample_counts.get("train", 0))
        valid_n = int(sample_counts.get("valid", 0))
        test_n = int(sample_counts.get("test", 0))
        print(f"[WARNING] train/valid/test months = {train_n}/{valid_n}/{test_n} (expected about 240/120/12)")

    sample_map = (
        node_returns[[DATE_COL, "sample"]]
        .drop_duplicates()
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    wide = node_returns.pivot(index=DATE_COL, columns="column_name", values="ret").sort_index().reset_index()
    wide.columns.name = None
    print(f"[INFO] Wide return matrix shape before RF: {wide.shape}")

    wide_excess = subtract_rf(wide, RF_PATH)
    print(f"[INFO] Wide return matrix shape after RF: {wide_excess.shape}")

    n_missing = int(wide_excess.drop(columns=[DATE_COL], errors="ignore").isna().sum().sum())
    print(f"[INFO] Missing cells after RF subtraction: {n_missing:,}")

    if drop_missing_months:
        before_n = len(wide_excess)
        wide_excess = wide_excess.dropna(axis=0, how="any").reset_index(drop=True)
        after_n = len(wide_excess)
        print(f"[INFO] Dropped months with missing: {before_n - after_n}")

        sample_map = sample_map[sample_map[DATE_COL].isin(wide_excess[DATE_COL])].copy()
        sample_map = sample_map.sort_values(DATE_COL).reset_index(drop=True)

    before_cols = wide_excess.shape[1]
    wide_excess, _ = drop_zero_variance_columns(wide_excess)
    after_cols = wide_excess.shape[1]
    print(f"[INFO] Dropped zero variance columns: {before_cols - after_cols}")

    if wide_excess.shape[1] <= 1:
        raise ValueError("No portfolio columns survived zero-variance filtering. Only date remains.")

    before_cols = wide_excess.shape[1]
    wide_excess, keep_cols = dedup_columns_by_values(wide_excess)
    after_cols = wide_excess.shape[1]
    print(f"[INFO] Dropped duplicated columns: {before_cols - after_cols}")

    metadata_filtered = metadata[metadata["column_name"].isin(set(keep_cols))].copy()
    metadata_filtered["column_name"] = pd.Categorical(
        metadata_filtered["column_name"], categories=keep_cols, ordered=True
    )
    metadata_filtered = metadata_filtered.sort_values("column_name").reset_index(drop=True)
    metadata_filtered["column_name"] = metadata_filtered["column_name"].astype(str)

    ret_cols_final = list(wide_excess.columns[1:])
    meta_cols_final = metadata_filtered["column_name"].tolist()
    if ret_cols_final != meta_cols_final:
        raise ValueError("Combined return columns and metadata are not aligned after filtering.")

    if "depth" in metadata_filtered.columns:
        adj_w = metadata_filtered[["column_name", "depth"]].copy()
        adj_w["adj_w"] = 1.0 / np.sqrt(2.0 ** adj_w["depth"].astype(float))
    else:
        adj_w = pd.DataFrame({"column_name": ret_cols_final, "adj_w": 1.0})

    wide_excess.to_pickle(os.path.join(repeat_output_dir, "level_all_excess_ret_combined.pkl.gz"))
    wide_excess.to_csv(os.path.join(repeat_output_dir, "level_all_excess_ret_combined.csv"), index=False)
    wide_excess.to_parquet(os.path.join(repeat_output_dir, "level_all_excess_ret_combined.parquet"), index=False)

    metadata_filtered.to_csv(os.path.join(repeat_output_dir, "portfolio_metadata.csv"), index=False)
    metadata_filtered.to_parquet(os.path.join(repeat_output_dir, "portfolio_metadata.parquet"), index=False)

    sample_map.to_csv(os.path.join(repeat_output_dir, "sample_map.csv"), index=False)
    sample_map.to_parquet(os.path.join(repeat_output_dir, "sample_map.parquet"), index=False)

    adj_w.to_csv(os.path.join(repeat_output_dir, "adj_w.csv"), index=False)
    adj_w.to_parquet(os.path.join(repeat_output_dir, "adj_w.parquet"), index=False)

    for src in [combo_path, split_rules_path]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(repeat_output_dir, os.path.basename(src)))

    summary = pd.DataFrame([{
        "year": year,
        "repeat_id": repeat_id,
        "keep_nodes": keep_nodes,
        "n_rows": wide_excess.shape[0],
        "n_portfolios": max(wide_excess.shape[1] - 1, 0),
        "n_metadata_rows": metadata_filtered.shape[0],
        "n_missing_cells_final": int(wide_excess.drop(columns=[DATE_COL], errors="ignore").isna().sum().sum()),
    }])
    summary.to_csv(os.path.join(repeat_output_dir, "combine_summary.csv"), index=False)

    print(f"[INFO] Final combined shape: {wide_excess.shape}")
    print(f"[INFO] Metadata shape: {metadata_filtered.shape}")
    print(f"[INFO] Repeat {repeat_id} combine finished successfully")


def main(args):
    year = args.year
    N = args.N
    keep_nodes = args.keep_nodes
    drop_missing_months = args.drop_missing_months
    enforce_expected_counts = args.enforce_expected_counts

    split_input_dir = os.path.join(SPLIT_OUTPUT_BASE_DIR, f"oos_{year}")
    output_dir = os.path.join(COMBINE_OUTPUT_BASE_DIR, f"oos_{year}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] OOS year: {year}")
    print(f"[INFO] Split input dir: {split_input_dir}")
    print(f"[INFO] Combine output dir: {output_dir}")
    print(f"[INFO] N repeats to combine: {N}")

    for repeat_id in range(1, N + 1):
        repeat_output_dir = os.path.join(output_dir, f"repeat_{repeat_id:02d}")
        combine_one_repeat(
            year=year,
            repeat_id=repeat_id,
            split_input_dir=split_input_dir,
            repeat_output_dir=repeat_output_dir,
            keep_nodes=keep_nodes,
            drop_missing_months=drop_missing_months,
            enforce_expected_counts=enforce_expected_counts,
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--keep_nodes", type=str, default="leaves", choices=["all", "leaves"])
    parser.add_argument("--drop_missing_months", action="store_true")
    parser.add_argument("--enforce_expected_counts", action="store_true")
    args = parser.parse_args()
    main(args)
