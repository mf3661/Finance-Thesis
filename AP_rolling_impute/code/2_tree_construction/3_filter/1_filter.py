import os
import re
import argparse
import numpy as np
import pandas as pd

# =========================================================
# paths
# =========================================================
COMBINE_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_combine_portfolio/output"
FILTER_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/3_filter/output"

# fixed columns
DATE_COL = "date"


def make_repeat_tag(repeat_idx: int, seed: int, B: int, depth: int):
    return f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"


def parse_repeat_tag(repeat_tag: str):
    """
    Parse repeat tag like:
        repeat_01_seed_0_B_10_depth_4
    Returns:
        repeat_idx, seed, B, depth
    """
    pattern = r"^repeat_(\d+)_seed_(-?\d+)_B_(\d+)_depth_(\d+)$"
    m = re.match(pattern, repeat_tag)
    if m is None:
        raise ValueError(f"Invalid repeat tag format: {repeat_tag}")

    repeat_idx = int(m.group(1))
    seed = int(m.group(2))
    B = int(m.group(3))
    depth = int(m.group(4))
    return repeat_idx, seed, B, depth


def drop_sparse_columns(df: pd.DataFrame, min_valid_frac: float):
    """
    Drop columns whose fraction of non-missing observations is below min_valid_frac.
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]
    n_rows = len(df)

    keep_cols = []
    for c in non_date_cols:
        valid_frac = df[c].notna().sum() / n_rows
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
    Since sparse columns should already be mostly removed,
    the remaining NaN issue should be much milder.
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    if len(non_date_cols) == 0:
        return df[[DATE_COL]].copy(), []

    tmp = df[non_date_cols].T
    keep_mask = ~tmp.duplicated()

    keep_cols = list(tmp.index[keep_mask])
    out = df[[DATE_COL] + keep_cols].copy()

    return out, keep_cols


def ensure_metadata_columns(metadata: pd.DataFrame):
    expected_cols = [
        "column_name",
        "combo_id",
        "tree_id",
        "node_id",
        "sequence_digits",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    for c in expected_cols:
        if c not in metadata.columns:
            metadata[c] = pd.Series(dtype="object")
    return metadata[expected_cols + [c for c in metadata.columns if c not in expected_cols]]


def filter_one_repeat(input_dir: str, output_dir: str, repeat_tag: str, min_valid_frac: float):
    os.makedirs(output_dir, exist_ok=True)

    ret_path = os.path.join(input_dir, "level_all_excess_ret_combined.pkl.gz")
    meta_path = os.path.join(input_dir, "portfolio_metadata.csv")

    if not os.path.exists(ret_path):
        raise FileNotFoundError(f"Cannot find return file: {ret_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find metadata file: {meta_path}")

    # 先读 return
    ret_df = pd.read_pickle(ret_path).copy()

    # metadata 文件可能是空文件，这里做更友好的处理
    if os.path.getsize(meta_path) == 0:
        metadata = pd.DataFrame(columns=[
            "column_name", "combo_id", "tree_id", "node_id",
            "sequence_digits", "feature_1", "feature_2", "feature_3"
        ])
    else:
        metadata = pd.read_csv(meta_path).copy()

    metadata = ensure_metadata_columns(metadata)

    repeat_idx, seed, B, depth = parse_repeat_tag(repeat_tag)

    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Return shape before filtering: {ret_df.shape}")
    print(f"[INFO] Metadata shape before filtering: {metadata.shape}")
    print(f"[INFO] min_valid_frac: {min_valid_frac}")

    # =====================================================
    # 1. drop sparse columns
    # =====================================================
    before_cols = ret_df.shape[1]
    ret_df_sparse, keep_cols_sparse = drop_sparse_columns(ret_df, min_valid_frac=min_valid_frac)
    after_cols = ret_df_sparse.shape[1]
    print(f"[INFO] Dropped sparse columns: {before_cols - after_cols}")

    # =====================================================
    # 2. drop zero variance columns
    # =====================================================
    before_cols = ret_df_sparse.shape[1]
    ret_df_var, keep_cols_var = drop_zero_variance_columns(ret_df_sparse)
    after_cols = ret_df_var.shape[1]
    print(f"[INFO] Dropped zero variance columns: {before_cols - after_cols}")

    # fail fast
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

    # add explicit repeat-level identifiers
    metadata_filtered["repeat_idx"] = repeat_idx
    metadata_filtered["seed"] = seed
    metadata_filtered["B"] = B
    metadata_filtered["depth"] = depth

    preferred_order = [
        "repeat_idx", "seed", "B", "depth",
        "column_name", "combo_id", "tree_id", "node_id",
        "sequence_digits", "feature_1", "feature_2", "feature_3"
    ]
    existing = [c for c in preferred_order if c in metadata_filtered.columns]
    others = [c for c in metadata_filtered.columns if c not in existing]
    metadata_filtered = metadata_filtered[existing + others]

    ret_filtered.to_pickle(os.path.join(output_dir, "level_all_excess_ret_combined_filtered.pkl.gz"))
    ret_filtered.to_csv(os.path.join(output_dir, "level_all_excess_ret_combined_filtered.csv"), index=False)
    metadata_filtered.to_csv(os.path.join(output_dir, "portfolio_metadata_filtered.csv"), index=False)

    print(f"[INFO] Return shape after filtering: {ret_filtered.shape}")
    print(f"[INFO] Metadata shape after filtering: {metadata_filtered.shape}")
    print("[INFO] Filter step finished successfully")


def main(args):
    B = args.B
    depth = args.depth
    N = args.N
    base_seed = args.base_seed
    min_valid_frac = args.min_valid_frac

    if N <= 0:
        raise ValueError("N must be positive.")

    if not (0 < min_valid_frac <= 1):
        raise ValueError("min_valid_frac must be in (0, 1].")

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        repeat_tag = make_repeat_tag(repeat_idx, seed, B, depth)

        input_dir = os.path.join(COMBINE_OUTPUT_BASE_DIR, repeat_tag)
        output_dir = os.path.join(FILTER_OUTPUT_BASE_DIR, repeat_tag)

        print("=" * 80)
        print(f"[INFO] Filtering repeat {repeat_idx}/{N}")
        print(f"[INFO] Seed used: {seed}")

        filter_one_repeat(
            input_dir=input_dir,
            output_dir=output_dir,
            repeat_tag=repeat_tag,
            min_valid_frac=min_valid_frac
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)

    # N = how many independent repeats were generated upstream
    parser.add_argument("--N", type=int, required=True)

    # seed for repeat r is base_seed + r - 1
    parser.add_argument("--base_seed", type=int, default=0)

    # 新增：至少多少比例的月份非缺失才保留该 portfolio
    parser.add_argument("--min_valid_frac", type=float, default=0.8)

    args = parser.parse_args()
    main(args)