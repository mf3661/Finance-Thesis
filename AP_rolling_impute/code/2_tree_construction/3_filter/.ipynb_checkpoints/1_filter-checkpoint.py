import os
import re
import argparse
import pandas as pd

# =========================================================
# paths
# =========================================================
COMBINE_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_combine_portfolio/output"
FILTER_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/3_filter/output"

# fixed columns
DATE_COL = "date"


def dedup_columns_by_values(df: pd.DataFrame):
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    tmp = df[non_date_cols].T
    keep_mask = ~tmp.duplicated()

    keep_cols = list(tmp.index[keep_mask])
    out = df[[DATE_COL] + keep_cols].copy()

    return out, keep_cols


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


def filter_one_repeat(input_dir: str, output_dir: str, repeat_tag: str):
    os.makedirs(output_dir, exist_ok=True)

    ret_path = os.path.join(input_dir, "level_all_excess_ret_combined.pkl.gz")
    meta_path = os.path.join(input_dir, "portfolio_metadata.csv")

    if not os.path.exists(ret_path):
        raise FileNotFoundError(f"Cannot find return file: {ret_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Cannot find metadata file: {meta_path}")

    ret_df = pd.read_pickle(ret_path).copy()
    metadata = pd.read_csv(meta_path).copy()

    repeat_idx, seed, B, depth = parse_repeat_tag(repeat_tag)

    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Return shape before dedup: {ret_df.shape}")
    print(f"[INFO] Metadata shape before dedup: {metadata.shape}")

    ret_filtered, keep_cols = dedup_columns_by_values(ret_df)

    keep_col_set = set(keep_cols)
    metadata_filtered = metadata[metadata["column_name"].isin(keep_col_set)].copy()

    metadata_filtered["column_name"] = pd.Categorical(
        metadata_filtered["column_name"],
        categories=keep_cols,
        ordered=True
    )
    metadata_filtered = metadata_filtered.sort_values("column_name").reset_index(drop=True)
    metadata_filtered["column_name"] = metadata_filtered["column_name"].astype(str)

    # add explicit repeat-level identifiers for downstream reconstruction/debugging
    metadata_filtered["repeat_idx"] = repeat_idx
    metadata_filtered["seed"] = seed
    metadata_filtered["B"] = B
    metadata_filtered["depth"] = depth

    # reorder columns a bit more nicely if these fields exist
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

    print(f"[INFO] Return shape after dedup: {ret_filtered.shape}")
    print(f"[INFO] Metadata shape after dedup: {metadata_filtered.shape}")
    print("[INFO] Filter step finished successfully")


def main(args):
    B = args.B
    depth = args.depth
    N = args.N
    base_seed = args.base_seed

    if N <= 0:
        raise ValueError("N must be positive.")

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
            repeat_tag=repeat_tag
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

    args = parser.parse_args()
    main(args)