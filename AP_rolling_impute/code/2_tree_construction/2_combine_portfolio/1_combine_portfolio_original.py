import os
import argparse
import numpy as np
import pandas as pd

# =========================================================
# paths
# =========================================================
SPLIT_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/1_split/output"
COMBINE_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_combine_portfolio/output"
RF_PATH = "../../../data/common/rf.csv"

# fixed columns
DATE_COL = "date"


def list_year_dirs(input_dir: str):
    year_dirs = []
    for name in os.listdir(input_dir):
        full_path = os.path.join(input_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            year_dirs.append(int(name))
    return sorted(year_dirs)


def load_combo_specs(input_dir: str):
    path = os.path.join(input_dir, "combo_specs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find combo specs file: {path}")
    return pd.read_csv(path)


def load_tree_specs(input_dir: str):
    path = os.path.join(input_dir, "tree_specs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find tree specs file: {path}")
    return pd.read_csv(
        path,
        dtype={
            "tree_id": str,
            "sequence_digits": str,
            "sequence_idx": str
        }
    )


def concat_yearly_tree_ret(file_paths):
    dfs = []

    for fp in file_paths:
        if not os.path.exists(fp):
            continue

        df = pd.read_pickle(fp)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        dfs.append(df)

    if len(dfs) == 0:
        return None

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return out


def add_prefix(df: pd.DataFrame, combo_id: str, tree_id: str):
    df = df.copy()
    rename_map = {}

    for c in df.columns:
        if c == DATE_COL:
            rename_map[c] = c
        else:
            rename_map[c] = f"{combo_id}.{tree_id}.{c}"

    return df.rename(columns=rename_map)


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


def dedup_columns_by_values(df: pd.DataFrame):
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    if len(non_date_cols) == 0:
        return df[[DATE_COL]].copy(), []

    tmp = df[non_date_cols].T
    keep_mask = ~tmp.duplicated()

    keep_cols = list(tmp.index[keep_mask])
    out = df[[DATE_COL] + keep_cols].copy()

    return out, keep_cols


def drop_zero_variance_columns(df: pd.DataFrame):
    """
    修复点：
    不再要求整列全是 finite。
    只要 finite 部分至少有 2 个观测，并且 finite 部分标准差 > 0，就保留。
    """
    non_date_cols = [c for c in df.columns if c != DATE_COL]

    keep_cols = []
    for c in non_date_cols:
        x = df[c].to_numpy(dtype=float)
        good = np.isfinite(x)

        # 至少要有两个有效观测，否则方差没意义
        if good.sum() < 2:
            continue

        # 只在有效观测上看标准差
        if np.std(x[good], ddof=0) > 0:
            keep_cols.append(c)

    out = df[[DATE_COL] + keep_cols].copy()
    return out, keep_cols


def get_node_labels_from_depth(depth: int):
    labels = ["1"]
    frontier = ["1"]

    for _ in range(depth):
        new_frontier = []
        for node in frontier:
            left = node + "1"
            right = node + "2"
            labels.extend([left, right])
            new_frontier.extend([left, right])
        frontier = new_frontier

    return labels


def build_column_metadata(combo_specs: pd.DataFrame, tree_specs: pd.DataFrame, keep_cols=None):
    """
    修复点：
    即使 records 为空，也保留列名，避免写出 truly empty csv。
    """
    combo_map = combo_specs.set_index("combo_id").to_dict(orient="index")
    tree_map = tree_specs.set_index("tree_id").to_dict(orient="index")

    records = []

    for combo_id in combo_specs["combo_id"]:
        combo_info = combo_map[combo_id]

        for tree_id in tree_specs["tree_id"]:
            tree_info = tree_map[tree_id]
            sequence_digits = str(tree_info["sequence_digits"])

            node_labels = get_node_labels_from_depth(len(sequence_digits))

            for node_id in node_labels:
                colname = f"{combo_id}.{tree_id}.{node_id}"

                if keep_cols is not None and colname not in keep_cols:
                    continue

                records.append({
                    "column_name": colname,
                    "combo_id": combo_id,
                    "tree_id": tree_id,
                    "node_id": node_id,
                    "sequence_digits": sequence_digits,
                    "feature_1": combo_info["feature_1"],
                    "feature_2": combo_info["feature_2"],
                    "feature_3": combo_info["feature_3"]
                })

    cols = [
        "column_name",
        "combo_id",
        "tree_id",
        "node_id",
        "sequence_digits",
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    return pd.DataFrame(records, columns=cols)


def make_repeat_tag(repeat_idx: int, seed: int, B: int, depth: int):
    return f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"


def combine_one_repeat(split_input_dir: str, output_dir: str, drop_missing_months: bool):
    if not os.path.exists(split_input_dir):
        raise FileNotFoundError(f"Cannot find split output dir: {split_input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    combo_specs = load_combo_specs(split_input_dir)
    tree_specs = load_tree_specs(split_input_dir)
    years = list_year_dirs(split_input_dir)

    if len(years) == 0:
        raise ValueError(f"No year folders found in {split_input_dir}")

    print(f"[INFO] Split input dir: {split_input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Years found: {years[0]} - {years[-1]}")
    print(f"[INFO] Number of combos: {len(combo_specs)}")
    print(f"[INFO] Number of valid trees per combo: {len(tree_specs)}")

    ret_dfs = []

    for _, combo_row in combo_specs.iterrows():
        combo_id = combo_row["combo_id"]

        for _, tree_row in tree_specs.iterrows():
            tree_id = tree_row["tree_id"]

            file_paths = [
                os.path.join(split_input_dir, str(year), combo_id, tree_id, "ret.pkl.gz")
                for year in years
            ]

            ret_df = concat_yearly_tree_ret(file_paths)

            if ret_df is None:
                continue

            ret_df = add_prefix(ret_df, combo_id=combo_id, tree_id=tree_id)
            ret_df = ret_df.set_index(DATE_COL)

            ret_dfs.append(ret_df)

    if len(ret_dfs) == 0:
        raise ValueError("No tree return files found")

    print(f"[INFO] Number of tree return blocks loaded: {len(ret_dfs)}")

    combined_ret = pd.concat(ret_dfs, axis=1, join="outer")
    combined_ret = combined_ret.sort_index().reset_index()

    print(f"[INFO] Combined shape before rf: {combined_ret.shape}")

    combined_excess = subtract_rf(combined_ret, RF_PATH)
    print(f"[INFO] Combined shape after rf: {combined_excess.shape}")

    if drop_missing_months:
        before_n = combined_excess.shape[0]
        combined_excess = combined_excess.dropna(axis=0, how="any").reset_index(drop=True)
        after_n = combined_excess.shape[0]
        print(f"[INFO] Dropped months with missing: {before_n - after_n}")

    before_cols = combined_excess.shape[1]
    combined_excess, _ = drop_zero_variance_columns(combined_excess)
    after_cols = combined_excess.shape[1]
    print(f"[INFO] Dropped zero variance columns: {before_cols - after_cols}")

    # fail fast：如果只剩 date，直接报错，别继续写坏文件
    if combined_excess.shape[1] <= 1:
        raise ValueError(
            f"No portfolio columns survived zero-variance filtering in {split_input_dir}. "
            f"Only date column remains."
        )

    before_cols = combined_excess.shape[1]
    combined_excess, keep_cols = dedup_columns_by_values(combined_excess)
    after_cols = combined_excess.shape[1]
    print(f"[INFO] Dropped duplicated columns: {before_cols - after_cols}")

    metadata = build_column_metadata(combo_specs, tree_specs, keep_cols=keep_cols)

    # 再加一道保险
    if metadata.shape[1] == 0:
        raise ValueError(
            f"Metadata is empty with zero columns in {split_input_dir}. "
            f"This should not happen after the fixes."
        )

    combined_excess.to_pickle(os.path.join(output_dir, "level_all_excess_ret_combined.pkl.gz"))
    combined_excess.to_csv(os.path.join(output_dir, "level_all_excess_ret_combined.csv"), index=False)
    metadata.to_csv(os.path.join(output_dir, "portfolio_metadata.csv"), index=False)

    print(f"[INFO] Final combined shape: {combined_excess.shape}")
    print(f"[INFO] Metadata shape: {metadata.shape}")
    print("[INFO] Combine step finished successfully")


def main(args):
    B = args.B
    depth = args.depth
    N = args.N
    base_seed = args.base_seed
    drop_missing_months = args.drop_missing_months

    if N <= 0:
        raise ValueError("N must be positive.")

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        repeat_tag = make_repeat_tag(repeat_idx, seed, B, depth)

        split_input_dir = os.path.join(SPLIT_OUTPUT_BASE_DIR, repeat_tag)
        output_dir = os.path.join(COMBINE_OUTPUT_BASE_DIR, repeat_tag)

        print("=" * 80)
        print(f"[INFO] Combining repeat {repeat_idx}/{N}")
        print(f"[INFO] Seed used: {seed}")

        combine_one_repeat(
            split_input_dir=split_input_dir,
            output_dir=output_dir,
            drop_missing_months=drop_missing_months
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)

    # N = how many independent repeats were generated in split
    parser.add_argument("--N", type=int, required=True)

    # seed for repeat r is base_seed + r - 1
    parser.add_argument("--base_seed", type=int, default=0)

    parser.add_argument("--drop_missing_months", action="store_true")

    args = parser.parse_args()
    main(args)