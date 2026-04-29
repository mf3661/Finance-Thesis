#!/apps/anaconda3/bin/python3
import os
import math
import itertools
import argparse
import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
INPUT_DIR = "/user/hd2570/AP_rolling_impute/data/1_data_processing/1_rank_data/output/by_year"
CHARS_PATH = "/user/hd2570/AP_rolling_impute/data/1_data_processing/1_rank_data/input/chars_summary.csv"
OUTPUT_BASE_DIR = "/user/hd2570/AP_rolling_impute/data/2_tree_construction/1_split/output"

# fixed columns
DATE_COL = "date"
ID_COL = "permno"
RET_COL = "ret"
WEIGHT_COL = "size"
SIZE_RANK_COL = "size_rank"


# =========================================================
# helpers
# =========================================================
def get_node_labels(depth: int):
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


# def weighted_return(x: pd.DataFrame, ret_col: str, weight_col: str):
#     if x.empty:
#         return np.nan

#     w = x[weight_col].astype(float).to_numpy()
#     r = x[ret_col].astype(float).to_numpy()

#     good = np.isfinite(w) & np.isfinite(r)
#     if good.sum() == 0:
#         return np.nan

#     w = w[good]
#     r = r[good]

#     w_sum = w.sum()
#     if w_sum <= 0:
#         return np.nan

#     return np.dot(w, r) / w_sum

def weighted_return(x: pd.DataFrame, ret_col: str, weight_col: str):
    if x.empty:
        return np.nan

    r = x[ret_col].astype(float).to_numpy()
    good = np.isfinite(r)
    if good.sum() == 0:
        return np.nan

    r = r[good]
    w = np.ones(len(r), dtype=float)
    return np.dot(w, r) / w.sum()

def compute_node_stats(node_df: pd.DataFrame, split_features, ret_col, weight_col):
    out = {"ret": weighted_return(node_df, ret_col, weight_col)}
    for f in split_features:
        out[f"{f}_min"] = np.nan if node_df.empty else node_df[f].min()
        out[f"{f}_max"] = np.nan if node_df.empty else node_df[f].max()
    return out


def split_node_half(node_df: pd.DataFrame, feature_col: str, id_col: str):
    n = len(node_df)

    if n == 0:
        return node_df.copy(), node_df.copy()
    if n == 1:
        return node_df.copy(), node_df.iloc[0:0].copy()

    tmp = node_df.sort_values([feature_col, id_col], ascending=[True, True]).copy()
    left_n = math.ceil(n / 2)

    left_df = tmp.iloc[:left_n].copy()
    right_df = tmp.iloc[left_n:].copy()

    return left_df, right_df


def feature_short_name(feat: str):
    return feat.replace("_rank", "")


def build_combo_specs(seed: int, B: int, chars_path: str):
    chars = pd.read_csv(chars_path)
    acronyms = chars["Acronym"].astype(str).str.rstrip().tolist()

    non_size_features = [f"{c}_rank" for c in acronyms if c != "size"]
    all_pairs = list(itertools.combinations(non_size_features, 2))

    if B > len(all_pairs):
        raise ValueError(f"B={B} is larger than the number of unique pairs: {len(all_pairs)}")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(all_pairs), size=B, replace=False)

    records = []
    for i, idx in enumerate(chosen_idx, start=1):
        f2, f3 = all_pairs[idx]
        records.append({
            "combo_id": f"combo_{i:04d}",
            "feature_1": SIZE_RANK_COL,
            "feature_2": f2,
            "feature_3": f3
        })

    return pd.DataFrame(records)


def build_tree_specs(depth: int):
    all_sequences = list(itertools.product(range(3), repeat=depth))

    valid_records = []
    for seq in all_sequences:
        if len(set(seq)) == 1:
            continue

        seq_digits = "".join(str(x + 1) for x in seq)
        valid_records.append({
            "tree_id": f"tree_{seq_digits}",
            "sequence_digits": seq_digits,
            "sequence_idx": "|".join(str(x) for x in seq)
        })

    return pd.DataFrame(valid_records)


def read_year_data(year: int, combo_specs_all: list, input_dir: str):
    year_path = os.path.join(input_dir, f"{year}.parquet")
    if not os.path.exists(year_path):
        raise FileNotFoundError(f"Cannot find input file: {year_path}")

    used_rank_features = set()
    for combo_specs in combo_specs_all:
        used_rank_features.update(combo_specs["feature_1"].tolist())
        used_rank_features.update(combo_specs["feature_2"].tolist())
        used_rank_features.update(combo_specs["feature_3"].tolist())

    used_rank_features = sorted(used_rank_features)

    cols_to_read = [ID_COL, DATE_COL, RET_COL, WEIGHT_COL] + used_rank_features
    df = pd.read_parquet(year_path, columns=cols_to_read)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return df


def _downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def combo_outputs_exist(year_dir: str, combo_id: str) -> bool:
    ret_path = os.path.join(year_dir, f"{combo_id}_ret.parquet")
    leaf_path = os.path.join(year_dir, f"{combo_id}_leaf.parquet")
    minmax_path = os.path.join(year_dir, f"{combo_id}_minmax.pkl.gz")
    return os.path.exists(ret_path) and os.path.exists(leaf_path) and os.path.exists(minmax_path)


# =========================================================
# core logic
# =========================================================
def process_one_month_for_one_tree(
    month_df,
    split_sequence,
    split_features,
    ret_col,
    weight_col,
    id_col,
    date_col
):
    node_data = {}
    frontier = {"1": month_df.copy()}

    node_data["1"] = compute_node_stats(month_df, split_features, ret_col, weight_col)

    for feat_idx in split_sequence:
        feat_col = split_features[feat_idx]
        new_frontier = {}

        for node_label, node_df in frontier.items():
            left_df, right_df = split_node_half(node_df, feat_col, id_col)

            left_label = node_label + "1"
            right_label = node_label + "2"

            node_data[left_label] = compute_node_stats(left_df, split_features, ret_col, weight_col)
            node_data[right_label] = compute_node_stats(right_df, split_features, ret_col, weight_col)

            new_frontier[left_label] = left_df
            new_frontier[right_label] = right_df

        frontier = new_frontier

    leaf_records = []
    for leaf_label, leaf_node_df in frontier.items():
        if leaf_node_df.empty:
            continue

        tmp = leaf_node_df[[date_col, id_col]].copy()
        tmp["leaf_id"] = leaf_label
        leaf_records.append(tmp)

    if len(leaf_records) == 0:
        leaf_df = pd.DataFrame(columns=[date_col, id_col, "leaf_id"])
    else:
        leaf_df = pd.concat(leaf_records, axis=0, ignore_index=True)
        leaf_df = leaf_df.sort_values([date_col, id_col]).reset_index(drop=True)

    return node_data, leaf_df


def build_one_tree_all_months(df_combo, split_features, split_sequence, depth):
    all_node_labels = get_node_labels(depth)
    all_dates = pd.to_datetime(np.sort(df_combo[DATE_COL].dropna().unique()))

    df_combo = df_combo.copy()
    df_combo[DATE_COL] = pd.to_datetime(df_combo[DATE_COL])
    month_groups = {pd.Timestamp(dt): g.copy() for dt, g in df_combo.groupby(DATE_COL)}

    ret_records = []
    min_records = {f: [] for f in split_features}
    max_records = {f: [] for f in split_features}
    leaf_dfs = []

    for dt in all_dates:
        dt_key = pd.Timestamp(dt)
        month_df = month_groups[dt_key]

        node_stats, leaf_df = process_one_month_for_one_tree(
            month_df=month_df,
            split_sequence=split_sequence,
            split_features=split_features,
            ret_col=RET_COL,
            weight_col=WEIGHT_COL,
            id_col=ID_COL,
            date_col=DATE_COL,
        )

        ret_row = {label: node_stats.get(label, {}).get("ret", np.nan) for label in all_node_labels}
        ret_row[DATE_COL] = dt_key
        ret_records.append(ret_row)

        for f in split_features:
            min_row = {label: node_stats.get(label, {}).get(f"{f}_min", np.nan) for label in all_node_labels}
            max_row = {label: node_stats.get(label, {}).get(f"{f}_max", np.nan) for label in all_node_labels}
            min_row[DATE_COL] = dt_key
            max_row[DATE_COL] = dt_key
            min_records[f].append(min_row)
            max_records[f].append(max_row)

        if not leaf_df.empty:
            leaf_dfs.append(leaf_df)

    ret_df = pd.DataFrame(ret_records).sort_values(DATE_COL).reset_index(drop=True)

    feature_minmax = {}
    for f in split_features:
        feature_minmax[f"{f}_min"] = pd.DataFrame(min_records[f]).sort_values(DATE_COL).reset_index(drop=True)
        feature_minmax[f"{f}_max"] = pd.DataFrame(max_records[f]).sort_values(DATE_COL).reset_index(drop=True)

    if len(leaf_dfs) == 0:
        leaf_membership_df = pd.DataFrame(columns=[DATE_COL, ID_COL, "leaf_id"])
    else:
        leaf_membership_df = pd.concat(leaf_dfs, axis=0, ignore_index=True)
        leaf_membership_df = leaf_membership_df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return ret_df, feature_minmax, leaf_membership_df


def save_combo_outputs(year_dir, combo_id, ret_df, minmax_payload, leaf_membership_df):
    os.makedirs(year_dir, exist_ok=True)

    ret_path = os.path.join(year_dir, f"{combo_id}_ret.parquet")
    leaf_path = os.path.join(year_dir, f"{combo_id}_leaf.parquet")
    minmax_path = os.path.join(year_dir, f"{combo_id}_minmax.pkl.gz")

    ret_out = _downcast_numeric_columns(ret_df)
    ret_out.to_parquet(
        ret_path,
        compression="zstd",
        index=False
    )

    leaf_out = _downcast_numeric_columns(leaf_membership_df)
    leaf_out.to_parquet(
        leaf_path,
        compression="zstd",
        index=False
    )

    pd.to_pickle(minmax_payload, minmax_path, compression="gzip")


def run_one_repeat(
    repeat_idx: int,
    seed: int,
    year: int,
    B: int,
    depth: int,
    N: int,
    df_year: pd.DataFrame,
    combo_specs: pd.DataFrame,
    tree_specs: pd.DataFrame,
):
    repeat_tag = f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, repeat_tag)
    os.makedirs(output_dir, exist_ok=True)

    combo_specs.to_csv(os.path.join(output_dir, "combo_specs.csv"), index=False)
    tree_specs.to_csv(os.path.join(output_dir, "tree_specs.csv"), index=False)

    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    print("=" * 80)
    print(f"[INFO] Repeat: {repeat_idx}/{N}")
    print(f"[INFO] Seed used: {seed}")
    print(f"[INFO] Year: {year}")
    print(f"[INFO] Input shape: {df_year.shape}")
    print(f"[INFO] Number of combos: {len(combo_specs)}")
    print(f"[INFO] Number of valid trees per combo: {len(tree_specs)}")
    print(f"[INFO] Output dir: {output_dir}", flush=True)

    for _, combo_row in combo_specs.iterrows():
        combo_id = combo_row["combo_id"]
        split_features = [
            combo_row["feature_1"],
            combo_row["feature_2"],
            combo_row["feature_3"]
        ]

        if combo_outputs_exist(year_dir, combo_id):
            print(f"[INFO] Repeat {repeat_idx} | {combo_id} already exists, skip.", flush=True)
            continue

        needed_cols = [DATE_COL, ID_COL, RET_COL, WEIGHT_COL] + split_features
        df_combo = df_year[needed_cols].dropna(subset=needed_cols).copy()

        print(
            f"[INFO] Repeat {repeat_idx} | {combo_id} | features: {split_features} | shape after dropna: {df_combo.shape}",
            flush=True
        )

        if df_combo.empty:
            print(f"[WARNING] Repeat {repeat_idx} | {combo_id} has no valid rows after dropna, skip.", flush=True)
            continue

        combo_ret_list = []
        combo_leaf_list = []
        combo_minmax_list = []

        for _, tree_row in tree_specs.iterrows():
            tree_id = tree_row["tree_id"]
            split_sequence = [int(x) for x in tree_row["sequence_idx"].split("|")]

            ret_df, feature_minmax, leaf_membership_df = build_one_tree_all_months(
                df_combo=df_combo,
                split_features=split_features,
                split_sequence=split_sequence,
                depth=depth
            )

            ret_df = ret_df.copy()
            ret_df["tree_id"] = tree_id
            ret_df["combo_id"] = combo_id
            combo_ret_list.append(ret_df)

            leaf_membership_df = leaf_membership_df.copy()
            leaf_membership_df["tree_id"] = tree_id
            leaf_membership_df["combo_id"] = combo_id
            combo_leaf_list.append(leaf_membership_df)

            combo_minmax_list.append({
                "tree_id": tree_id,
                "split_sequence": split_sequence,
                "split_features": list(split_features),
                "feature_minmax": feature_minmax
            })

        if not combo_ret_list:
            print(f"[WARNING] Repeat {repeat_idx} | {combo_id} has no valid tree outputs, skip.", flush=True)
            continue

        combo_ret_df = pd.concat(combo_ret_list, axis=0, ignore_index=True)

        if combo_leaf_list:
            combo_leaf_df = pd.concat(combo_leaf_list, axis=0, ignore_index=True)
            combo_leaf_df = combo_leaf_df.sort_values([DATE_COL, ID_COL, "tree_id"]).reset_index(drop=True)
        else:
            combo_leaf_df = pd.DataFrame(columns=[DATE_COL, ID_COL, "leaf_id", "tree_id", "combo_id"])

        save_combo_outputs(
            year_dir=year_dir,
            combo_id=combo_id,
            ret_df=combo_ret_df,
            minmax_payload=combo_minmax_list,
            leaf_membership_df=combo_leaf_df
        )

        print(f"[INFO] Repeat {repeat_idx} finished {combo_id}", flush=True)

    print(f"[INFO] Repeat {repeat_idx} finished successfully", flush=True)


# =========================================================
# main
# =========================================================
def main(args):
    year = args.year
    B = args.B
    depth = args.depth
    N = args.N
    base_seed = args.base_seed

    if N <= 0:
        raise ValueError("N must be positive.")

    combo_specs_all = []
    tree_specs = build_tree_specs(depth=depth)

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs = build_combo_specs(seed=seed, B=B, chars_path=CHARS_PATH)
        combo_specs_all.append(combo_specs)

    df_year = read_year_data(
        year=year,
        combo_specs_all=combo_specs_all,
        input_dir=INPUT_DIR
    )

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs = combo_specs_all[repeat_idx - 1]

        run_one_repeat(
            repeat_idx=repeat_idx,
            seed=seed,
            year=year,
            B=B,
            depth=depth,
            N=N,
            df_year=df_year,
            combo_specs=combo_specs,
            tree_specs=tree_specs,
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--base_seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
