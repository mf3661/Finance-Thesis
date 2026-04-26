import os
import math
import itertools
import argparse
import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
INPUT_DIR = "../../../data/1_data_processing/1_rank_data/output/by_year"
CHARS_PATH = "../../../data/1_data_processing/1_rank_data/input/chars_summary.csv"
OUTPUT_BASE_DIR = "../../../data/2_tree_construction/1_split/output"

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
    """
    Return all node labels in a full binary tree up to the given split depth.
    Root is "1". Each split appends "1" or "2".
    """
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


def weighted_return(x: pd.DataFrame, ret_col: str, weight_col: str):
    if x.empty:
        return np.nan

    w = x[weight_col].astype(float).to_numpy()
    r = x[ret_col].astype(float).to_numpy()

    good = np.isfinite(w) & np.isfinite(r)
    if good.sum() == 0:
        return np.nan

    w = w[good]
    r = r[good]

    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan

    return np.dot(w, r) / w_sum


def compute_node_stats(node_df: pd.DataFrame, split_features, ret_col, weight_col):
    out = {"ret": weighted_return(node_df, ret_col, weight_col)}
    for f in split_features:
        out[f"{f}_min"] = np.nan if node_df.empty else node_df[f].min()
        out[f"{f}_max"] = np.nan if node_df.empty else node_df[f].max()
    return out


def split_node_half(node_df: pd.DataFrame, feature_col: str, id_col: str):
    """
    Split one node into two halves by sorting on feature_col then id_col.
    Left gets ceil(n/2), right gets the rest.
    """
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
    """
    Randomly sample B combos:
        feature_1 = size_rank
        feature_2, feature_3 = two non-size ranked features
    """
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

    combo_specs = pd.DataFrame(records)
    return combo_specs


def build_tree_specs(depth: int):
    """
    Enumerate all split sequences of length depth using feature index {0,1,2},
    excluding degenerate sequences where all layers use the same feature.
    """
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

    tree_specs = pd.DataFrame(valid_records)
    return tree_specs


def read_year_data(year: int, combo_specs_all: list, input_dir: str):
    """
    Read one year of ranked data, only keeping the ranked features that appear
    in any repeat's combo_specs, to reduce I/O.
    """
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
    """
    For one month and one tree:
    1. compute node stats for all nodes
    2. record final leaf assignment for each stock

    Returns
    -------
    node_data : dict
        Node-level return/min/max stats
    leaf_df : pd.DataFrame
        Columns: date, permno, leaf_id
    """
    node_data = {}
    frontier = {"1": month_df.copy()}

    # root stats
    node_data["1"] = compute_node_stats(month_df, split_features, ret_col, weight_col)

    # recursively split
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

    # after the final split, frontier contains only leaves
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
    """
    Build:
    1. node return time series
    2. node feature min/max time series
    3. leaf membership table for all stock-month observations
    """
    all_node_labels = get_node_labels(depth)
    all_dates = np.sort(df_combo[DATE_COL].dropna().unique())

    month_groups = {dt: g.copy() for dt, g in df_combo.groupby(DATE_COL)}

    ret_records = []
    min_records = {f: [] for f in split_features}
    max_records = {f: [] for f in split_features}
    leaf_dfs = []

    for dt in all_dates:
        month_df = month_groups[dt]

        node_stats, leaf_df = process_one_month_for_one_tree(
            month_df=month_df,
            split_sequence=split_sequence,
            split_features=split_features,
            ret_col=RET_COL,
            weight_col=WEIGHT_COL,
            id_col=ID_COL,
            date_col=DATE_COL,
        )

        # node returns
        ret_row = {label: node_stats.get(label, {}).get("ret", np.nan) for label in all_node_labels}
        ret_row[DATE_COL] = dt
        ret_records.append(ret_row)

        # node min/max
        for f in split_features:
            min_row = {label: node_stats.get(label, {}).get(f"{f}_min", np.nan) for label in all_node_labels}
            max_row = {label: node_stats.get(label, {}).get(f"{f}_max", np.nan) for label in all_node_labels}
            min_row[DATE_COL] = dt
            max_row[DATE_COL] = dt
            min_records[f].append(min_row)
            max_records[f].append(max_row)

        # leaf membership
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


def save_tree_outputs(tree_dir, ret_df, feature_minmax, split_features, leaf_membership_df):
    """
    Save:
    - ret.pkl.gz
    - feature min/max pkl.gz
    - leaf_membership.parquet
    """
    os.makedirs(tree_dir, exist_ok=True)

    ret_df.to_pickle(os.path.join(tree_dir, "ret.pkl.gz"))

    for f in split_features:
        short_name = feature_short_name(f)
        feature_minmax[f"{f}_min"].to_pickle(os.path.join(tree_dir, f"{short_name}_min.pkl.gz"))
        feature_minmax[f"{f}_max"].to_pickle(os.path.join(tree_dir, f"{short_name}_max.pkl.gz"))

    leaf_membership_df.to_parquet(
        os.path.join(tree_dir, "leaf_membership.parquet"),
        index=False
    )


def run_one_repeat(
    repeat_idx: int,
    seed: int,
    year: int,
    B: int,
    depth: int,
    df_year: pd.DataFrame,
    combo_specs: pd.DataFrame,
    tree_specs: pd.DataFrame,
):
    """
    Run one independent repeat for one year.
    """
    repeat_tag = f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"
    output_dir = os.path.join(OUTPUT_BASE_DIR, repeat_tag)
    os.makedirs(output_dir, exist_ok=True)

    combo_specs.to_csv(os.path.join(output_dir, "combo_specs.csv"), index=False)
    tree_specs.to_csv(os.path.join(output_dir, "tree_specs.csv"), index=False)

    year_dir = os.path.join(output_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    combo_specs.to_csv(os.path.join(year_dir, "combo_specs.csv"), index=False)
    tree_specs.to_csv(os.path.join(year_dir, "tree_specs.csv"), index=False)

    print("=" * 80)
    print(f"[INFO] Repeat: {repeat_idx}/{args.N}")
    print(f"[INFO] Seed used: {seed}")
    print(f"[INFO] Year: {year}")
    print(f"[INFO] Input shape: {df_year.shape}")
    print(f"[INFO] Number of combos: {len(combo_specs)}")
    print(f"[INFO] Number of valid trees per combo: {len(tree_specs)}")
    print(f"[INFO] Output dir: {output_dir}")

    for _, combo_row in combo_specs.iterrows():
        combo_id = combo_row["combo_id"]
        split_features = [
            combo_row["feature_1"],
            combo_row["feature_2"],
            combo_row["feature_3"]
        ]

        combo_dir = os.path.join(year_dir, combo_id)
        os.makedirs(combo_dir, exist_ok=True)

        combo_meta = pd.DataFrame([{
            "repeat_idx": repeat_idx,
            "seed": seed,
            "combo_id": combo_id,
            "year": year,
            "feature_1": split_features[0],
            "feature_2": split_features[1],
            "feature_3": split_features[2],
            "feature_1_short": feature_short_name(split_features[0]),
            "feature_2_short": feature_short_name(split_features[1]),
            "feature_3_short": feature_short_name(split_features[2]),
            "B": B,
            "depth": depth
        }])
        combo_meta.to_csv(os.path.join(combo_dir, "combo_meta.csv"), index=False)
        tree_specs.to_csv(os.path.join(combo_dir, "tree_specs.csv"), index=False)

        needed_cols = [DATE_COL, ID_COL, RET_COL, WEIGHT_COL] + split_features
        df_combo = df_year[needed_cols].dropna(subset=needed_cols).copy()

        print(f"[INFO] Repeat {repeat_idx} | {combo_id} | features: {split_features} | shape after dropna: {df_combo.shape}")

        if df_combo.empty:
            print(f"[WARNING] Repeat {repeat_idx} | {combo_id} has no valid rows after dropna, skip.")
            continue

        for _, tree_row in tree_specs.iterrows():
            tree_id = tree_row["tree_id"]
            split_sequence = [int(x) for x in tree_row["sequence_idx"].split("|")]

            tree_dir = os.path.join(combo_dir, tree_id)

            ret_df, feature_minmax, leaf_membership_df = build_one_tree_all_months(
                df_combo=df_combo,
                split_features=split_features,
                split_sequence=split_sequence,
                depth=depth
            )

            save_tree_outputs(
                tree_dir=tree_dir,
                ret_df=ret_df,
                feature_minmax=feature_minmax,
                split_features=split_features,
                leaf_membership_df=leaf_membership_df
            )

        print(f"[INFO] Repeat {repeat_idx} finished {combo_id}")

    print(f"[INFO] Repeat {repeat_idx} finished successfully")


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

    # Pre-build combo specs for all repeats so we can read the year file once
    combo_specs_all = []
    tree_specs = build_tree_specs(depth=depth)

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs = build_combo_specs(seed=seed, B=B, chars_path=CHARS_PATH)
        combo_specs_all.append(combo_specs)

    # Read year data once, keeping only all ranked features used across all repeats
    df_year = read_year_data(
        year=year,
        combo_specs_all=combo_specs_all,
        input_dir=INPUT_DIR
    )

    # Run repeats one by one
    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs = combo_specs_all[repeat_idx - 1]

        run_one_repeat(
            repeat_idx=repeat_idx,
            seed=seed,
            year=year,
            B=B,
            depth=depth,
            df_year=df_year,
            combo_specs=combo_specs,
            tree_specs=tree_specs,
        )

    print("=" * 80)
    print("[INFO] All repeats finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)

    # N = how many independent B-combo runs to do internally
    parser.add_argument("--N", type=int, required=True)

    # seed for repeat r is base_seed + r - 1
    parser.add_argument("--base_seed", type=int, default=0)

    args = parser.parse_args()
    main(args)