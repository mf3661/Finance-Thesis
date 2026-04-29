import os
import argparse
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =========================================================
# paths
# =========================================================
INPUT_DIR = "../../../data/feature_construction/1_4_rank_feature/by_year/no_impute"
CHARS_PATH = "../../../data/common/chars_summary.csv"
OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_1_split/2_1_5_split_ptree/output"

# fixed columns
DATE_COL = "date"
ID_COL = "permno"
RET_COL = "ret"
WEIGHT_COL = "size"
SIZE_RANK_COL = "size_rank"

# split defaults
DEFAULT_CUTOFFS = (1.0 / 3.0, 2.0 / 3.0)


# =========================================================
# safe writers
# =========================================================
def atomic_write_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet", dir=os.path.dirname(path)) as tmp:
        tmp_path = tmp.name
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def atomic_write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir=os.path.dirname(path)) as tmp:
        tmp_path = tmp.name
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


# =========================================================
# helpers
# =========================================================
def feature_short_name(feat: str) -> str:
    return feat.replace("_rank", "")


def weighted_return(df: pd.DataFrame, ret_col: str = RET_COL, weight_col: str = WEIGHT_COL) -> float:
    if df.empty:
        return np.nan

    w = df[weight_col].astype(float).to_numpy()
    r = df[ret_col].astype(float).to_numpy()
    good = np.isfinite(w) & np.isfinite(r)
    if good.sum() == 0:
        return np.nan

    w = w[good]
    r = r[good]
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan
    return float(np.dot(w, r) / w_sum)


def sharpe_from_returns(ret_mat: np.ndarray, ridge_gamma: float = 1e-4) -> float:
    """
    Tangency Sharpe on portfolio-level returns.
    ret_mat shape: T x K
    """
    if ret_mat.ndim != 2 or ret_mat.shape[0] < 2 or ret_mat.shape[1] < 1:
        return np.nan

    good_cols = np.isfinite(ret_mat).all(axis=0)
    ret_mat = ret_mat[:, good_cols]
    if ret_mat.shape[1] == 0:
        return np.nan

    mu = ret_mat.mean(axis=0)
    if ret_mat.shape[1] == 1:
        f = ret_mat[:, 0]
        sd = np.std(f, ddof=1)
        if sd <= 1e-12:
            return np.nan
        return float(np.mean(f) / sd)

    sigma = np.cov(ret_mat, rowvar=False)
    sigma = np.atleast_2d(sigma)
    sigma = sigma + ridge_gamma * np.eye(sigma.shape[0])

    try:
        w = np.linalg.solve(sigma, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(sigma) @ mu

    denom = np.sum(np.abs(w))
    if denom <= 1e-12:
        return np.nan
    w = w / denom

    f = ret_mat @ w
    sd = np.std(f, ddof=1)
    if sd <= 1e-12:
        return np.nan
    return float(np.mean(f) / sd)


def build_combo_specs(seed: int, B: int, chars_path: str, total_features: int = 6) -> pd.DataFrame:
    """
    Build B random feature subsets.
    feature_1 is always size_rank.
    total_features includes size_rank.
    """
    if total_features < 2:
        raise ValueError("total_features must be >= 2")

    chars = pd.read_csv(chars_path)
    acronyms = chars["Acronym"].astype(str).str.rstrip().tolist()
    non_size_features = [f"{c}_rank" for c in acronyms if c != "size"]

    choose_k = total_features - 1
    if choose_k > len(non_size_features):
        raise ValueError(f"Need {choose_k} non-size features but only have {len(non_size_features)}")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(non_size_features))

    records = []
    seen = set()
    attempts = 0
    max_attempts = max(B * 20, 1000)

    while len(records) < B and attempts < max_attempts:
        attempts += 1
        chosen_idx = tuple(sorted(rng.choice(all_indices, size=choose_k, replace=False).tolist()))
        if chosen_idx in seen:
            continue
        seen.add(chosen_idx)
        chosen_feats = [non_size_features[i] for i in chosen_idx]
        row = {"combo_id": f"combo_{len(records)+1:04d}"}
        feats = [SIZE_RANK_COL] + chosen_feats
        for j, feat in enumerate(feats, start=1):
            row[f"feature_{j}"] = feat
            row[f"feature_{j}_short"] = feature_short_name(feat)
        records.append(row)

    if len(records) < B:
        raise ValueError(f"Could only generate {len(records)} unique combos, need {B}")

    return pd.DataFrame(records)


def get_combo_feature_cols(combo_row: pd.Series) -> List[str]:
    return [combo_row[c] for c in combo_row.index if c.startswith("feature_") and c.endswith(tuple(str(i) for i in range(10))) is False]


def combo_feature_cols_from_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("feature_") and c.count("_") == 1]
    cols_sorted = sorted(cols, key=lambda x: int(x.split("_")[1]))
    return cols_sorted


def build_year_ranges(oos_year: int) -> Tuple[List[int], List[int], List[int], List[int]]:
    train_years = list(range(oos_year - 30, oos_year - 10))
    valid_years = list(range(oos_year - 10, oos_year))
    test_years = [oos_year]
    all_years = train_years + valid_years + test_years
    return train_years, valid_years, test_years, all_years


def read_multi_year_data(years: List[int], combo_specs_all: List[pd.DataFrame], input_dir: str) -> pd.DataFrame:
    used_rank_features = set()
    for combo_specs in combo_specs_all:
        for feat_col in combo_feature_cols_from_columns(combo_specs):
            used_rank_features.update(combo_specs[feat_col].tolist())

    cols_to_read = [ID_COL, DATE_COL, RET_COL, WEIGHT_COL] + sorted(used_rank_features)

    dfs = []
    for year in years:
        path = os.path.join(input_dir, f"{year}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")
        df_year = pd.read_parquet(path, columns=cols_to_read)
        df_year[DATE_COL] = pd.to_datetime(df_year[DATE_COL])
        dfs.append(df_year)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    return df


def split_node_by_cutoff(node_df: pd.DataFrame, feature_col: str, cutoff: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    left_df = node_df[node_df[feature_col] <= cutoff].copy()
    right_df = node_df[node_df[feature_col] > cutoff].copy()
    return left_df, right_df


@dataclass
class TreeState:
    split_rules: List[dict]
    active_leaves: List[str]
    created_nodes: List[str]


def init_tree_state() -> TreeState:
    return TreeState(split_rules=[], active_leaves=["1"], created_nodes=["1"])


def make_column_name(repeat_idx: int, combo_id: str, tree_id: str, node_id: str) -> str:
    return f"r{repeat_idx:02d}.{combo_id}.{tree_id}.{node_id}"


def apply_tree_one_month(month_df: pd.DataFrame, split_rules: List[dict], keep_leaf_membership: bool = False) -> Tuple[Dict[str, float], Dict[str, int], pd.DataFrame]:
    """
    Apply a learned tree to one month.
    Returns node_returns, node_sizes, optional leaf membership with leaf weights.
    """
    node_frames = {"1": month_df.copy()}
    node_returns = {"1": weighted_return(month_df)}
    node_sizes = {"1": int(len(month_df))}

    for rule in sorted(split_rules, key=lambda x: x["split_step"]):
        parent = rule["parent_node_id"]
        parent_df = node_frames.get(parent)
        if parent_df is None:
            continue

        feat = rule["split_feature"]
        cutoff = float(rule["split_cutoff"])
        left_id = rule["left_node_id"]
        right_id = rule["right_node_id"]

        left_df, right_df = split_node_by_cutoff(parent_df, feat, cutoff)
        node_frames[left_id] = left_df
        node_frames[right_id] = right_df
        node_returns[left_id] = weighted_return(left_df)
        node_returns[right_id] = weighted_return(right_df)
        node_sizes[left_id] = int(len(left_df))
        node_sizes[right_id] = int(len(right_df))

    leaf_membership = pd.DataFrame(columns=[DATE_COL, ID_COL, "leaf_id", "stock_weight_in_leaf"])
    if keep_leaf_membership:
        child_nodes = {r["left_node_id"] for r in split_rules}.union({r["right_node_id"] for r in split_rules})
        parent_nodes = {r["parent_node_id"] for r in split_rules}
        final_leaves = sorted(list((set(["1"]) | child_nodes) - parent_nodes), key=lambda x: (len(x), x))

        leaf_parts = []
        for leaf_id in final_leaves:
            leaf_df = node_frames.get(leaf_id)
            if leaf_df is None or leaf_df.empty:
                continue
            tmp = leaf_df[[DATE_COL, ID_COL, WEIGHT_COL]].copy()
            denom = tmp[WEIGHT_COL].sum()
            if denom > 0:
                tmp["stock_weight_in_leaf"] = tmp[WEIGHT_COL] / denom
            else:
                tmp["stock_weight_in_leaf"] = np.nan
            tmp = tmp.drop(columns=[WEIGHT_COL])
            tmp["leaf_id"] = leaf_id
            leaf_parts.append(tmp)
        if leaf_parts:
            leaf_membership = pd.concat(leaf_parts, axis=0, ignore_index=True)
            leaf_membership = leaf_membership.sort_values([DATE_COL, ID_COL, "leaf_id"]).reset_index(drop=True)

    return node_returns, node_sizes, leaf_membership


def build_leaf_return_matrix(df_period: pd.DataFrame, split_rules: List[dict], active_leaves: List[str]) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    month_groups = {dt: g.copy() for dt, g in df_period.groupby(DATE_COL)}
    dates = sorted(month_groups.keys())

    ret_rows = []
    leaf_count_map = {leaf: [] for leaf in active_leaves}

    for dt in dates:
        node_frames = {"1": month_groups[dt]}
        for rule in sorted(split_rules, key=lambda x: x["split_step"]):
            parent = rule["parent_node_id"]
            parent_df = node_frames[parent]
            left_df, right_df = split_node_by_cutoff(parent_df, rule["split_feature"], float(rule["split_cutoff"]))
            node_frames[rule["left_node_id"]] = left_df
            node_frames[rule["right_node_id"]] = right_df

        row = []
        for leaf in active_leaves:
            leaf_df = node_frames.get(leaf, pd.DataFrame(columns=df_period.columns))
            row.append(weighted_return(leaf_df))
            leaf_count_map[leaf].append(int(len(leaf_df)))
        ret_rows.append(row)

    ret_mat = np.asarray(ret_rows, dtype=float)
    return ret_mat, leaf_count_map


def compute_tree_sharpe(df_period: pd.DataFrame, state: TreeState, ridge_gamma: float) -> Tuple[float, Dict[str, List[int]]]:
    ret_mat, leaf_count_map = build_leaf_return_matrix(df_period, state.split_rules, state.active_leaves)
    sr = sharpe_from_returns(ret_mat, ridge_gamma=ridge_gamma)
    return sr, leaf_count_map


def node_sort_key(node_id: str) -> Tuple[int, str]:
    return (len(str(node_id)), str(node_id))


def get_full_tree_nodes(max_depth: int) -> List[str]:
    """Return all node ids in a complete binary tree up to max_depth."""
    nodes = ["1"]
    frontier = ["1"]
    for _ in range(max_depth):
        new_frontier = []
        for node in frontier:
            left = node + "1"
            right = node + "2"
            nodes.extend([left, right])
            new_frontier.extend([left, right])
        frontier = new_frontier
    return sorted(nodes, key=node_sort_key)


def get_internal_nodes(max_depth: int) -> List[str]:
    """Internal nodes of a complete binary tree, ordered parent-before-child."""
    return [node for node in get_full_tree_nodes(max_depth) if len(node) - 1 < max_depth]


def get_leaf_nodes(max_depth: int) -> List[str]:
    """Final leaves of a complete binary tree."""
    return [node for node in get_full_tree_nodes(max_depth) if len(node) - 1 == max_depth]


def resolve_feature_sequence(feature_cols: List[str], max_depth: int, split_sequence: str = "") -> List[str]:
    """
    Choose one split variable for each tree depth.

    Default: use the first max_depth features in the combo. Since feature_1 is size_rank,
    the root split is size_rank. If max_depth is larger than the number of available
    features, cycle through the selected features.

    Optional split_sequence examples:
      - "0,1,2,3" means use feature_cols[0], feature_cols[1], ...
      - "size_rank,mom12m_rank,roa_rank" means use those exact columns.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if len(feature_cols) == 0:
        raise ValueError("feature_cols cannot be empty")

    if split_sequence is not None and str(split_sequence).strip() != "":
        tokens = [x.strip() for x in str(split_sequence).split(",") if x.strip() != ""]
        seq = []
        for tok in tokens:
            if tok.isdigit():
                idx = int(tok)
                if idx < 0 or idx >= len(feature_cols):
                    raise ValueError(f"split_sequence index {idx} out of range for feature_cols={feature_cols}")
                seq.append(feature_cols[idx])
            else:
                if tok not in feature_cols:
                    raise ValueError(f"split_sequence feature {tok} not in feature_cols={feature_cols}")
                seq.append(tok)
        if len(seq) == 0:
            raise ValueError("split_sequence is empty after parsing")
    else:
        seq = feature_cols.copy()

    if len(seq) < max_depth:
        repeats = int(np.ceil(max_depth / len(seq)))
        seq = (seq * repeats)[:max_depth]
    else:
        seq = seq[:max_depth]

    return seq


def make_complete_tree_state(
    combo_id: str,
    tree_id: str,
    max_depth: int,
    feature_sequence: List[str],
    cutoff_values: Tuple[float, ...],
) -> TreeState:
    """Build a complete tree from a joint vector of cutoff choices."""
    internal_nodes = get_internal_nodes(max_depth)
    if len(cutoff_values) != len(internal_nodes):
        raise ValueError(
            f"Need {len(internal_nodes)} cutoff values for depth={max_depth}, "
            f"got {len(cutoff_values)}"
        )

    split_rules = []
    for split_step, (parent_node, cutoff) in enumerate(zip(internal_nodes, cutoff_values), start=1):
        parent_depth = len(parent_node) - 1
        feat = feature_sequence[parent_depth]
        split_rules.append({
            "combo_id": combo_id,
            "tree_id": tree_id,
            "split_step": split_step,
            "parent_node_id": parent_node,
            "left_node_id": parent_node + "1",
            "right_node_id": parent_node + "2",
            "split_feature": feat,
            "split_feature_short": feature_short_name(feat),
            "split_cutoff": float(cutoff),
        })

    return TreeState(
        split_rules=split_rules,
        active_leaves=get_leaf_nodes(max_depth),
        created_nodes=get_full_tree_nodes(max_depth),
    )


def validate_min_leaf_size(leaf_count_map: Dict[str, List[int]], min_leaf_size: int) -> Tuple[bool, int, float]:
    """Check final leaves, not only the newly split child nodes."""
    if len(leaf_count_map) == 0:
        return False, 0, np.nan

    mins = []
    avgs = []
    for counts in leaf_count_map.values():
        arr = np.asarray(counts, dtype=float)
        if arr.size == 0:
            return False, 0, np.nan
        mins.append(np.nanmin(arr))
        avgs.append(np.nanmean(arr))

    global_min = int(np.nanmin(mins))
    global_avg = float(np.nanmean(avgs))
    return global_min >= min_leaf_size, global_min, global_avg


def grow_one_tree(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    feature_cols: List[str],
    max_depth: int,
    cutoffs: Tuple[float, ...],
    ridge_gamma: float,
    min_leaf_size: int,
    min_gain: float,
    combo_id: str,
    tree_id: str,
    split_sequence: str = "",
) -> Tuple[TreeState, pd.DataFrame]:
    """
    Global complete-tree cutoff search.

    This replaces the old greedy procedure. For a fixed full tree structure and a
    fixed depth-wise feature sequence, it enumerates every combination of candidate
    cutoffs over all internal nodes. The selected tree is the one whose final-leaf
    return matrix has the highest in-sample MVO Sharpe.

    Important: number of candidate trees is len(cutoffs) ** (2 ** max_depth - 1).
    For depth=4 and two cutoffs this is 32,768 candidates per combo.
    """
    import itertools

    if len(cutoffs) == 0:
        raise ValueError("cutoffs cannot be empty")

    baseline_state = init_tree_state()
    train_sr_before, _ = compute_tree_sharpe(df_train, baseline_state, ridge_gamma=ridge_gamma)
    valid_sr_before, _ = compute_tree_sharpe(df_valid, baseline_state, ridge_gamma=ridge_gamma)

    feature_sequence = resolve_feature_sequence(
        feature_cols=feature_cols,
        max_depth=max_depth,
        split_sequence=split_sequence,
    )

    internal_nodes = get_internal_nodes(max_depth)
    n_internal = len(internal_nodes)
    n_candidates = len(cutoffs) ** n_internal

    print(
        f"[INFO] {combo_id} | global search | depth={max_depth} | "
        f"internal_nodes={n_internal} | cutoffs={len(cutoffs)} | candidates={n_candidates} | "
        f"feature_sequence={','.join(feature_short_name(x) for x in feature_sequence)}"
    )

    best = None

    for cand_idx, cutoff_values in enumerate(itertools.product(cutoffs, repeat=n_internal), start=1):
        cand_state = make_complete_tree_state(
            combo_id=combo_id,
            tree_id=tree_id,
            max_depth=max_depth,
            feature_sequence=feature_sequence,
            cutoff_values=tuple(float(x) for x in cutoff_values),
        )

        train_sr_after, leaf_count_map = compute_tree_sharpe(
            df_train,
            cand_state,
            ridge_gamma=ridge_gamma,
        )
        if not np.isfinite(train_sr_after):
            continue

        size_ok, leaf_min_n, leaf_avg_n = validate_min_leaf_size(leaf_count_map, min_leaf_size=min_leaf_size)
        if not size_ok:
            continue

        train_gain = train_sr_after - train_sr_before
        if best is None or train_sr_after > best["train_sr_after"]:
            best = {
                "cand_state": cand_state,
                "train_sr_after": train_sr_after,
                "train_gain": train_gain,
                "leaf_min_n": leaf_min_n,
                "leaf_avg_n": leaf_avg_n,
                "candidate_rank": cand_idx,
            }

    if best is None:
        # Fall back to the unsplit root. This keeps downstream files non-empty.
        empty_rules = pd.DataFrame()
        return baseline_state, empty_rules

    if np.isfinite(best["train_gain"]) and best["train_gain"] <= min_gain:
        # Same stopping interpretation as before: if the best complete tree does
        # not beat the root by enough, do not split. Set min_gain <= 0 to force
        # a full tree whenever a feasible candidate exists.
        empty_rules = pd.DataFrame()
        return baseline_state, empty_rules

    valid_sr_after, _ = compute_tree_sharpe(df_valid, best["cand_state"], ridge_gamma=ridge_gamma)

    split_records = []
    for rule in best["cand_state"].split_rules:
        rec = dict(rule)
        rec.update({
            "train_sr_before": train_sr_before,
            "train_sr_after": best["train_sr_after"],
            "train_gain": best["train_gain"],
            "valid_sr_before": valid_sr_before,
            "valid_sr_after": valid_sr_after,
            "valid_gain": valid_sr_after - valid_sr_before if np.isfinite(valid_sr_before) and np.isfinite(valid_sr_after) else np.nan,
            # Keep old column names for compatibility with any downstream script.
            "left_min_n": np.nan,
            "right_min_n": np.nan,
            "left_avg_n": np.nan,
            "right_avg_n": np.nan,
        })
        split_records.append(rec)

    split_rules_df = pd.DataFrame(split_records)
    return best["cand_state"], split_rules_df


def build_node_metadata(repeat_idx: int, combo_row: pd.Series, tree_id: str, state: TreeState) -> pd.DataFrame:
    split_df = pd.DataFrame(state.split_rules)
    child_nodes = set(split_df["left_node_id"].tolist()) | set(split_df["right_node_id"].tolist()) if not split_df.empty else set()
    parent_nodes = set(split_df["parent_node_id"].tolist()) if not split_df.empty else set()
    all_nodes = sorted(list((set(["1"]) | child_nodes)), key=lambda x: (len(x), x))

    records = []
    feat_cols = combo_feature_cols_from_columns(pd.DataFrame([combo_row]))
    for node_id in all_nodes:
        parent = node_id[:-1] if node_id != "1" else None
        node_type = "leaf" if node_id not in parent_nodes else ("root" if node_id == "1" else "internal")
        rec = {
            "repeat_id": repeat_idx,
            "combo_id": combo_row["combo_id"],
            "tree_id": tree_id,
            "node_id": node_id,
            "column_name": make_column_name(repeat_idx, combo_row["combo_id"], tree_id, node_id),
            "depth": len(node_id) - 1,
            "parent_node_id": parent,
            "node_type": node_type,
            "is_final_leaf": bool(node_id in state.active_leaves),
        }
        for c in feat_cols:
            rec[c] = combo_row[c]
            rec[f"{c}_short"] = feature_short_name(combo_row[c])
        records.append(rec)
    return pd.DataFrame(records)


def build_period_node_returns(
    df_period: pd.DataFrame,
    repeat_idx: int,
    combo_id: str,
    tree_id: str,
    state: TreeState,
    sample_label: str,
) -> pd.DataFrame:
    if df_period.empty:
        return pd.DataFrame(columns=["sample", DATE_COL, "repeat_id", "combo_id", "tree_id", "node_id", "column_name", "ret", "n_stocks"])

    month_groups = {dt: g.copy() for dt, g in df_period.groupby(DATE_COL)}
    rows = []
    for dt in sorted(month_groups.keys()):
        node_returns, node_sizes, _ = apply_tree_one_month(month_groups[dt], state.split_rules, keep_leaf_membership=False)
        for node_id in state.created_nodes:
            rows.append({
                "sample": sample_label,
                DATE_COL: dt,
                "repeat_id": repeat_idx,
                "combo_id": combo_id,
                "tree_id": tree_id,
                "node_id": node_id,
                "column_name": make_column_name(repeat_idx, combo_id, tree_id, node_id),
                "ret": node_returns.get(node_id, np.nan),
                "n_stocks": node_sizes.get(node_id, 0),
            })
    return pd.DataFrame(rows).sort_values([DATE_COL, "repeat_id", "combo_id", "tree_id", "node_id"]).reset_index(drop=True)


def build_test_leaf_weights(
    df_test: pd.DataFrame,
    repeat_idx: int,
    combo_id: str,
    tree_id: str,
    state: TreeState,
) -> pd.DataFrame:
    if df_test.empty:
        return pd.DataFrame(columns=[DATE_COL, ID_COL, "repeat_id", "combo_id", "tree_id", "leaf_id", "stock_weight_in_leaf"])

    month_groups = {dt: g.copy() for dt, g in df_test.groupby(DATE_COL)}
    parts = []
    for dt in sorted(month_groups.keys()):
        _, _, leaf_df = apply_tree_one_month(month_groups[dt], state.split_rules, keep_leaf_membership=True)
        if leaf_df.empty:
            continue
        leaf_df = leaf_df.copy()
        leaf_df["repeat_id"] = repeat_idx
        leaf_df["combo_id"] = combo_id
        leaf_df["tree_id"] = tree_id
        parts.append(leaf_df)
    if not parts:
        return pd.DataFrame(columns=[DATE_COL, ID_COL, "repeat_id", "combo_id", "tree_id", "leaf_id", "stock_weight_in_leaf"])
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out.sort_values([DATE_COL, ID_COL, "repeat_id", "combo_id", "tree_id"]).reset_index(drop=True)


def run_one_repeat(
    repeat_idx: int,
    seed: int,
    oos_year: int,
    B: int,
    max_depth: int,
    df_all: pd.DataFrame,
    combo_specs: pd.DataFrame,
    cutoffs: Tuple[float, ...],
    ridge_gamma: float,
    min_leaf_size: int,
    min_gain: float,
    split_sequence: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_years, valid_years, test_years, _ = build_year_ranges(oos_year)

    node_ret_parts = []
    node_meta_parts = []
    split_rule_parts = []
    combo_spec_parts = []
    test_leaf_parts = []

    feat_cols = combo_feature_cols_from_columns(combo_specs)

    df_train = df_all[df_all[DATE_COL].dt.year.isin(train_years)].copy()
    df_valid = df_all[df_all[DATE_COL].dt.year.isin(valid_years)].copy()
    df_test = df_all[df_all[DATE_COL].dt.year.isin(test_years)].copy()

    for _, combo_row in combo_specs.iterrows():
        combo_id = combo_row["combo_id"]
        tree_id = f"tree_{combo_id.split('_')[-1]}"
        feature_cols = [combo_row[c] for c in feat_cols]
        needed_cols = [DATE_COL, ID_COL, RET_COL, WEIGHT_COL] + feature_cols

        df_train_combo = df_train[needed_cols].dropna(subset=needed_cols).copy()
        df_valid_combo = df_valid[needed_cols].dropna(subset=needed_cols).copy()
        df_test_combo = df_test[needed_cols].dropna(subset=needed_cols).copy()

        if df_train_combo.empty or df_valid_combo.empty or df_test_combo.empty:
            continue

        combo_row_out = combo_row.to_dict()
        combo_row_out.update({
            "repeat_id": repeat_idx,
            "oos_year": oos_year,
            "tree_id": tree_id,
            "seed": seed,
        })
        combo_spec_parts.append(pd.DataFrame([combo_row_out]))

        state, split_rules_df = grow_one_tree(
            df_train=df_train_combo,
            df_valid=df_valid_combo,
            feature_cols=feature_cols,
            max_depth=max_depth,
            cutoffs=cutoffs,
            ridge_gamma=ridge_gamma,
            min_leaf_size=min_leaf_size,
            min_gain=min_gain,
            combo_id=combo_id,
            tree_id=tree_id,
            split_sequence=split_sequence,
        )

        if not split_rules_df.empty:
            split_rules_df = split_rules_df.copy()
            split_rules_df["repeat_id"] = repeat_idx
            split_rules_df["oos_year"] = oos_year
            split_rule_parts.append(split_rules_df)

        node_meta_parts.append(build_node_metadata(repeat_idx, combo_row, tree_id, state))
        node_ret_parts.append(build_period_node_returns(df_train_combo, repeat_idx, combo_id, tree_id, state, "train"))
        node_ret_parts.append(build_period_node_returns(df_valid_combo, repeat_idx, combo_id, tree_id, state, "valid"))
        node_ret_parts.append(build_period_node_returns(df_test_combo, repeat_idx, combo_id, tree_id, state, "test"))
        test_leaf_parts.append(build_test_leaf_weights(df_test_combo, repeat_idx, combo_id, tree_id, state))

        print(f"[INFO] OOS {oos_year} | repeat {repeat_idx} | {combo_id} | splits learned: {len(state.split_rules)} | final leaves: {len(state.active_leaves)}")

    node_returns = pd.concat(node_ret_parts, axis=0, ignore_index=True) if node_ret_parts else pd.DataFrame()
    node_metadata = pd.concat(node_meta_parts, axis=0, ignore_index=True) if node_meta_parts else pd.DataFrame()
    split_rules = pd.concat(split_rule_parts, axis=0, ignore_index=True) if split_rule_parts else pd.DataFrame()
    combo_specs_out = pd.concat(combo_spec_parts, axis=0, ignore_index=True) if combo_spec_parts else pd.DataFrame()
    test_leaf_weights = pd.concat(test_leaf_parts, axis=0, ignore_index=True) if test_leaf_parts else pd.DataFrame()

    return combo_specs_out, split_rules, node_metadata, node_returns, test_leaf_weights


def main(args):
    oos_year = args.year
    B = args.B
    max_depth = args.depth
    N = args.N
    base_seed = args.base_seed

    if oos_year < 1981 or oos_year > 2024:
        raise ValueError("For this rolling design, --year should be an OOS year in [1981, 2024].")

    total_features = args.total_features
    cutoffs = tuple(float(x) for x in args.cutoffs.split(","))
    ridge_gamma = args.ridge_gamma
    min_leaf_size = args.min_leaf_size
    min_gain = args.min_gain
    split_sequence = args.split_sequence

    train_years, valid_years, test_years, all_years = build_year_ranges(oos_year)
    print("=" * 80)
    print(f"[INFO] OOS year: {oos_year}")
    print(f"[INFO] Train years: {train_years[0]}-{train_years[-1]}")
    print(f"[INFO] Valid years: {valid_years[0]}-{valid_years[-1]}")
    print(f"[INFO] Test years : {test_years[0]}-{test_years[-1]}")
    print(f"[INFO] N repeats  : {N}")
    print(f"[INFO] B combos   : {B}")
    print(f"[INFO] Total feats: {total_features}")
    print(f"[INFO] Cutoffs    : {cutoffs}")
    print(f"[INFO] Max depth  : {max_depth}")
    print(f"[INFO] Min leaf n : {min_leaf_size}")
    print(f"[INFO] Min gain   : {min_gain}")
    print(f"[INFO] Search mode: global_complete_tree")
    print(f"[INFO] Split seq  : {split_sequence if split_sequence else 'first depth features in each combo'}")

    combo_specs_all = []
    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs_all.append(build_combo_specs(seed=seed, B=B, chars_path=CHARS_PATH, total_features=total_features))

    df_all = read_multi_year_data(all_years, combo_specs_all, INPUT_DIR)

    run_dir = os.path.join(OUTPUT_BASE_DIR, f"oos_{oos_year}")
    os.makedirs(run_dir, exist_ok=True)

    combo_parts = []
    rule_parts = []
    meta_parts = []
    ret_parts = []
    leaf_parts = []

    for repeat_idx in range(1, N + 1):
        seed = base_seed + repeat_idx - 1
        combo_specs = combo_specs_all[repeat_idx - 1]
        combo_out, rules_out, meta_out, ret_out, leaf_out = run_one_repeat(
            repeat_idx=repeat_idx,
            seed=seed,
            oos_year=oos_year,
            B=B,
            max_depth=max_depth,
            df_all=df_all,
            combo_specs=combo_specs,
            cutoffs=cutoffs,
            ridge_gamma=ridge_gamma,
            min_leaf_size=min_leaf_size,
            min_gain=min_gain,
            split_sequence=split_sequence,
        )
        if not combo_out.empty:
            combo_parts.append(combo_out)
        if not rules_out.empty:
            rule_parts.append(rules_out)
        if not meta_out.empty:
            meta_parts.append(meta_out)
        if not ret_out.empty:
            ret_parts.append(ret_out)
        if not leaf_out.empty:
            leaf_parts.append(leaf_out)

    combo_all = pd.concat(combo_parts, axis=0, ignore_index=True) if combo_parts else pd.DataFrame()
    rules_all = pd.concat(rule_parts, axis=0, ignore_index=True) if rule_parts else pd.DataFrame()
    meta_all = pd.concat(meta_parts, axis=0, ignore_index=True) if meta_parts else pd.DataFrame()
    ret_all = pd.concat(ret_parts, axis=0, ignore_index=True) if ret_parts else pd.DataFrame()
    leaf_all = pd.concat(leaf_parts, axis=0, ignore_index=True) if leaf_parts else pd.DataFrame()

    if not combo_all.empty:
        combo_all = combo_all.sort_values(["repeat_id", "combo_id"]).reset_index(drop=True)
    if not rules_all.empty:
        rules_all = rules_all.sort_values(["repeat_id", "combo_id", "tree_id", "split_step"]).reset_index(drop=True)
    if not meta_all.empty:
        meta_all = meta_all.sort_values(["repeat_id", "combo_id", "tree_id", "node_id"]).reset_index(drop=True)
    if not ret_all.empty:
        ret_all = ret_all.sort_values(["sample", DATE_COL, "repeat_id", "combo_id", "tree_id", "node_id"]).reset_index(drop=True)
    if not leaf_all.empty:
        leaf_all = leaf_all.sort_values([DATE_COL, ID_COL, "repeat_id", "combo_id", "tree_id", "leaf_id"]).reset_index(drop=True)

    atomic_write_parquet(combo_all, os.path.join(run_dir, "combo_specs.parquet"))
    atomic_write_parquet(rules_all, os.path.join(run_dir, "split_rules.parquet"))
    atomic_write_parquet(meta_all, os.path.join(run_dir, "node_metadata.parquet"))
    atomic_write_parquet(ret_all, os.path.join(run_dir, "node_returns.parquet"))
    # This file is the only stock-level output, and only for the OOS year.
    atomic_write_parquet(leaf_all, os.path.join(run_dir, "test_leaf_weights.parquet"))

    summary = pd.DataFrame([{
        "oos_year": oos_year,
        "train_start": train_years[0],
        "train_end": train_years[-1],
        "valid_start": valid_years[0],
        "valid_end": valid_years[-1],
        "test_start": test_years[0],
        "test_end": test_years[-1],
        "N": N,
        "B": B,
        "total_features": total_features,
        "max_depth": max_depth,
        "cutoffs": ",".join(str(x) for x in cutoffs),
        "ridge_gamma": ridge_gamma,
        "min_leaf_size": min_leaf_size,
        "min_gain": min_gain,
        "n_combo_rows": len(combo_all),
        "n_split_rows": len(rules_all),
        "n_node_rows": len(meta_all),
        "n_return_rows": len(ret_all),
        "n_test_leaf_rows": len(leaf_all),
    }])
    atomic_write_csv(summary, os.path.join(run_dir, "run_summary.csv"))

    print("=" * 80)
    print(f"[INFO] Finished OOS year {oos_year}")
    print(f"[INFO] Output dir: {run_dir}")
    print(f"[INFO] combo rows      : {len(combo_all)}")
    print(f"[INFO] split rule rows : {len(rules_all)}")
    print(f"[INFO] metadata rows   : {len(meta_all)}")
    print(f"[INFO] return rows     : {len(ret_all)}")
    print(f"[INFO] test leaf rows  : {len(leaf_all)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="OOS year, e.g. 1981")
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--base_seed", type=int, default=0)

    parser.add_argument("--total_features", type=int, default=6, help="Total features per combo, including size_rank")
    parser.add_argument("--cutoffs", type=str, default="0.3333333333,0.6666666667")
    parser.add_argument("--ridge_gamma", type=float, default=1e-4)
    parser.add_argument("--min_leaf_size", type=int, default=30)
    parser.add_argument("--min_gain", type=float, default=-1e18)
    parser.add_argument(
        "--split_sequence",
        type=str,
        default="",
        help=(
            "Optional depth-wise split sequence. Examples: '0,1,2,3' uses feature_cols by index; "
            "'size_rank,mom12m_rank,roa_rank' uses exact feature names. "
            "Default: first --depth features in each combo, cycling only if needed."
        ),
    )

    args = parser.parse_args()
    main(args)
