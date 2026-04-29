from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# AP forest I paths
# =========================================================
RANK_FEATURE_BASE = Path("../../../data/feature_construction/1_4_rank_feature/by_year/no_impute")

SPLIT_BASE = Path("../../../data/2_tree_construction/2_1_split/2_1_4_split_benchmark/output")
FILTER_BASE = Path("../../../data/2_tree_construction/2_3_filter/2_3_4_filter_benchmark/output")
PRUNE_BASE = Path("../../../data/3_ap_prune/3_1_prune_cv/3_1_4_prune_cv_benchmark/output")

OUTPUT_BASE = Path("../../../data/4_results/4_4_turnover/4_4_2_turnover_ap_tree/output")

DATE_COL = "date"
ID_COL = "permno"
WEIGHT_COL = "size"


# =========================================================
# basic helpers
# =========================================================
def parse_int_list(s: str) -> list[int]:
    """
    Parse strings like:
        "1-10"
        "1,2,3"
        "1,2,5-10"
    """
    out: list[int] = []

    for part in s.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))

    return sorted(set(out))


def make_repeat_tag(repeat_idx: int, seed: int, B: int, depth: int) -> str:
    return f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"


def get_beta_cols(row: pd.Series) -> list[str]:
    beta_cols = [c for c in row.index if str(c).startswith("beta_")]
    beta_cols = sorted(beta_cols, key=lambda x: int(str(x).split("_")[1]))
    return beta_cols


def read_stock_data_for_years(
    years: list[int],
    cache: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Read date-permno-size-ret from the same ranked feature files used by split.

    size is used to reconstruct value-weighted node holdings.
    ret is used for drift-adjusted turnover.
    """
    if cache is None:
        cache = {}

    dfs: list[pd.DataFrame] = []

    for year in years:
        if year not in cache:
            fp = RANK_FEATURE_BASE / f"{year}.parquet"
            if not fp.exists():
                raise FileNotFoundError(f"Cannot find ranked feature file: {fp}")

            df = pd.read_parquet(
                fp,
                columns=[DATE_COL, ID_COL, WEIGHT_COL, RET_COL],
            ).copy()

            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce")
            df[RET_COL] = pd.to_numeric(df[RET_COL], errors="coerce")
            df = df.dropna(subset=[DATE_COL, ID_COL])
            df = df.drop_duplicates(subset=[DATE_COL, ID_COL])
            df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

            cache[year] = df

        dfs.append(cache[year])

    if len(dfs) == 0:
        raise ValueError("No stock data loaded.")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=[DATE_COL, ID_COL])
    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return out


# =========================================================
# prune-selection helpers: same selection rule as PnL
# =========================================================
def choose_best_cv_row(window_dir: Path, cvN: int) -> dict | None:
    """
    Scan all results_cv_{cvN}_l0_*_l2_*.csv files.
    Choose the row with the highest valid_SR.
    """
    cv_files = sorted(window_dir.glob(f"results_cv_{cvN}_l0_*_l2_*.csv"))

    if len(cv_files) == 0:
        return None

    best_cv: dict | None = None
    pattern = rf"results_cv_{cvN}_l0_(\d+)_l2_(\d+)\.csv$"

    for fp in cv_files:
        m = re.search(pattern, fp.name)
        if m is None:
            continue

        l0_idx = int(m.group(1))
        l2_idx = int(m.group(2))

        df = pd.read_csv(fp)
        if df.empty or "valid_SR" not in df.columns:
            continue

        df["valid_SR"] = pd.to_numeric(df["valid_SR"], errors="coerce")
        df = df[np.isfinite(df["valid_SR"])].copy()

        if df.empty:
            continue

        local_idx = df["valid_SR"].idxmax()
        row = df.loc[local_idx].copy()

        cand = {
            "l0_idx": l0_idx,
            "l2_idx": l2_idx,
            "lambda0": float(row["lambda0"]),
            "lambda2": float(row["lambda2"]),
            "portsN": int(row["portsN"]),
            "valid_SR": float(row["valid_SR"]),
            "cv_file": str(fp),
        }

        if best_cv is None or cand["valid_SR"] > best_cv["valid_SR"]:
            best_cv = cand

    return best_cv


def choose_matching_full_row(window_dir: Path, best_cv: dict) -> pd.Series | None:
    """
    Given the best lambda0/lambda2/portsN selected by CV,
    open the corresponding full file and choose the row with same portsN.

    If multiple rows have same portsN, choose the one with max train_SR.
    Do not use test_SR for model selection.
    """
    full_fp = window_dir / f"results_full_l0_{best_cv['l0_idx']}_l2_{best_cv['l2_idx']}.csv"

    if not full_fp.exists():
        return None

    full_df = pd.read_csv(full_fp)
    if full_df.empty:
        return None

    full_df["train_SR"] = pd.to_numeric(full_df["train_SR"], errors="coerce")
    full_sub = full_df[full_df["portsN"] == best_cv["portsN"]].copy()

    if full_sub.empty:
        return None

    full_sub = full_sub[np.isfinite(full_sub["train_SR"])].copy()
    if full_sub.empty:
        return None

    full_row = full_sub.loc[full_sub["train_SR"].idxmax()].copy()
    return full_row


def get_selected_nodes(window_dir: Path, cvN: int) -> pd.DataFrame | None:
    """
    Recover selected node portfolios and their final raw beta weights.

    Output columns include:
        column_name, combo_id, tree_id, node_id, beta
    """
    meta_fp = window_dir / "portfolio_metadata_window.csv"
    if not meta_fp.exists():
        print(f"[WARNING] Missing portfolio_metadata_window.csv: {meta_fp}")
        return None

    metadata = pd.read_csv(meta_fp).copy()

    required_meta_cols = ["column_name", "combo_id", "tree_id", "node_id"]
    missing = [c for c in required_meta_cols if c not in metadata.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing metadata columns {missing}: {meta_fp}")

    metadata["column_name"] = metadata["column_name"].astype(str)
    metadata["combo_id"] = metadata["combo_id"].astype(str)
    metadata["tree_id"] = metadata["tree_id"].astype(str)
    metadata["node_id"] = metadata["node_id"].astype(str)

    best_cv = choose_best_cv_row(window_dir=window_dir, cvN=cvN)
    if best_cv is None:
        print(f"[WARNING] No valid CV candidate: {window_dir}")
        return None

    full_row = choose_matching_full_row(window_dir=window_dir, best_cv=best_cv)
    if full_row is None:
        print(
            f"[WARNING] No matching full row: {window_dir}, "
            f"l0={best_cv['l0_idx']}, l2={best_cv['l2_idx']}, portsN={best_cv['portsN']}"
        )
        return None

    beta_cols = get_beta_cols(full_row)

    if len(beta_cols) != len(metadata):
        print(
            f"[WARNING] beta length mismatch in {window_dir}: "
            f"{len(beta_cols)} betas vs {len(metadata)} metadata rows"
        )
        return None

    beta_vec = full_row[beta_cols].astype(float).to_numpy()

    selected = metadata.copy()
    selected["beta"] = beta_vec

    # beta_* are already final raw node-portfolio weights from prune.
    selected = selected[np.abs(selected["beta"]) > 1e-12].copy()

    if selected.empty:
        print(f"[WARNING] No nonzero beta selected: {window_dir}")
        return None

    selected["best_lambda0_idx"] = best_cv["l0_idx"]
    selected["best_lambda2_idx"] = best_cv["l2_idx"]
    selected["best_lambda0"] = best_cv["lambda0"]
    selected["best_lambda2"] = best_cv["lambda2"]
    selected["best_portsN"] = best_cv["portsN"]
    selected["valid_SR"] = best_cv["valid_SR"]
    selected["train_SR_full"] = float(full_row["train_SR"])
    selected["test_SR_full"] = float(full_row["test_SR"])

    return selected


# =========================================================
# reconstruct node stock weights from leaf_membership + size
# =========================================================
def load_leaf_membership(
    repeat_tag: str,
    years: list[int],
    test_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    """
    Load leaf_membership.parquet from split output.

    leaf_membership only contains final leaves. For any internal node_id,
    stock membership can be reconstructed by:
        leaf_id.startswith(node_id)
    """
    dfs: list[pd.DataFrame] = []

    repeat_split_dir = SPLIT_BASE / repeat_tag

    for year in years:
        fp = repeat_split_dir / str(year) / "leaf_membership.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Cannot find leaf membership file: {fp}")

        df = pd.read_parquet(fp).copy()

        required_cols = [DATE_COL, ID_COL, "combo_id", "tree_id", "leaf_id"]
        missing = [c for c in required_cols if c not in df.columns]
        if len(missing) > 0:
            raise ValueError(f"Missing columns {missing}: {fp}")

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df[df[DATE_COL].isin(test_dates)].copy()

        if not df.empty:
            df["combo_id"] = df["combo_id"].astype(str)
            df["tree_id"] = df["tree_id"].astype(str)
            df["leaf_id"] = df["leaf_id"].astype(str)
            dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame(columns=[DATE_COL, ID_COL, "combo_id", "tree_id", "leaf_id"])

    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def reconstruct_strategy_holdings_one_window(
    repeat_idx: int,
    window_idx: int,
    repeat_tag: str,
    ret_df: pd.DataFrame,
    prune_repeat_dir: Path,
    cvN: int,
    stock_data_cache: dict[int, pd.DataFrame],
) -> pd.DataFrame | None:
    """
    For one repeat-window:
      1. recover selected node beta from prune output
      2. read split leaf_membership for OOS test year/months
      3. reconstruct each selected node's stock members via leaf_id prefix
      4. compute node-level stock weights using size
      5. aggregate beta * node_stock_weight to strategy stock holdings
    """
    window_tag = f"window_{window_idx:04d}"
    window_dir = prune_repeat_dir / window_tag

    if not window_dir.exists():
        print(f"[WARNING] Missing window dir: {window_dir}")
        return None

    selected_nodes = get_selected_nodes(window_dir=window_dir, cvN=cvN)
    if selected_nodes is None or selected_nodes.empty:
        return None

    summary_fp = prune_repeat_dir / "rolling_window_summary.csv"
    if not summary_fp.exists():
        print(f"[WARNING] Missing rolling_window_summary.csv: {summary_fp}")
        return None

    summary_df = pd.read_csv(summary_fp)
    summary_df["test_start_date"] = pd.to_datetime(summary_df["test_start_date"])
    summary_df["test_end_date"] = pd.to_datetime(summary_df["test_end_date"])

    one_summary = summary_df[summary_df["window_idx"] == window_idx].copy()
    if one_summary.empty:
        print(f"[WARNING] window_idx={window_idx} not found in rolling_window_summary")
        return None

    test_start = pd.to_datetime(one_summary["test_start_date"].iloc[0])
    test_end = pd.to_datetime(one_summary["test_end_date"].iloc[0])

    test_dates = (
        ret_df.loc[
            (ret_df[DATE_COL] >= test_start) & (ret_df[DATE_COL] <= test_end),
            DATE_COL,
        ]
        .drop_duplicates()
        .sort_values()
    )

    if test_dates.empty:
        print(f"[WARNING] Empty test dates: repeat={repeat_idx}, window={window_idx}")
        return None

    years = sorted(set(pd.to_datetime(test_dates).dt.year.astype(int).tolist()))
    test_date_set = set(pd.to_datetime(test_dates).tolist())

    stock_df = read_stock_data_for_years(years, cache=stock_data_cache)
    size_df = stock_df[[DATE_COL, ID_COL, WEIGHT_COL]].copy()
    size_df = size_df[size_df[DATE_COL].isin(test_date_set)].copy()
    size_df = size_df.dropna(subset=[WEIGHT_COL])
    size_df = size_df[np.isfinite(size_df[WEIGHT_COL]) & (size_df[WEIGHT_COL] > 0)].copy()

    leaf_df = load_leaf_membership(
        repeat_tag=repeat_tag,
        years=years,
        test_dates=test_date_set,
    )

    if leaf_df.empty:
        print(f"[WARNING] Empty leaf membership: repeat={repeat_idx}, window={window_idx}")
        return None

    # Only keep combo-tree pairs that are actually selected.
    selected_pairs = selected_nodes[["combo_id", "tree_id"]].drop_duplicates()
    leaf_df = leaf_df.merge(selected_pairs, on=["combo_id", "tree_id"], how="inner")

    if leaf_df.empty:
        print(f"[WARNING] No leaf rows after selected combo-tree filter: repeat={repeat_idx}, window={window_idx}")
        return None

    # Attach size once.
    leaf_df = leaf_df.merge(size_df, on=[DATE_COL, ID_COL], how="left", validate="many_to_one")
    leaf_df = leaf_df.dropna(subset=[WEIGHT_COL])
    leaf_df = leaf_df[np.isfinite(leaf_df[WEIGHT_COL]) & (leaf_df[WEIGHT_COL] > 0)].copy()

    if leaf_df.empty:
        print(f"[WARNING] Empty leaf rows after size merge: repeat={repeat_idx}, window={window_idx}")
        return None

    components: list[pd.DataFrame] = []

    # Process selected nodes by combo-tree to avoid any ambiguous external holdings file.
    for (combo_id, tree_id), selected_pair in selected_nodes.groupby(["combo_id", "tree_id"]):
        leaf_pair = leaf_df[
            (leaf_df["combo_id"] == combo_id)
            & (leaf_df["tree_id"] == tree_id)
        ].copy()

        if leaf_pair.empty:
            continue

        leaf_id_str = leaf_pair["leaf_id"].astype(str)

        for _, node_row in selected_pair.iterrows():
            node_id = str(node_row["node_id"])
            beta = float(node_row["beta"])

            # A stock assigned to final leaf L belongs to internal node N iff L starts with N.
            cur = leaf_pair[leaf_id_str.str.startswith(node_id)].copy()

            if cur.empty:
                continue

            denom = cur.groupby(DATE_COL)[WEIGHT_COL].transform("sum")
            cur = cur[denom > 0].copy()
            denom = denom.loc[cur.index]

            cur["node_stock_weight"] = cur[WEIGHT_COL].to_numpy(dtype=float) / denom.to_numpy(dtype=float)
            cur["strategy_weight_component"] = beta * cur["node_stock_weight"]

            components.append(
                cur[[DATE_COL, ID_COL, "strategy_weight_component"]].copy()
            )

    if len(components) == 0:
        print(f"[WARNING] No strategy holding components: repeat={repeat_idx}, window={window_idx}")
        return None

    strategy_h = pd.concat(components, axis=0, ignore_index=True)

    strategy_h = (
        strategy_h
        .groupby([DATE_COL, ID_COL], as_index=False)["strategy_weight_component"]
        .sum()
        .rename(columns={"strategy_weight_component": "strategy_weight"})
    )

    strategy_h = strategy_h[np.abs(strategy_h["strategy_weight"]) > 1e-15].copy()
    strategy_h["repeat_idx"] = repeat_idx
    strategy_h["window_idx"] = window_idx
    strategy_h["strategy_id"] = f"repeat_{repeat_idx:02d}_window_{window_idx:04d}"

    return strategy_h


# =========================================================
# ensemble holdings
# =========================================================
def compute_ensemble_holdings(strategy_holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Combine repeat-window holdings into AP forest I ensemble holdings.

    Missing stock in a strategy means zero weight.
    Therefore:
        ensemble_weight_{i,t}
        = sum_strategy weight_{i,t,strategy} / number_of_active_strategies_t
    """
    n_contributors = (
        strategy_holdings[[DATE_COL, "strategy_id"]]
        .drop_duplicates()
        .groupby(DATE_COL)
        .size()
        .rename("n_contributors")
        .reset_index()
    )

    summed = (
        strategy_holdings
        .groupby([DATE_COL, ID_COL], as_index=False)["strategy_weight"]
        .sum()
    )

    ensemble = summed.merge(n_contributors, on=DATE_COL, how="left")
    ensemble["weight"] = ensemble["strategy_weight"] / ensemble["n_contributors"]

    ensemble = ensemble[[DATE_COL, ID_COL, "weight", "n_contributors"]].copy()
    ensemble = ensemble[np.abs(ensemble["weight"]) > 1e-15].copy()
    ensemble = ensemble.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return ensemble


# =========================================================
# drift-adjusted turnover
# =========================================================
def compute_monthly_turnover_drift_adjusted(
    ensemble_holdings: pd.DataFrame,
    stock_ret_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drift-adjusted turnover:

        turnover_{t+1}
        = 0.5 * sum_i | w_{i,t+1}
            - w_{i,t} * (1 + r_{i,t+1}) / (1 + sum_j w_{j,t} r_{j,t+1}) |

    Here:
        w_{i,t}      = post-rebalance stock weight at date t
        r_{i,t+1}    = stock return over next period, recorded at date t+1
        w_{i,t+1}    = new post-rebalance stock weight at date t+1

    The first month has no previous portfolio, so turnover is NaN.

    simple_turnover is also saved for debugging:
        simple_turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
    """
    h = ensemble_holdings.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])
    h = h.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    stock_ret = stock_ret_df[[DATE_COL, ID_COL, RET_COL]].copy()
    stock_ret[DATE_COL] = pd.to_datetime(stock_ret[DATE_COL])
    stock_ret[RET_COL] = pd.to_numeric(stock_ret[RET_COL], errors="coerce")
    stock_ret = stock_ret.drop_duplicates(subset=[DATE_COL, ID_COL])

    dates = sorted(h[DATE_COL].drop_duplicates().tolist())

    rows = []
    prev_w: pd.Series | None = None

    for d in dates:
        cur = h[h[DATE_COL] == d].copy()
        cur_w = cur.set_index(ID_COL)["weight"].astype(float)

        gross_exposure = float(np.abs(cur_w).sum())
        net_exposure = float(cur_w.sum())
        n_stocks = int(cur_w.shape[0])
        n_contributors = int(cur["n_contributors"].iloc[0])

        if prev_w is None:
            turnover = np.nan
            simple_turnover = np.nan
            portfolio_ret_from_prev = np.nan
            n_prev_stocks = np.nan
            n_missing_prev_ret = np.nan
        else:
            # r_{i,t+1}: return ending at current date d.
            r_df = stock_ret[stock_ret[DATE_COL] == d].copy()
            r = r_df.set_index(ID_COL)[RET_COL].astype(float)

            # Need returns for stocks held at previous date.
            # Missing return is filled with 0 for robustness, and counted for diagnostics.
            r_prev = r.reindex(prev_w.index)
            n_missing_prev_ret = int(r_prev.isna().sum())
            r_prev = r_prev.fillna(0.0)

            n_prev_stocks = int(prev_w.shape[0])

            # Portfolio return from previous post-rebalance weights.
            portfolio_ret_from_prev = float((prev_w * r_prev).sum())

            denom = 1.0 + portfolio_ret_from_prev

            if not np.isfinite(denom) or abs(denom) < 1e-12:
                turnover = np.nan
            else:
                # Pre-rebalance weight at current date after return drift.
                prev_w_drifted = prev_w * (1.0 + r_prev) / denom

                # Current non-held-before stocks have drifted weight 0.
                # Previously held but not currently targeted stocks are included via fill_value=0.
                diff = cur_w.subtract(prev_w_drifted, fill_value=0.0)
                turnover = 0.5 * float(np.abs(diff).sum())

            simple_diff = cur_w.subtract(prev_w, fill_value=0.0)
            simple_turnover = 0.5 * float(np.abs(simple_diff).sum())

        rows.append({
            DATE_COL: d,
            "turnover": turnover,
            "simple_turnover": simple_turnover,
            "portfolio_ret_from_prev": portfolio_ret_from_prev,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "n_stocks": n_stocks,
            "n_prev_stocks": n_prev_stocks,
            "n_missing_prev_ret": n_missing_prev_ret,
            "n_contributors": n_contributors,
        })

        prev_w = cur_w

    out = pd.DataFrame(rows)
    return out


# =========================================================
# main
# =========================================================
def main(args: argparse.Namespace) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    repeats = parse_int_list(args.repeats)
    all_strategy_holdings: list[pd.DataFrame] = []

    # Cache ranked-feature year files because many windows share the same years.
    stock_data_cache: dict[int, pd.DataFrame] = {}

    for repeat_idx in repeats:
        seed = args.base_seed + repeat_idx - 1
        repeat_tag = make_repeat_tag(
            repeat_idx=repeat_idx,
            seed=seed,
            B=args.B,
            depth=args.depth,
        )

        print("=" * 100)
        print(f"[INFO] Processing {repeat_tag}")

        filter_repeat_dir = FILTER_BASE / repeat_tag
        prune_repeat_dir = PRUNE_BASE / repeat_tag
        split_repeat_dir = SPLIT_BASE / repeat_tag

        if not filter_repeat_dir.exists():
            print(f"[WARNING] Missing filter repeat dir: {filter_repeat_dir}")
            continue

        if not prune_repeat_dir.exists():
            print(f"[WARNING] Missing prune repeat dir: {prune_repeat_dir}")
            continue

        if not split_repeat_dir.exists():
            print(f"[WARNING] Missing split repeat dir: {split_repeat_dir}")
            continue

        ret_fp = filter_repeat_dir / "level_all_excess_ret_combined_filtered.pkl.gz"
        if not ret_fp.exists():
            print(f"[WARNING] Missing filtered return file: {ret_fp}")
            continue

        summary_fp = prune_repeat_dir / "rolling_window_summary.csv"
        if not summary_fp.exists():
            print(f"[WARNING] Missing rolling_window_summary.csv: {summary_fp}")
            continue

        ret_df = pd.read_pickle(ret_fp).copy()
        ret_df[DATE_COL] = pd.to_datetime(ret_df[DATE_COL])
        ret_df = ret_df.sort_values(DATE_COL).reset_index(drop=True)

        summary_df = pd.read_csv(summary_fp)

        if args.windows.lower() == "auto":
            windows = sorted(summary_df["window_idx"].astype(int).unique().tolist())
        else:
            windows = parse_int_list(args.windows)

        print(f"[INFO] n_windows: {len(windows)}")

        for window_idx in windows:
            one_h = reconstruct_strategy_holdings_one_window(
                repeat_idx=repeat_idx,
                window_idx=window_idx,
                repeat_tag=repeat_tag,
                ret_df=ret_df,
                prune_repeat_dir=prune_repeat_dir,
                cvN=args.cvN,
                stock_data_cache=stock_data_cache,
            )

            if one_h is not None and not one_h.empty:
                all_strategy_holdings.append(one_h)

    if len(all_strategy_holdings) == 0:
        raise ValueError("No strategy holdings reconstructed. Please check split/prune/filter paths.")

    strategy_holdings = pd.concat(all_strategy_holdings, axis=0, ignore_index=True)

    ensemble_holdings = compute_ensemble_holdings(strategy_holdings)

    all_dates = pd.to_datetime(ensemble_holdings[DATE_COL]).drop_duplicates().sort_values()
    years_needed = sorted(all_dates.dt.year.astype(int).unique().tolist())
    date_set = set(all_dates.tolist())

    stock_ret_df = read_stock_data_for_years(years_needed, cache=stock_data_cache)
    stock_ret_df = stock_ret_df[stock_ret_df[DATE_COL].isin(date_set)].copy()

    monthly_turnover = compute_monthly_turnover_drift_adjusted(
        ensemble_holdings=ensemble_holdings,
        stock_ret_df=stock_ret_df,
    )

    output_fp = OUTPUT_BASE / "monthly_turnover.csv"
    monthly_turnover.to_csv(output_fp, index=False)

    print("=" * 100)
    print("[INFO] AP forest I drift-adjusted turnover finished successfully")
    print(f"[INFO] Saved monthly turnover to: {output_fp}")
    print(f"[INFO] strategy_holdings rows: {len(strategy_holdings):,}")
    print(f"[INFO] ensemble_holdings rows: {len(ensemble_holdings):,}")
    print(f"[INFO] stock_ret_df rows: {len(stock_ret_df):,}")
    print(f"[INFO] monthly_turnover shape: {monthly_turnover.shape}")

    avg_turnover = monthly_turnover["turnover"].dropna().mean()
    avg_simple_turnover = monthly_turnover["simple_turnover"].dropna().mean()

    print(f"[INFO] average drift-adjusted monthly turnover: {avg_turnover:.6f}")
    print(f"[INFO] average simple monthly turnover        : {avg_simple_turnover:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--base_seed", type=int, default=0)

    # Examples:
    #   --repeats 1-10
    #   --repeats 1,2,3,4,6,7,8,9,10
    parser.add_argument("--repeats", type=str, default="1-10")

    # Use "auto" to read all windows from rolling_window_summary.csv.
    # Or use "1-44", "1,2,3", etc.
    parser.add_argument("--windows", type=str, default="auto")

    parser.add_argument("--cvN", type=int, default=3)

    args = parser.parse_args()
    main(args)