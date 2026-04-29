from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# AP forest II / ptree paths
# =========================================================
RANK_FEATURE_BASE = Path("../../../data/feature_construction/1_4_rank_feature/by_year/no_impute")

SPLIT_BASE = Path("../../../data/2_tree_construction/2_1_split/2_1_5_split_ptree/output")
FILTER_BASE = Path("../../../data/2_tree_construction/2_3_filter/2_3_5_filter_ptree/output")
PRUNE_BASE = Path("../../../data/3_ap_prune/3_1_prune_cv/3_1_5_prune_cv_ptree/output")

OUTPUT_BASE = Path("../../../data/4_results/4_4_turnover/4_4_3_turnover_ap_forestII/output")

DATE_COL = "date"
ID_COL = "permno"
RET_COL = "ret"


# =========================================================
# basic helpers
# =========================================================
def parse_int_list(s: str) -> list[int]:
    """
    Parse strings like:
        "1981-2024"
        "1981,1982,1983"
        "1981,1985,1990-2000"
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


def get_beta_cols(row: pd.Series) -> list[str]:
    beta_cols = [c for c in row.index if str(c).startswith("beta_")]
    beta_cols = sorted(beta_cols, key=lambda x: int(str(x).split("_")[1]))
    return beta_cols


def read_return_matrix(filter_repeat_dir: Path) -> pd.DataFrame:
    pkl_fp = filter_repeat_dir / "level_all_excess_ret_combined_filtered.pkl.gz"
    parquet_fp = filter_repeat_dir / "level_all_excess_ret_combined_filtered.parquet"
    csv_fp = filter_repeat_dir / "level_all_excess_ret_combined_filtered.csv"

    if pkl_fp.exists():
        df = pd.read_pickle(pkl_fp).copy()
    elif parquet_fp.exists():
        df = pd.read_parquet(parquet_fp).copy()
    elif csv_fp.exists():
        df = pd.read_csv(csv_fp).copy()
    else:
        raise FileNotFoundError(
            f"Cannot find filtered return matrix under {filter_repeat_dir}"
        )

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def read_stock_returns_for_years(years: list[int]) -> pd.DataFrame:
    """
    Read stock-level returns from the same ranked feature files used by ptree split.

    Required output:
        date, permno, ret

    In the turnover formula, for transition from t to t+1, this script uses
    stock returns recorded at date t+1.
    """
    dfs: list[pd.DataFrame] = []

    for year in years:
        fp = RANK_FEATURE_BASE / f"{year}.parquet"

        if not fp.exists():
            raise FileNotFoundError(f"Cannot find ranked feature file: {fp}")

        df = pd.read_parquet(fp, columns=[DATE_COL, ID_COL, RET_COL]).copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[RET_COL] = pd.to_numeric(df[RET_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL, ID_COL])
        df = df.drop_duplicates(subset=[DATE_COL, ID_COL])

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No stock return data loaded.")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=[DATE_COL, ID_COL])
    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return out


# =========================================================
# prune selection helpers
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
    Given best lambda0/lambda2/portsN selected by CV,
    open corresponding full file and choose the row with same portsN.

    If multiple rows have same portsN, choose max train_SR.
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

    return full_sub.loc[full_sub["train_SR"].idxmax()].copy()


def get_selected_leaf_nodes(window_dir: Path, cvN: int) -> pd.DataFrame | None:
    """
    Recover selected final-leaf portfolios and their final raw beta weights.

    Output columns:
        repeat_id, combo_id, tree_id, node_id, column_name, beta
    """
    meta_fp_csv = window_dir / "portfolio_metadata_window.csv"

    if not meta_fp_csv.exists():
        print(f"[WARNING] Missing portfolio_metadata_window.csv: {meta_fp_csv}")
        return None

    metadata = pd.read_csv(meta_fp_csv).copy()

    required_cols = ["column_name", "repeat_id", "combo_id", "tree_id", "node_id"]
    missing = [c for c in required_cols if c not in metadata.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing metadata columns {missing}: {meta_fp_csv}")

    metadata["column_name"] = metadata["column_name"].astype(str)
    metadata["combo_id"] = metadata["combo_id"].astype(str)
    metadata["tree_id"] = metadata["tree_id"].astype(str)
    metadata["node_id"] = metadata["node_id"].astype(str)
    metadata["repeat_id"] = metadata["repeat_id"].astype(int)

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

    # beta_* are already final raw leaf-portfolio weights from prune.
    selected = selected[np.abs(selected["beta"]) > 1e-12].copy()

    if selected.empty:
        print(f"[WARNING] No nonzero beta selected: {window_dir}")
        return None

    # This ptree turnover script uses test_leaf_weights.parquet.
    # That file stores final leaf weights. Therefore selected nodes must be final leaves.
    if "is_final_leaf" in selected.columns:
        bad = selected[selected["is_final_leaf"] != True].copy()

        if not bad.empty:
            example = bad[["column_name", "node_id", "is_final_leaf"]].head(5)
            raise ValueError(
                "Selected non-final-leaf portfolios were found. "
                "This turnover script assumes ptree combine was run with keep_nodes='leaves'. "
                f"Examples:\n{example}"
            )

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
# reconstruct strategy holdings from test_leaf_weights
# =========================================================
def load_test_leaf_weights(year: int) -> pd.DataFrame:
    """
    ptree split saves stock-level OOS leaf weights here:
        2_1_5_split_ptree/output/oos_YYYY/test_leaf_weights.parquet

    Columns expected:
        date, permno, repeat_id, combo_id, tree_id, leaf_id, stock_weight_in_leaf
    """
    fp = SPLIT_BASE / f"oos_{year}" / "test_leaf_weights.parquet"

    if not fp.exists():
        raise FileNotFoundError(f"Cannot find test leaf weights: {fp}")

    df = pd.read_parquet(fp).copy()

    required_cols = [
        DATE_COL,
        ID_COL,
        "repeat_id",
        "combo_id",
        "tree_id",
        "leaf_id",
        "stock_weight_in_leaf",
    ]
    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing columns {missing}: {fp}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["repeat_id"] = df["repeat_id"].astype(int)
    df["combo_id"] = df["combo_id"].astype(str)
    df["tree_id"] = df["tree_id"].astype(str)
    df["leaf_id"] = df["leaf_id"].astype(str)
    df["stock_weight_in_leaf"] = pd.to_numeric(df["stock_weight_in_leaf"], errors="coerce")

    df = df.dropna(subset=[DATE_COL, ID_COL, "stock_weight_in_leaf"])
    df = df[np.isfinite(df["stock_weight_in_leaf"])].copy()
    df = df.sort_values([DATE_COL, ID_COL, "repeat_id", "combo_id", "tree_id", "leaf_id"]).reset_index(drop=True)

    return df


def reconstruct_strategy_holdings_one_oos_repeat(
    year: int,
    repeat_id: int,
    leaf_all: pd.DataFrame,
    filter_repeat_dir: Path,
    prune_repeat_dir: Path,
    cvN: int,
) -> pd.DataFrame | None:
    """
    For one OOS year and one repeat:
      1. recover selected final leaf beta from prune output
      2. merge with test_leaf_weights
      3. aggregate beta * stock_weight_in_leaf to stock-level strategy holdings
    """
    window_dir = prune_repeat_dir / "window_0001"

    if not window_dir.exists():
        print(f"[WARNING] Missing window dir: {window_dir}")
        return None

    selected = get_selected_leaf_nodes(window_dir=window_dir, cvN=cvN)

    if selected is None or selected.empty:
        return None

    ret_df = read_return_matrix(filter_repeat_dir)

    summary_fp = prune_repeat_dir / "rolling_window_summary.csv"

    if summary_fp.exists():
        summary_df = pd.read_csv(summary_fp)
        summary_df["test_start_date"] = pd.to_datetime(summary_df["test_start_date"])
        summary_df["test_end_date"] = pd.to_datetime(summary_df["test_end_date"])

        one_summary = summary_df[summary_df["window_idx"] == 1].copy()

        if not one_summary.empty:
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
        else:
            test_dates = ret_df[ret_df[DATE_COL].dt.year == year][DATE_COL].drop_duplicates().sort_values()
    else:
        test_dates = ret_df[ret_df[DATE_COL].dt.year == year][DATE_COL].drop_duplicates().sort_values()

    if test_dates.empty:
        print(f"[WARNING] Empty test dates: year={year}, repeat={repeat_id}")
        return None

    test_date_set = set(pd.to_datetime(test_dates).tolist())

    selected_small = selected[
        ["repeat_id", "combo_id", "tree_id", "node_id", "column_name", "beta"]
    ].copy()
    selected_small = selected_small.rename(columns={"node_id": "leaf_id"})

    leaf = leaf_all[
        (leaf_all["repeat_id"] == repeat_id)
        & (leaf_all[DATE_COL].isin(test_date_set))
    ].copy()

    if leaf.empty:
        print(f"[WARNING] Empty leaf weights after repeat/date filter: year={year}, repeat={repeat_id}")
        return None

    h = leaf.merge(
        selected_small,
        on=["repeat_id", "combo_id", "tree_id", "leaf_id"],
        how="inner",
        validate="many_to_many",
    )

    if h.empty:
        print(f"[WARNING] Empty selected leaf holdings: year={year}, repeat={repeat_id}")
        return None

    h["strategy_weight_component"] = h["beta"].astype(float) * h["stock_weight_in_leaf"].astype(float)

    strategy_h = (
        h
        .groupby([DATE_COL, ID_COL], as_index=False)["strategy_weight_component"]
        .sum()
        .rename(columns={"strategy_weight_component": "strategy_weight"})
    )

    strategy_h = strategy_h[np.abs(strategy_h["strategy_weight"]) > 1e-15].copy()

    if strategy_h.empty:
        print(f"[WARNING] Empty strategy holdings after aggregation: year={year}, repeat={repeat_id}")
        return None

    strategy_h["year"] = year
    strategy_h["repeat_id"] = repeat_id
    strategy_h["strategy_id"] = f"oos_{year}_repeat_{repeat_id:02d}"

    return strategy_h


# =========================================================
# ensemble holdings
# =========================================================
def compute_ensemble_holdings(strategy_holdings: pd.DataFrame) -> pd.DataFrame:
    """
    AP forest II ensemble holding.

    Missing stock in a repeat means zero weight, so for each date:
        ensemble_weight = sum_repeat weight / number_of_active_repeats
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

    This function also saves simple_turnover for debugging:
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
            # If a previously held stock has missing current return, fill 0 for robustness
            # and record the count for diagnostics.
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

                # Stocks not previously held have drifted weight 0.
                # Stocks previously held but not currently targeted are captured by fill_value=0.
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

    return pd.DataFrame(rows)


# =========================================================
# main
# =========================================================
def main(args: argparse.Namespace) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    years = parse_int_list(args.years)
    all_strategy_holdings: list[pd.DataFrame] = []

    for year in years:
        print("=" * 100)
        print(f"[INFO] Processing OOS year: {year}")

        split_year_dir = SPLIT_BASE / f"oos_{year}"
        filter_year_dir = FILTER_BASE / f"oos_{year}"
        prune_year_dir = PRUNE_BASE / f"oos_{year}"

        if not split_year_dir.exists():
            print(f"[WARNING] Missing split year dir: {split_year_dir}")
            continue

        if not filter_year_dir.exists():
            print(f"[WARNING] Missing filter year dir: {filter_year_dir}")
            continue

        if not prune_year_dir.exists():
            print(f"[WARNING] Missing prune year dir: {prune_year_dir}")
            continue

        try:
            leaf_all = load_test_leaf_weights(year)
        except Exception as e:
            print(f"[WARNING] Cannot load test leaf weights for {year}: {e}")
            continue

        print(f"[INFO] test_leaf_weights rows: {len(leaf_all):,}")

        for repeat_id in range(1, args.N + 1):
            repeat_tag = f"repeat_{repeat_id:02d}"

            filter_repeat_dir = filter_year_dir / repeat_tag
            prune_repeat_dir = prune_year_dir / repeat_tag

            if not filter_repeat_dir.exists():
                print(f"[WARNING] Missing filter repeat dir: {filter_repeat_dir}")
                continue

            if not prune_repeat_dir.exists():
                print(f"[WARNING] Missing prune repeat dir: {prune_repeat_dir}")
                continue

            one_h = reconstruct_strategy_holdings_one_oos_repeat(
                year=year,
                repeat_id=repeat_id,
                leaf_all=leaf_all,
                filter_repeat_dir=filter_repeat_dir,
                prune_repeat_dir=prune_repeat_dir,
                cvN=args.cvN,
            )

            if one_h is not None and not one_h.empty:
                all_strategy_holdings.append(one_h)

    if len(all_strategy_holdings) == 0:
        raise ValueError("No strategy holdings reconstructed. Please check ptree split/filter/prune paths.")

    strategy_holdings = pd.concat(all_strategy_holdings, axis=0, ignore_index=True)

    ensemble_holdings = compute_ensemble_holdings(strategy_holdings)

    all_dates = pd.to_datetime(ensemble_holdings[DATE_COL]).drop_duplicates().sort_values()
    years_needed = sorted(all_dates.dt.year.astype(int).unique().tolist())
    date_set = set(all_dates.tolist())

    stock_ret_df = read_stock_returns_for_years(years_needed)
    stock_ret_df = stock_ret_df[stock_ret_df[DATE_COL].isin(date_set)].copy()

    monthly_turnover = compute_monthly_turnover_drift_adjusted(
        ensemble_holdings=ensemble_holdings,
        stock_ret_df=stock_ret_df,
    )

    output_fp = OUTPUT_BASE / "monthly_turnover.csv"
    monthly_turnover.to_csv(output_fp, index=False)

    print("=" * 100)
    print("[INFO] AP forest II / ptree drift-adjusted turnover finished successfully")
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

    parser.add_argument("--years", type=str, default="1981-2024")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--cvN", type=int, default=3)

    args = parser.parse_args()
    main(args)