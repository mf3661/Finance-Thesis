from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
FILTER_BASE = Path("../../../data/2_tree_construction/2_3_filter/2_3_5_filter_ptree/output")
PRUNE_BASE = Path("../../../data/3_ap_prune/3_1_prune_cv/3_1_5_prune_cv_ptree/output")

OUTPUT_BASE = Path("../../../data/4_results/4_1_pnl/4_1_5_ap_forestII/output")

DATE_COL = "date"


# =========================================================
# helpers
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


def choose_best_cv_row(window_dir: Path, cvN: int) -> dict | None:
    """
    Scan all results_cv_{cvN}_l0_*_l2_*.csv files.
    Pick the row with the highest valid_SR.
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

    If multiple rows have the same portsN, choose max train_SR.
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


def compute_one_oos_repeat_monthly_ret(
    ret_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    window_dir: Path,
    year: int,
    repeat_idx: int,
    cvN: int,
) -> pd.DataFrame | None:
    """
    Compute monthly OOS returns for one AP_forestII oos-year and repeat.
    AP_forestII has only one fixed window: window_0001.
    """
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

    one_summary = summary_df[summary_df["window_idx"] == 1].copy()
    if one_summary.empty:
        print(f"[WARNING] window_idx=1 not found in rolling_window_summary: {window_dir}")
        return None

    test_start = pd.to_datetime(one_summary["test_start_date"].iloc[0])
    test_end = pd.to_datetime(one_summary["test_end_date"].iloc[0])

    meta_window_fp = window_dir / "portfolio_metadata_window.csv"
    if not meta_window_fp.exists():
        print(f"[WARNING] Missing portfolio_metadata_window.csv: {meta_window_fp}")
        return None

    meta_window = pd.read_csv(meta_window_fp).copy()
    colnames = meta_window["column_name"].astype(str).tolist()

    missing_cols = [c for c in colnames if c not in ret_df.columns]
    if len(missing_cols) > 0:
        print(
            f"[WARNING] Missing {len(missing_cols)} columns in return df. "
            f"Example: {missing_cols[:5]}"
        )
        return None

    beta_cols = get_beta_cols(full_row)

    if len(beta_cols) != len(colnames):
        print(
            f"[WARNING] beta length mismatch in {window_dir}: "
            f"{len(beta_cols)} beta columns vs {len(colnames)} metadata columns"
        )
        return None

    beta_vec = full_row[beta_cols].astype(float).to_numpy()

    # =====================================================
    # Important:
    # AP_forestII prune saves beta_* as final raw weights.
    # Therefore, do NOT divide by adj_w here.
    # Directly apply beta_vec to raw filtered portfolio returns.
    # =====================================================
    mask = (ret_df[DATE_COL] >= test_start) & (ret_df[DATE_COL] <= test_end)
    test_ret_df = ret_df.loc[mask, [DATE_COL] + colnames].copy().reset_index(drop=True)

    if test_ret_df.empty:
        print(f"[WARNING] Empty test return df: year={year}, repeat={repeat_idx}")
        return None

    X = test_ret_df[colnames].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        n_nan = int(np.isnan(X).sum())
        print(f"[WARNING] NaN found in test return matrix: year={year}, repeat={repeat_idx}, n_nan={n_nan}")
        return None

    strategy_ret = X @ beta_vec

    out = pd.DataFrame({
        "date": test_ret_df[DATE_COL].to_numpy(),
        "year": year,
        "repeat_idx": repeat_idx,
        "window_idx": 1,
        "strategy_ret": strategy_ret,

        # model-selection info
        "best_lambda0_idx": best_cv["l0_idx"],
        "best_lambda2_idx": best_cv["l2_idx"],
        "best_lambda0": best_cv["lambda0"],
        "best_lambda2": best_cv["lambda2"],
        "best_portsN": best_cv["portsN"],
        "valid_SR": best_cv["valid_SR"],
        "train_SR_full": float(full_row["train_SR"]),
        "test_SR_full": float(full_row["test_SR"]),
    })

    return out


# =========================================================
# main
# =========================================================
def main(args: argparse.Namespace) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    years = parse_int_list(args.years)

    all_monthly_ret_records: list[pd.DataFrame] = []

    for year in years:
        filter_year_dir = FILTER_BASE / f"oos_{year}"
        prune_year_dir = PRUNE_BASE / f"oos_{year}"

        print("=" * 100)
        print(f"[INFO] Processing OOS year: {year}")
        print(f"[INFO] Filter year dir: {filter_year_dir}")
        print(f"[INFO] Prune year dir : {prune_year_dir}")

        if not filter_year_dir.exists():
            print(f"[WARNING] Missing filter year dir: {filter_year_dir}")
            continue

        if not prune_year_dir.exists():
            print(f"[WARNING] Missing prune year dir: {prune_year_dir}")
            continue

        for repeat_idx in range(1, args.N + 1):
            repeat_tag = f"repeat_{repeat_idx:02d}"

            filter_repeat_dir = filter_year_dir / repeat_tag
            prune_repeat_dir = prune_year_dir / repeat_tag
            window_dir = prune_repeat_dir / "window_0001"

            if not filter_repeat_dir.exists():
                print(f"[WARNING] Missing filter repeat dir: {filter_repeat_dir}")
                continue

            if not window_dir.exists():
                print(f"[WARNING] Missing prune window dir: {window_dir}")
                continue

            ret_path = filter_repeat_dir / "level_all_excess_ret_combined_filtered.pkl.gz"
            if not ret_path.exists():
                print(f"[WARNING] Missing filtered return file: {ret_path}")
                continue

            summary_path = prune_repeat_dir / "rolling_window_summary.csv"
            if not summary_path.exists():
                print(f"[WARNING] Missing rolling_window_summary.csv: {summary_path}")
                continue

            ret_df = pd.read_pickle(ret_path).copy()
            ret_df[DATE_COL] = pd.to_datetime(ret_df[DATE_COL])
            ret_df = ret_df.sort_values(DATE_COL).reset_index(drop=True)

            summary_df = pd.read_csv(summary_path)
            summary_df["test_start_date"] = pd.to_datetime(summary_df["test_start_date"])
            summary_df["test_end_date"] = pd.to_datetime(summary_df["test_end_date"])

            one_ret = compute_one_oos_repeat_monthly_ret(
                ret_df=ret_df,
                summary_df=summary_df,
                window_dir=window_dir,
                year=year,
                repeat_idx=repeat_idx,
                cvN=args.cvN,
            )

            if one_ret is not None and not one_ret.empty:
                all_monthly_ret_records.append(one_ret)

    if len(all_monthly_ret_records) == 0:
        raise ValueError("No monthly return records were computed. Please check paths/years/N.")

    monthly_ret_df = pd.concat(all_monthly_ret_records, axis=0, ignore_index=True)
    monthly_ret_df = monthly_ret_df.sort_values(["date", "year", "repeat_idx"]).reset_index(drop=True)

    # Save all year-repeat monthly returns
    monthly_ret_path = OUTPUT_BASE / "monthly_strategy_returns_by_oos_repeat.csv"
    monthly_ret_df.to_csv(monthly_ret_path, index=False)

    # AP_forestII ensemble monthly return:
    # For each OOS month, average across available repeats.
    ensemble_monthly_ret = (
        monthly_ret_df
        .groupby("date", as_index=False)
        .agg(
            ensemble_ret=("strategy_ret", "mean"),
            n_contributors=("strategy_ret", "size"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    ensemble_ret_path = OUTPUT_BASE / "ensemble_monthly_returns.csv"
    ensemble_monthly_ret.to_csv(ensemble_ret_path, index=False)

    # Optional but useful: selection summary at year-repeat level
    selection_cols = [
        "year",
        "repeat_idx",
        "best_lambda0_idx",
        "best_lambda2_idx",
        "best_lambda0",
        "best_lambda2",
        "best_portsN",
        "valid_SR",
        "train_SR_full",
        "test_SR_full",
    ]

    selection_summary = (
        monthly_ret_df[selection_cols]
        .drop_duplicates()
        .sort_values(["year", "repeat_idx"])
        .reset_index(drop=True)
    )

    selection_summary_path = OUTPUT_BASE / "selection_summary_by_oos_repeat.csv"
    selection_summary.to_csv(selection_summary_path, index=False)

    print("=" * 100)
    print("[INFO] AP_forestII PnL computation finished successfully")
    print(f"[INFO] Saved repeat-level monthly returns to: {monthly_ret_path}")
    print(f"[INFO] Saved ensemble monthly returns to    : {ensemble_ret_path}")
    print(f"[INFO] Saved selection summary to          : {selection_summary_path}")
    print(f"[INFO] monthly_ret_df shape       : {monthly_ret_df.shape}")
    print(f"[INFO] ensemble_monthly_ret shape : {ensemble_monthly_ret.shape}")
    print(f"[INFO] selection_summary shape    : {selection_summary.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # OOS years, for example:
    #   --years 1981-2024
    #   --years 1981,1982,1983
    parser.add_argument("--years", type=str, default="1981-2024")

    # Number of repeats under each oos_year folder:
    #   repeat_01, repeat_02, ..., repeat_N
    parser.add_argument("--N", type=int, default=10)

    parser.add_argument("--cvN", type=int, default=3)

    args = parser.parse_args()
    main(args)