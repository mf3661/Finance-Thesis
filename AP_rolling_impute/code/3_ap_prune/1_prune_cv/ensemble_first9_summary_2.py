#!/usr/bin/env python3

import re
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# paths
# =========================
PRUNE_BASE = Path("/user/yw4389/Finance_Thesis/AP_rolling_impute/data/3_ap_prune/1_prune_cv/output")
FILTER_BASE = Path("/user/yw4389/Finance_Thesis/AP_rolling_impute/data/2_tree_construction/3_filter/output")
OUT_DIR = PRUNE_BASE

# =========================
# settings
# =========================
REPEATS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
WINDOWS = list(range(1, 45))   # 1..44

B = 10
DEPTH = 4
BASE_SEED = 0


# =========================
# helpers
# =========================
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        print(f"[WARNING] Empty CSV skipped: {path}")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to read {path}: {e}")
        return None


def annualized_return(r, periods_per_year=12):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    return (1 + r).prod() ** (periods_per_year / len(r)) - 1


def annualized_vol(r, periods_per_year=12):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    return r.std(ddof=0) * np.sqrt(periods_per_year)


def annualized_sharpe(r, periods_per_year=12):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    vol = r.std(ddof=0)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return np.sqrt(periods_per_year) * r.mean() / vol


def max_drawdown(r):
    r = pd.Series(r).fillna(0)
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return dd.min()


# =========================
# containers
# =========================
all_yearly_sr_records = []
all_monthly_ret_records = []


# =========================
# main loop
# =========================
for repeat_idx in REPEATS:
    seed = BASE_SEED + repeat_idx - 1
    repeat_tag = f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{DEPTH}"

    prune_repeat_dir = PRUNE_BASE / repeat_tag
    filter_repeat_dir = FILTER_BASE / repeat_tag

    print("=" * 100)
    print(f"Processing {repeat_tag}")

    # ---------------------------------
    # repeat-level filtered portfolio returns
    # ---------------------------------
    ret_path = filter_repeat_dir / "level_all_excess_ret_combined_filtered.pkl.gz"
    if not ret_path.exists():
        print(f"[WARNING] Missing return pickle: {ret_path}")
        continue

    try:
        ret_df = pd.read_pickle(ret_path).copy()
    except Exception as e:
        print(f"[WARNING] Failed to read pickle {ret_path}: {e}")
        continue

    if ret_df.empty:
        print(f"[WARNING] Empty return pickle: {ret_path}")
        continue

    if "date" not in ret_df.columns:
        print(f"[WARNING] 'date' column missing in {ret_path}")
        continue

    ret_df["date"] = pd.to_datetime(ret_df["date"])
    ret_df = ret_df.sort_values("date").reset_index(drop=True)

    # ---------------------------------
    # rolling summary
    # ---------------------------------
    summary_path = prune_repeat_dir / "rolling_window_summary.csv"
    summary_df = safe_read_csv(summary_path)
    if summary_df is None or summary_df.empty:
        print(f"[WARNING] Missing or invalid rolling summary: {summary_path}")
        continue

    required_summary_cols = {"window_idx", "test_start_date", "test_end_date"}
    if not required_summary_cols.issubset(summary_df.columns):
        print(f"[WARNING] Missing required columns in rolling summary: {summary_path}")
        continue

    summary_df["test_start_date"] = pd.to_datetime(summary_df["test_start_date"])
    summary_df["test_end_date"] = pd.to_datetime(summary_df["test_end_date"])

    # ---------------------------------
    # per window
    # ---------------------------------
    for window_idx in WINDOWS:
        window_tag = f"window_{window_idx:04d}"
        window_dir = prune_repeat_dir / window_tag

        if not window_dir.exists():
            print(f"[WARNING] Missing window dir: {window_dir}")
            continue

        # ------------------------------------------------
        # Step 1: scan cv_3 files, pick best valid_SR
        # ------------------------------------------------
        cv_files = sorted(window_dir.glob("results_cv_3_l0_*_l2_*.csv"))
        if len(cv_files) == 0:
            print(f"[WARNING] No cv_3 files in {window_dir}")
            continue

        best_cv = None

        for fp in cv_files:
            m = re.search(r"results_cv_3_l0_(\d+)_l2_(\d+)\.csv$", fp.name)
            if m is None:
                continue

            l0_idx = int(m.group(1))
            l2_idx = int(m.group(2))

            df = safe_read_csv(fp)
            if df is None or df.empty:
                continue

            needed = {"valid_SR", "lambda0", "lambda2", "portsN"}
            if not needed.issubset(df.columns):
                print(f"[WARNING] Missing required columns in {fp}")
                continue

            df = df[np.isfinite(df["valid_SR"])].copy()
            if df.empty:
                continue

            local_idx = df["valid_SR"].idxmax()
            row = df.loc[local_idx].copy()

            cand = {
                "repeat_idx": repeat_idx,
                "window_idx": window_idx,
                "l0_idx": l0_idx,
                "l2_idx": l2_idx,
                "lambda0": row["lambda0"],
                "lambda2": row["lambda2"],
                "portsN": int(row["portsN"]),
                "valid_SR": row["valid_SR"],
            }

            if (best_cv is None) or (cand["valid_SR"] > best_cv["valid_SR"]):
                best_cv = cand

        if best_cv is None:
            print(f"[WARNING] No valid candidate found in {window_dir}")
            continue

        # ------------------------------------------------
        # Step 2: matching row in results_full
        # ------------------------------------------------
        full_fp = window_dir / f"results_full_l0_{best_cv['l0_idx']}_l2_{best_cv['l2_idx']}.csv"
        if not full_fp.exists():
            print(f"[WARNING] Missing full file: {full_fp}")
            continue

        full_df = safe_read_csv(full_fp)
        if full_df is None or full_df.empty:
            continue

        needed = {"portsN", "train_SR", "test_SR"}
        if not needed.issubset(full_df.columns):
            print(f"[WARNING] Missing required columns in {full_fp}")
            continue

        full_sub = full_df[full_df["portsN"] == best_cv["portsN"]].copy()
        if full_sub.empty:
            print(f"[WARNING] No matching portsN={best_cv['portsN']} in {full_fp}")
            continue

        full_sub = full_sub[np.isfinite(full_sub["train_SR"])].copy()
        if full_sub.empty:
            print(f"[WARNING] Matching full rows all invalid in {full_fp}")
            continue

        full_row = full_sub.loc[full_sub["train_SR"].idxmax()].copy()

        # ------------------------------------------------
        # Step 3: yearly SR record
        # ------------------------------------------------
        one_summary = summary_df[summary_df["window_idx"] == window_idx].copy()
        if one_summary.empty:
            print(f"[WARNING] window_idx={window_idx} not found in rolling_window_summary")
            continue

        test_start = one_summary["test_start_date"].iloc[0]
        test_end = one_summary["test_end_date"].iloc[0]
        test_year = int(test_start.year)

        all_yearly_sr_records.append({
            "repeat_idx": repeat_idx,
            "window_idx": window_idx,
            "year": test_year,
            "best_lambda0_idx": best_cv["l0_idx"],
            "best_lambda2_idx": best_cv["l2_idx"],
            "best_lambda0": best_cv["lambda0"],
            "best_lambda2": best_cv["lambda2"],
            "best_portsN": best_cv["portsN"],
            "valid_SR": best_cv["valid_SR"],
            "train_SR_full": full_row["train_SR"],
            "test_SR_full": full_row["test_SR"],
        })

        # ------------------------------------------------
        # Step 4: monthly strategy return for this window
        # ------------------------------------------------
        meta_window_fp = window_dir / "portfolio_metadata_window.csv"
        if not meta_window_fp.exists():
            print(f"[WARNING] Missing metadata_window: {meta_window_fp}")
            continue

        meta_window = safe_read_csv(meta_window_fp)
        if meta_window is None or meta_window.empty:
            continue

        if "column_name" not in meta_window.columns or "node_id" not in meta_window.columns:
            print(f"[WARNING] Invalid metadata columns in {meta_window_fp}")
            continue

        colnames = meta_window["column_name"].tolist()

        beta_cols = [c for c in full_row.index if c.startswith("beta_")]
        beta_cols = sorted(beta_cols, key=lambda x: int(x.split("_")[1]))

        if len(beta_cols) != len(colnames):
            print(
                f"[WARNING] beta length mismatch in {window_dir}: "
                f"{len(beta_cols)} betas vs {len(colnames)} columns"
            )
            continue

        beta_vec = full_row[beta_cols].astype(float).to_numpy()

        node_depth = meta_window["node_id"].astype(str).str.len() - 1
        adj_w = 1.0 / np.sqrt(2.0 ** node_depth.to_numpy(dtype=float))
        eff_weight = beta_vec / adj_w

        mask = (ret_df["date"] >= test_start) & (ret_df["date"] <= test_end)
        needed_cols = ["date"] + colnames

        missing_cols = [c for c in colnames if c not in ret_df.columns]
        if missing_cols:
            print(f"[WARNING] Missing return columns in pickle for {window_dir}: first few {missing_cols[:5]}")
            continue

        test_ret_df = ret_df.loc[mask, needed_cols].copy().reset_index(drop=True)
        if test_ret_df.empty:
            print(f"[WARNING] No test-period returns for {window_dir}")
            continue

        X = test_ret_df[colnames].to_numpy(dtype=float)
        if np.isnan(X).any():
            print(f"[WARNING] NaN found in test-period return matrix for {window_dir}")
            continue

        strategy_ret = X @ eff_weight

        tmp_monthly = pd.DataFrame({
            "date": test_ret_df["date"].to_numpy(),
            "repeat_idx": repeat_idx,
            "window_idx": window_idx,
            "year": test_year,
            "strategy_ret": strategy_ret,
        })
        all_monthly_ret_records.append(tmp_monthly)


# =========================
# combine results
# =========================
yearly_sr_df = pd.DataFrame(all_yearly_sr_records)

if len(all_monthly_ret_records) > 0:
    monthly_ret_df = pd.concat(all_monthly_ret_records, axis=0, ignore_index=True)
else:
    monthly_ret_df = pd.DataFrame(columns=["date", "repeat_idx", "window_idx", "year", "strategy_ret"])

print("\nyearly_sr_df shape:", yearly_sr_df.shape)
print("monthly_ret_df shape:", monthly_ret_df.shape)

# -------------------------
# yearly SR outputs
# -------------------------
if not yearly_sr_df.empty:
    repeat_avg_sr = (
        yearly_sr_df.groupby("repeat_idx", as_index=False)["test_SR_full"]
        .mean()
        .rename(columns={"test_SR_full": "avg_test_SR_over_windows"})
    )

    year_avg_sr = (
        yearly_sr_df.groupby("year", as_index=False)["test_SR_full"]
        .mean()
        .rename(columns={"test_SR_full": "ensemble_avg_test_SR"})
    )
else:
    repeat_avg_sr = pd.DataFrame(columns=["repeat_idx", "avg_test_SR_over_windows"])
    year_avg_sr = pd.DataFrame(columns=["year", "ensemble_avg_test_SR"])

# -------------------------
# monthly return outputs
# -------------------------
if not monthly_ret_df.empty:
    ensemble_monthly_ret = (
        monthly_ret_df.groupby("date", as_index=False)["strategy_ret"]
        .mean()
        .rename(columns={"strategy_ret": "ensemble_ret"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    ensemble_monthly_ret["cum_log_ret"] = np.log1p(ensemble_monthly_ret["ensemble_ret"]).cumsum()
    ensemble_monthly_ret["cum_ret"] = (1 + ensemble_monthly_ret["ensemble_ret"]).cumprod() - 1
else:
    ensemble_monthly_ret = pd.DataFrame(columns=["date", "ensemble_ret", "cum_log_ret", "cum_ret"])

# -------------------------
# summary
# -------------------------
summary_rows = []

summary_rows.append(("n_repeats_used", len(REPEATS)))
summary_rows.append(("n_yearly_sr_records", len(yearly_sr_df)))
summary_rows.append(("n_monthly_return_obs", len(ensemble_monthly_ret)))

if not ensemble_monthly_ret.empty:
    r = ensemble_monthly_ret["ensemble_ret"]
    summary_rows.extend([
        ("mean_monthly_return", r.mean()),
        ("monthly_vol", r.std(ddof=0)),
        ("annualized_return", annualized_return(r, periods_per_year=12)),
        ("annualized_vol", annualized_vol(r, periods_per_year=12)),
        ("annualized_sharpe", annualized_sharpe(r, periods_per_year=12)),
        ("max_drawdown", max_drawdown(r)),
    ])
else:
    summary_rows.extend([
        ("mean_monthly_return", np.nan),
        ("monthly_vol", np.nan),
        ("annualized_return", np.nan),
        ("annualized_vol", np.nan),
        ("annualized_sharpe", np.nan),
        ("max_drawdown", np.nan),
    ])

if not year_avg_sr.empty:
    summary_rows.extend([
        ("mean_yearly_test_SR", year_avg_sr["ensemble_avg_test_SR"].mean()),
        ("median_yearly_test_SR", year_avg_sr["ensemble_avg_test_SR"].median()),
        ("std_yearly_test_SR", year_avg_sr["ensemble_avg_test_SR"].std(ddof=0)),
    ])
else:
    summary_rows.extend([
        ("mean_yearly_test_SR", np.nan),
        ("median_yearly_test_SR", np.nan),
        ("std_yearly_test_SR", np.nan),
    ])

# turnover not identified yet in the files shown
summary_rows.extend([
    ("mean_turnover_ratio", np.nan),
    ("median_turnover_ratio", np.nan),
])

summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

# =========================
# save outputs
# =========================
yearly_sr_df.to_csv(OUT_DIR / "ensemble_yearly_sr_detail.csv", index=False)
repeat_avg_sr.to_csv(OUT_DIR / "ensemble_repeat_avg_sr.csv", index=False)
year_avg_sr.to_csv(OUT_DIR / "ensemble_yearly_sr.csv", index=False)
monthly_ret_df.to_csv(OUT_DIR / "ensemble_monthly_ret_detail.csv", index=False)
ensemble_monthly_ret.to_csv(OUT_DIR / "ensemble_monthly_ret.csv", index=False)
summary_df.to_csv(OUT_DIR / "ensemble_summary.csv", index=False)

print("\nSaved files:")
print(OUT_DIR / "ensemble_yearly_sr_detail.csv")
print(OUT_DIR / "ensemble_repeat_avg_sr.csv")
print(OUT_DIR / "ensemble_yearly_sr.csv")
print(OUT_DIR / "ensemble_monthly_ret_detail.csv")
print(OUT_DIR / "ensemble_monthly_ret.csv")
print(OUT_DIR / "ensemble_summary.csv")

# =========================
# plots
# =========================
print("\nGenerating plots...")

if not ensemble_monthly_ret.empty:
    plt.figure(figsize=(14, 6))
    plt.plot(
        ensemble_monthly_ret["date"],
        ensemble_monthly_ret["cum_ret"],
        linewidth=2
    )
    plt.title("Ensemble Portfolio Cumulative Return (First 9 Repeats)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_cumulative_return.png", dpi=220)
    plt.close()

if not ensemble_monthly_ret.empty:
    plt.figure(figsize=(14, 6))
    plt.bar(
        ensemble_monthly_ret["date"],
        ensemble_monthly_ret["ensemble_ret"],
        width=25
    )
    plt.title("Monthly Ensemble Return")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_monthly_return.png", dpi=220)
    plt.close()

if not year_avg_sr.empty:
    plt.figure(figsize=(12, 6))
    plt.bar(
        year_avg_sr["year"].astype(str),
        year_avg_sr["ensemble_avg_test_SR"]
    )
    plt.title("Average Test Sharpe Ratio by Year")
    plt.xlabel("Year")
    plt.ylabel("Test Sharpe")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_yearly_sharpe.png", dpi=220)
    plt.close()

if not ensemble_monthly_ret.empty:
    wealth = (1 + ensemble_monthly_ret["ensemble_ret"]).cumprod()
    dd = wealth / wealth.cummax() - 1

    plt.figure(figsize=(14, 6))
    plt.fill_between(
        ensemble_monthly_ret["date"],
        dd,
        0,
        alpha=0.4
    )
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot_drawdown.png", dpi=220)
    plt.close()

print("Plots saved.")
print(OUT_DIR / "plot_cumulative_return.png")
print(OUT_DIR / "plot_monthly_return.png")
print(OUT_DIR / "plot_yearly_sharpe.png")
print(OUT_DIR / "plot_drawdown.png")

print("\nSummary:")
print(summary_df)