#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# User settings
# ============================================================
BASE_DIR = Path("/user/yw4389/Finance_Thesis/AP_rolling_impute/data/3_ap_prune/1_prune_cv/output")
N_REPEATS = 9

DISCOVERY_OUT = BASE_DIR / f"ensemble_first_{N_REPEATS}_repeats_file_inventory.csv"
TS_OUT = BASE_DIR / f"ensemble_first_{N_REPEATS}_repeats_timeseries.csv"
SUMMARY_OUT = BASE_DIR / f"ensemble_first_{N_REPEATS}_repeats_summary.csv"


# ============================================================
# Helpers
# ============================================================
def find_col(columns, candidates, exact_first=True):
    cols = list(columns)
    cols_lower = {c.lower(): c for c in cols}

    if exact_first:
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]

    for cand in candidates:
        cl = cand.lower()
        for c in cols:
            if cl in c.lower():
                return c
    return None


def safe_read_csv(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
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


def sharpe_ratio(r, rf=0.0, periods_per_year=12):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return np.nan
    ex = r - rf / periods_per_year
    vol = ex.std(ddof=0)
    if pd.isna(vol) or vol == 0:
        return np.nan
    return ex.mean() / vol * np.sqrt(periods_per_year)


def max_drawdown(r):
    r = pd.Series(r).fillna(0)
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return dd.min()


def infer_periods_per_year(date_series):
    s = pd.Series(pd.to_datetime(date_series)).dropna().sort_values().unique()
    if len(s) < 3:
        return 12
    diffs = pd.Series(s[1:]) - pd.Series(s[:-1])
    median_days = np.median([d / np.timedelta64(1, "D") for d in diffs])

    if median_days <= 7:
        return 252
    elif median_days <= 40:
        return 12
    elif median_days <= 100:
        return 4
    else:
        return 1


def summarize_metric(ts_df, date_col, ret_col=None, turnover_col=None, k_col=None, lambda_col=None):
    periods_per_year = infer_periods_per_year(ts_df[date_col]) if date_col in ts_df.columns else 12

    rows = [
        ("n_time_points", len(ts_df)),
        ("date_col", date_col),
        ("periods_per_year", periods_per_year),
    ]

    if ret_col is not None and ret_col in ts_df.columns:
        r = ts_df[ret_col].dropna()
        rows.extend([
            ("mean_portfolio_ret", r.mean() if len(r) else np.nan),
            ("std_portfolio_ret", r.std(ddof=0) if len(r) else np.nan),
            ("annualized_return", annualized_return(r, periods_per_year)),
            ("annualized_vol", annualized_vol(r, periods_per_year)),
            ("sharpe", sharpe_ratio(r, periods_per_year=periods_per_year)),
            ("max_drawdown", max_drawdown(r)),
        ])

    if turnover_col is not None and turnover_col in ts_df.columns:
        rows.extend([
            ("mean_turnover", ts_df[turnover_col].mean()),
            ("median_turnover", ts_df[turnover_col].median()),
        ])

    if k_col is not None and k_col in ts_df.columns:
        rows.extend([
            ("mean_K", ts_df[k_col].mean()),
            ("median_K", ts_df[k_col].median()),
        ])

    if lambda_col is not None and lambda_col in ts_df.columns:
        rows.extend([
            ("mean_lambda", ts_df[lambda_col].mean()),
            ("median_lambda", ts_df[lambda_col].median()),
        ])

    return pd.DataFrame(rows, columns=["metric", "value"])


# ============================================================
# Discovery functions
# ============================================================
def discover_repeat_files(repeat_dirs):
    rows = []

    for rpt in repeat_dirs:
        for f in sorted(rpt.rglob("*.csv")):
            df_head = safe_read_csv(f, nrows=5)
            if df_head is None:
                rows.append({
                    "repeat_name": rpt.name,
                    "relative_path": str(f.relative_to(rpt)),
                    "filename": f.name,
                    "readable": False,
                    "ncols": np.nan,
                    "columns": None,
                })
            else:
                rows.append({
                    "repeat_name": rpt.name,
                    "relative_path": str(f.relative_to(rpt)),
                    "filename": f.name,
                    "readable": True,
                    "ncols": len(df_head.columns),
                    "columns": " | ".join(df_head.columns.astype(str).tolist()),
                })

    inv = pd.DataFrame(rows)
    return inv


def load_root_files(repeat_dirs):
    rolling_list = []
    meta_list = []

    for i, rpt in enumerate(repeat_dirs, start=1):
        roll_f = rpt / "rolling_window_summary.csv"
        meta_f = rpt / "portfolio_metadata_filtered.csv"

        if roll_f.exists():
            df = pd.read_csv(roll_f)
            df["repeat_idx"] = i
            df["repeat_name"] = rpt.name
            rolling_list.append(df)

        if meta_f.exists():
            df = pd.read_csv(meta_f)
            df["repeat_idx"] = i
            df["repeat_name"] = rpt.name
            meta_list.append(df)

    rolling = pd.concat(rolling_list, ignore_index=True) if rolling_list else pd.DataFrame()
    meta = pd.concat(meta_list, ignore_index=True) if meta_list else pd.DataFrame()

    return rolling, meta


def make_base_window_calendar(rolling):
    if rolling.empty:
        return pd.DataFrame()

    out = rolling.copy()

    for c in ["train_valid_start_date", "train_valid_end_date", "test_start_date", "test_end_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    # prefer test_end_date as portfolio evaluation date
    date_col = None
    for cand in ["test_end_date", "test_start_date", "train_valid_end_date", "train_valid_start_date"]:
        if cand in out.columns:
            date_col = cand
            break

    if date_col is None:
        raise ValueError("No usable date column found in rolling_window_summary.csv")

    keep_cols = [c for c in [
        "window_idx", "window_id",
        "train_valid_start_date", "train_valid_end_date",
        "test_start_date", "test_end_date",
        "repeat_idx", "repeat_name"
    ] if c in out.columns]

    out = out[keep_cols].copy()
    out = out.rename(columns={date_col: "date"})
    return out


# ============================================================
# Attempt automatic extraction from portfolio_metadata_filtered
# ============================================================
def build_timeseries_from_metadata(rolling, meta):
    """
    Best-effort constructor.

    What it tries:
    1. Merge root metadata with rolling calendar by repeat/window.
    2. If metadata has direct columns for return / turnover / lambda, aggregate them.
    3. If metadata lacks K, compute K as number of rows per repeat-window.
    """
    if rolling.empty or meta.empty:
        return None, None

    cal = make_base_window_calendar(rolling)

    join_keys = [c for c in ["repeat_idx", "repeat_name", "window_idx", "window_id"] if c in cal.columns and c in meta.columns]
    if len(join_keys) == 0:
        return None, None

    df = meta.merge(cal, on=join_keys, how="left")

    # detect columns inside portfolio_metadata_filtered.csv
    ret_col = find_col(df.columns, [
        "portfolio_ret", "port_ret", "return", "ret", "oos_ret", "portfolio_return"
    ])

    turnover_col = find_col(df.columns, [
        "turnover", "tover"
    ])

    lambda_col = find_col(df.columns, [
        "lambda", "lamb"
    ])

    # K can often be inferred even if not explicit
    k_col_direct = find_col(df.columns, [
        "K", "k", "num_portfolios", "n_portfolios", "portfolio_count"
    ])

    # build per-repeat per-window aggregates first
    grp_keys = [c for c in ["repeat_idx", "repeat_name", "window_idx", "window_id", "date"] if c in df.columns]

    agg_map = {}
    if ret_col is not None:
        agg_map[ret_col] = "mean"
    if turnover_col is not None:
        agg_map[turnover_col] = "mean"
    if lambda_col is not None:
        agg_map[lambda_col] = "mean"
    if k_col_direct is not None:
        agg_map[k_col_direct] = "mean"

    if len(agg_map) > 0:
        per_repeat_window = df.groupby(grp_keys, as_index=False).agg(agg_map)
    else:
        per_repeat_window = df[grp_keys].drop_duplicates().copy()

    # infer K if missing
    if k_col_direct is None:
        k_count = (
            df.groupby(grp_keys, as_index=False)
              .size()
              .rename(columns={"size": "K"})
        )
        per_repeat_window = per_repeat_window.merge(k_count, on=grp_keys, how="left")
        k_col_final = "K"
    else:
        per_repeat_window = per_repeat_window.rename(columns={k_col_direct: "K"})
        k_col_final = "K"

    rename_map = {}
    if ret_col is not None:
        rename_map[ret_col] = "portfolio_ret"
    if turnover_col is not None:
        rename_map[turnover_col] = "turnover"
    if lambda_col is not None:
        rename_map[lambda_col] = "lambda"

    per_repeat_window = per_repeat_window.rename(columns=rename_map)

    # ensemble average across first 9 repeats by date
    avg_cols = [c for c in ["portfolio_ret", "turnover", "K", "lambda"] if c in per_repeat_window.columns]
    if len(avg_cols) == 0:
        return None, df

    ts = (
        per_repeat_window.groupby("date", as_index=False)[avg_cols]
        .mean()
        .sort_values("date")
        .reset_index(drop=True)
    )

    return ts, df


# ============================================================
# Main
# ============================================================
def main():
    repeat_dirs = sorted([
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("repeat_")
    ])[:N_REPEATS]

    if len(repeat_dirs) == 0:
        raise FileNotFoundError(f"No repeat_* folders found under {BASE_DIR}")

    print("Using repeat folders:")
    for p in repeat_dirs:
        print("  ", p.name)

    # 1) discovery
    inventory = discover_repeat_files(repeat_dirs)
    inventory.to_csv(DISCOVERY_OUT, index=False)
    print(f"\nSaved file inventory to:\n  {DISCOVERY_OUT}")

    # 2) load root-level files
    rolling, meta = load_root_files(repeat_dirs)

    print("\nrolling_window_summary columns:")
    print(rolling.columns.tolist() if not rolling.empty else "[missing]")

    print("\nportfolio_metadata_filtered columns:")
    print(meta.columns.tolist() if not meta.empty else "[missing]")

    # 3) try best-effort construction
    ts, merged_detail = build_timeseries_from_metadata(rolling, meta)

    if ts is None:
        # still produce a minimal summary explaining why
        summary_df = pd.DataFrame([
            ("status", "failed_to_construct_metrics"),
            ("reason", "No direct return/turnover/lambda columns were found in portfolio_metadata_filtered.csv; inspect the file inventory to locate the correct per-window files."),
            ("inventory_file", str(DISCOVERY_OUT)),
        ], columns=["metric", "value"])

        summary_df.to_csv(SUMMARY_OUT, index=False)
        print("\nCould not auto-construct time series.")
        print(f"Saved diagnostic summary to:\n  {SUMMARY_OUT}")
        print("\nNext step: open the inventory CSV and look for files under window_XXXX/ that contain return/weight/lambda information.")
        return

    summary_df = summarize_metric(
        ts_df=ts,
        date_col="date",
        ret_col="portfolio_ret" if "portfolio_ret" in ts.columns else None,
        turnover_col="turnover" if "turnover" in ts.columns else None,
        k_col="K" if "K" in ts.columns else None,
        lambda_col="lambda" if "lambda" in ts.columns else None,
    )

    ts.to_csv(TS_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print(f"\nSaved time series to:\n  {TS_OUT}")
    print(f"Saved summary to:\n  {SUMMARY_OUT}")

    print("\nTime-series preview:")
    print(ts.head())

    print("\nSummary:")
    print(summary_df)


if __name__ == "__main__":
    main()