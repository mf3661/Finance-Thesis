from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
IPCA_OUTPUT_BASE = Path("../../../data/3_IPCA/3_1_run_IPCA/3_1_1_ipca/output")

PNL_OUTPUT_BASE = Path("../../../data/4_results/4_1_pnl/4_1_6_IPCA/output")

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


def load_one_year_ipca_return(year: int) -> pd.DataFrame | None:
    """
    Load one OOS year's IPCA long-short monthly return.

    Expected input:
        ../../../data/3_IPCA/3_1_run_IPCA/3_1_1_ipca/output/oos_ls_ipca_YYYY.csv

    Expected core columns:
        date, long_ret, short_ret, ls_ret
    """
    fp = IPCA_OUTPUT_BASE / f"oos_ls_ipca_{year}.csv"

    if not fp.exists():
        print(f"[WARNING] Missing IPCA OOS return file: {fp}")
        return None

    df = pd.read_csv(fp)

    if df.empty:
        print(f"[WARNING] Empty IPCA OOS return file: {fp}")
        return None

    required_cols = [DATE_COL, "ls_ret"]
    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing columns {missing} in {fp}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["ls_ret"] = pd.to_numeric(df["ls_ret"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[DATE_COL, "ls_ret"]).copy()

    if df.empty:
        print(f"[WARNING] No valid rows after cleaning: {fp}")
        return None

    # Standardize to AP-forest-style naming.
    out = pd.DataFrame({
        DATE_COL: df[DATE_COL],
        "year": year,
        "strategy_ret": df["ls_ret"],
    })

    # Keep useful diagnostic columns if available.
    optional_cols = [
        "long_ret",
        "short_ret",
        "n_long",
        "n_short",
        "best_n_factors",
        "val_mse",
        "n_features",
        "top_q",
        "bot_q",
        "weighting",
        "prediction",
        "ipca_intercept",
        "ipca_max_iter",
        "x_scaling",
        "gross_exposure",
        "net_exposure",
    ]

    for c in optional_cols:
        if c in df.columns:
            out[c] = df[c]

    out = out.sort_values(DATE_COL).reset_index(drop=True)

    return out


# =========================================================
# main
# =========================================================
def main(args: argparse.Namespace) -> None:
    PNL_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    years = parse_int_list(args.years)

    all_monthly_records: list[pd.DataFrame] = []

    for year in years:
        print("=" * 100)
        print(f"[INFO] Loading IPCA OOS return for year {year}")

        one_year = load_one_year_ipca_return(year)

        if one_year is not None and not one_year.empty:
            all_monthly_records.append(one_year)

    if len(all_monthly_records) == 0:
        raise ValueError("No IPCA monthly return records were loaded. Please check input path and years.")

    monthly_ret_df = pd.concat(all_monthly_records, axis=0, ignore_index=True)
    monthly_ret_df = monthly_ret_df.sort_values(DATE_COL).reset_index(drop=True)

    # Save IPCA monthly strategy returns.
    monthly_ret_path = PNL_OUTPUT_BASE / "monthly_strategy_returns.csv"
    monthly_ret_df.to_csv(monthly_ret_path, index=False)

    # Same format as AP forest PnL output:
    # date, ensemble_ret, n_contributors
    #
    # IPCA only has one strategy return per month, but we still group by date
    # to keep the output interface identical.
    ensemble_monthly_ret = (
        monthly_ret_df
        .groupby(DATE_COL, as_index=False)
        .agg(
            ensemble_ret=("strategy_ret", "mean"),
            n_contributors=("strategy_ret", "size"),
        )
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    ensemble_ret_path = PNL_OUTPUT_BASE / "ensemble_monthly_returns.csv"
    ensemble_monthly_ret.to_csv(ensemble_ret_path, index=False)

    # Optional useful summary.
    mean_ret = ensemble_monthly_ret["ensemble_ret"].mean()
    std_ret = ensemble_monthly_ret["ensemble_ret"].std(ddof=1)

    if std_ret > 0:
        monthly_sr = mean_ret / std_ret
        annualized_sr = np.sqrt(12.0) * monthly_sr
    else:
        monthly_sr = np.nan
        annualized_sr = np.nan

    summary_df = pd.DataFrame([{
        "start_date": ensemble_monthly_ret[DATE_COL].min(),
        "end_date": ensemble_monthly_ret[DATE_COL].max(),
        "n_months": len(ensemble_monthly_ret),
        "mean_monthly_ret": mean_ret,
        "std_monthly_ret": std_ret,
        "monthly_sr": monthly_sr,
        "annualized_sr": annualized_sr,
        "annualized_mean_ret": 12.0 * mean_ret,
        "cum_log_ret_final": float(np.log1p(ensemble_monthly_ret["ensemble_ret"]).sum()),
    }])

    summary_path = PNL_OUTPUT_BASE / "pnl_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 100)
    print("[INFO] IPCA PnL computation finished successfully")
    print(f"[INFO] Saved monthly strategy returns to: {monthly_ret_path}")
    print(f"[INFO] Saved ensemble monthly returns to : {ensemble_ret_path}")
    print(f"[INFO] Saved PnL summary to              : {summary_path}")
    print(f"[INFO] monthly_ret_df shape       : {monthly_ret_df.shape}")
    print(f"[INFO] ensemble_monthly_ret shape : {ensemble_monthly_ret.shape}")
    print(f"[INFO] Date range: {ensemble_monthly_ret[DATE_COL].min()} -> {ensemble_monthly_ret[DATE_COL].max()}")
    print(f"[INFO] Annualized SR: {annualized_sr:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--years",
        type=str,
        default="1981-2024",
        help='OOS years, e.g. "1981-2024" or "1981,1982,1983".',
    )

    args = parser.parse_args()
    main(args)