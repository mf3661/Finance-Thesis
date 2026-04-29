from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Paths
# =========================================================
RANK_FEATURE_BASE = Path("../../../data/feature_construction/1_4_rank_feature/by_year/impute")

IPCA_OUTPUT_BASE = Path("../../../data/3_IPCA/3_1_run_IPCA/3_1_1_ipca/output")

TURNOVER_OUTPUT_BASE = Path("../../../data/4_results/4_4_turnover/4_4_4_turnover_IPCA/output")

DATE_COL = "date"
ID_COL = "permno"
RET_COL = "ret"


# =========================================================
# Helpers
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


def read_stock_returns_for_years(years: list[int]) -> pd.DataFrame:
    """
    Read full stock-level returns from the same ranked/imputed by-year files
    used by the new IPCA pipeline.

    This is needed for drift-adjusted turnover because a stock held at t may
    not be selected at t+1, so current IPCA weight files alone are not enough.
    """
    dfs: list[pd.DataFrame] = []

    for year in years:
        fp = RANK_FEATURE_BASE / f"{year}.parquet"

        if not fp.exists():
            raise FileNotFoundError(f"Cannot find ranked feature file: {fp}")

        df = pd.read_parquet(fp, columns=[DATE_COL, ID_COL, RET_COL]).copy()

        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df[RET_COL] = pd.to_numeric(df[RET_COL], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[DATE_COL, ID_COL])
        df = df.drop_duplicates(subset=[DATE_COL, ID_COL])

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No stock return data loaded.")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=[DATE_COL, ID_COL])
    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return out


def load_one_year_ipca_weights(year: int) -> pd.DataFrame | None:
    """
    Load IPCA OOS stock weights for one OOS year.

    Expected input:
        oos_stock_weights_ipca_YYYY.csv

    Expected core columns:
        date, permno, strategy_weight

    strategy_weight:
        positive for long leg
        negative for short leg
    """
    fp = IPCA_OUTPUT_BASE / f"oos_stock_weights_ipca_{year}.csv"

    if not fp.exists():
        print(f"[WARNING] Missing IPCA stock weights file: {fp}")
        return None

    df = pd.read_csv(fp)

    if df.empty:
        print(f"[WARNING] Empty IPCA stock weights file: {fp}")
        return None

    required_cols = [DATE_COL, ID_COL, "strategy_weight"]
    missing = [c for c in required_cols if c not in df.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing columns {missing} in {fp}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["strategy_weight"] = pd.to_numeric(df["strategy_weight"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[DATE_COL, ID_COL, "strategy_weight"]).copy()

    if df.empty:
        print(f"[WARNING] No valid IPCA weight rows after cleaning: {fp}")
        return None

    out = (
        df.groupby([DATE_COL, ID_COL], as_index=False)["strategy_weight"]
        .sum()
        .rename(columns={"strategy_weight": "weight"})
    )

    out["year"] = year
    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return out


def load_ipca_weights(years: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for year in years:
        one = load_one_year_ipca_weights(year)
        if one is not None and not one.empty:
            parts.append(one)

    if len(parts) == 0:
        raise ValueError("No IPCA stock weights were loaded. Please check IPCA output path and years.")

    weights = pd.concat(parts, axis=0, ignore_index=True)
    weights[DATE_COL] = pd.to_datetime(weights[DATE_COL])

    weights = (
        weights.groupby([DATE_COL, ID_COL], as_index=False)["weight"]
        .sum()
        .sort_values([DATE_COL, ID_COL])
        .reset_index(drop=True)
    )

    return weights


# =========================================================
# Turnover
# =========================================================
def compute_monthly_turnover_drift_adjusted(
    holdings: pd.DataFrame,
    stock_ret_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Drift-adjusted turnover:

        turnover_{t+1}
        = 0.5 * sum_i | w_{i,t+1}
            - w_{i,t} * (1 + r_{i,t+1})
              / (1 + sum_j w_{j,t} r_{j,t+1}) |

    Here:
        w_{i,t}      = post-rebalance stock weight at date t
        r_{i,t+1}    = stock return over next period, recorded at date t+1
        w_{i,t+1}    = new post-rebalance stock weight at date t+1

    The first month has no previous portfolio, so turnover is NaN.

    simple_turnover is also saved for diagnostics:
        simple_turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|
    """
    h = holdings.copy()
    h[DATE_COL] = pd.to_datetime(h[DATE_COL])
    h["weight"] = pd.to_numeric(h["weight"], errors="coerce")
    h = h.replace([np.inf, -np.inf], np.nan)
    h = h.dropna(subset=[DATE_COL, ID_COL, "weight"]).copy()

    h = (
        h.groupby([DATE_COL, ID_COL], as_index=False)["weight"]
        .sum()
        .sort_values([DATE_COL, ID_COL])
        .reset_index(drop=True)
    )

    stock_ret = stock_ret_df[[DATE_COL, ID_COL, RET_COL]].copy()
    stock_ret[DATE_COL] = pd.to_datetime(stock_ret[DATE_COL])
    stock_ret[RET_COL] = pd.to_numeric(stock_ret[RET_COL], errors="coerce")
    stock_ret = stock_ret.replace([np.inf, -np.inf], np.nan)
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

        if prev_w is None:
            turnover = np.nan
            simple_turnover = np.nan
            portfolio_ret_from_prev = np.nan
            n_prev_stocks = np.nan
            n_missing_prev_ret = np.nan
        else:
            r_df = stock_ret[stock_ret[DATE_COL] == d].copy()
            r = r_df.set_index(ID_COL)[RET_COL].astype(float)

            r_prev = r.reindex(prev_w.index)
            n_missing_prev_ret = int(r_prev.isna().sum())

            # If return is missing for a previously held stock, we cannot
            # observe its exact drift. Fill with 0 for robustness and report
            # the count in n_missing_prev_ret.
            r_prev = r_prev.fillna(0.0)

            n_prev_stocks = int(prev_w.shape[0])

            portfolio_ret_from_prev = float((prev_w * r_prev).sum())
            denom = 1.0 + portfolio_ret_from_prev

            if not np.isfinite(denom) or abs(denom) < 1e-12:
                turnover = np.nan
            else:
                prev_w_drifted = prev_w * (1.0 + r_prev) / denom
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
        })

        prev_w = cur_w

    return pd.DataFrame(rows)


# =========================================================
# Main
# =========================================================
def main(args: argparse.Namespace) -> None:
    TURNOVER_OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    years = parse_int_list(args.years)

    print("=" * 100)
    print("[INFO] Loading IPCA stock weights")
    print(f"[INFO] Years: {years[0]}-{years[-1]}")
    print(f"[INFO] IPCA output base: {IPCA_OUTPUT_BASE}")
    print(f"[INFO] Turnover output base: {TURNOVER_OUTPUT_BASE}")

    holdings = load_ipca_weights(years)

    all_dates = pd.to_datetime(holdings[DATE_COL]).drop_duplicates().sort_values()
    years_needed = sorted(all_dates.dt.year.astype(int).unique().tolist())
    date_set = set(all_dates.tolist())

    print("=" * 100)
    print("[INFO] Loading stock returns for drift adjustment")
    print(f"[INFO] Years needed for stock returns: {years_needed[0]}-{years_needed[-1]}")
    print(f"[INFO] Rank feature base: {RANK_FEATURE_BASE}")

    stock_ret_df = read_stock_returns_for_years(years_needed)
    stock_ret_df = stock_ret_df[stock_ret_df[DATE_COL].isin(date_set)].copy()

    monthly_turnover = compute_monthly_turnover_drift_adjusted(
        holdings=holdings,
        stock_ret_df=stock_ret_df,
    )

    output_fp = TURNOVER_OUTPUT_BASE / "monthly_turnover.csv"
    monthly_turnover.to_csv(output_fp, index=False)

    summary = pd.DataFrame([{
        "start_date": monthly_turnover[DATE_COL].min(),
        "end_date": monthly_turnover[DATE_COL].max(),
        "n_months": len(monthly_turnover),
        "avg_turnover": monthly_turnover["turnover"].dropna().mean(),
        "avg_simple_turnover": monthly_turnover["simple_turnover"].dropna().mean(),
        "avg_gross_exposure": monthly_turnover["gross_exposure"].mean(),
        "avg_net_exposure": monthly_turnover["net_exposure"].mean(),
        "avg_n_stocks": monthly_turnover["n_stocks"].mean(),
        "avg_missing_prev_ret": monthly_turnover["n_missing_prev_ret"].dropna().mean(),
    }])

    summary_fp = TURNOVER_OUTPUT_BASE / "turnover_summary.csv"
    summary.to_csv(summary_fp, index=False)

    print("=" * 100)
    print("[INFO] IPCA drift-adjusted turnover finished successfully")
    print(f"[INFO] Saved monthly turnover to: {output_fp}")
    print(f"[INFO] Saved turnover summary to : {summary_fp}")
    print(f"[INFO] holdings rows          : {len(holdings):,}")
    print(f"[INFO] stock_ret_df rows      : {len(stock_ret_df):,}")
    print(f"[INFO] monthly_turnover shape : {monthly_turnover.shape}")

    avg_turnover = monthly_turnover["turnover"].dropna().mean()
    avg_simple_turnover = monthly_turnover["simple_turnover"].dropna().mean()

    print(f"[INFO] average drift-adjusted monthly turnover: {avg_turnover:.6f}")
    print(f"[INFO] average simple monthly turnover        : {avg_simple_turnover:.6f}")


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