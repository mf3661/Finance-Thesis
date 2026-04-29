from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================
BASE_DIR = Path("../../../data/4_results/4_1_pnl")

OUTPUT_DIR = BASE_DIR / "4_1_11_figure9" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "date"
RETURN_COL_CANDIDATES = ["ensemble_ret", "strategy_ret", "ls_ret"]


# =========================================================
# Sample input mapping
# =========================================================
# This is SAMPLE CODE only.
# You must replace the placeholder paths below with the actual
# ensemble_monthly_returns.csv produced by your bash runs under
# different N and B settings.
#
# Example idea:
#   N=10, B=20
#   N=10, B=10
#   N=5,  B=20
#
# Each configuration should have its own pnl output file.
STRATEGIES = {
    "N=10, B=20": Path("REPLACE_ME_WITH_ACTUAL_PATH_1/ensemble_monthly_returns.csv"),
    "N=10, B=10": Path("REPLACE_ME_WITH_ACTUAL_PATH_2/ensemble_monthly_returns.csv"),
    "N=5, B=20": Path("REPLACE_ME_WITH_ACTUAL_PATH_3/ensemble_monthly_returns.csv"),
}


# =========================================================
# Helpers
# =========================================================
def validate_strategy_paths(strategy_files: dict[str, Path]) -> None:
    for label, fp in strategy_files.items():
        if "REPLACE_ME" in str(fp):
            raise ValueError(
                f"Placeholder path detected for {label}: {fp}\n"
                f"This script is sample code only. Replace STRATEGIES with actual paths."
            )


def find_return_column(df: pd.DataFrame, fp: Path) -> str:
    for col in RETURN_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        f"No valid return column found in {fp}. "
        f"Expected one of: {RETURN_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def load_one_strategy(label: str, fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"Missing input file for {label}: {fp}")

    df = pd.read_csv(fp)

    if DATE_COL not in df.columns:
        raise ValueError(
            f"Missing date column in {fp}. Available columns: {list(df.columns)}"
        )

    ret_col = find_return_column(df, fp)

    out = df[[DATE_COL, ret_col]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[DATE_COL, ret_col]).copy()
    out = out.sort_values(DATE_COL).reset_index(drop=True)
    out = out.rename(columns={ret_col: label})

    print(f"[INFO] Loaded {label}")
    print(f"       file       : {fp}")
    print(f"       return col : {ret_col}")
    print(f"       rows       : {len(out)}")
    print(f"       date range : {out[DATE_COL].min()} -> {out[DATE_COL].max()}")

    return out


def merge_strategy_returns(strategy_files: dict[str, Path]) -> pd.DataFrame:
    merged = None

    for label, fp in strategy_files.items():
        one = load_one_strategy(label, fp)

        if merged is None:
            merged = one
        else:
            merged = merged.merge(one, on=DATE_COL, how="inner")

    if merged is None:
        raise ValueError("No strategy returns loaded.")

    merged = merged.sort_values(DATE_COL).reset_index(drop=True)

    if merged.empty:
        raise ValueError("Merged return panel is empty after inner join.")

    strategy_cols = [c for c in merged.columns if c != DATE_COL]
    missing_count = int(merged[strategy_cols].isna().sum().sum())

    if missing_count > 0:
        raise ValueError(f"Merged return panel contains {missing_count} missing values.")

    print("=" * 100)
    print("[INFO] Merged return panel")
    print(f"       rows       : {len(merged)}")
    print(f"       date range : {merged[DATE_COL].min()} -> {merged[DATE_COL].max()}")
    print(f"       columns    : {strategy_cols}")

    return merged


def compute_cumulative_log_returns(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    out = monthly_returns.copy()
    strategy_cols = [c for c in out.columns if c != DATE_COL]

    for col in strategy_cols:
        r = pd.to_numeric(out[col], errors="coerce")

        if (r <= -1.0).any():
            bad_dates = out.loc[r <= -1.0, DATE_COL].head(5).tolist()
            raise ValueError(
                f"{col} has returns <= -100%, so log1p is invalid. "
                f"Example bad dates: {bad_dates}"
            )

        out[col] = np.log1p(r).cumsum()

    return out


def compute_summary_stats(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    strategy_cols = [c for c in monthly_returns.columns if c != DATE_COL]

    rows = []

    for col in strategy_cols:
        r = pd.to_numeric(monthly_returns[col], errors="coerce").dropna()

        mean_monthly = float(r.mean())
        std_monthly = float(r.std(ddof=1))
        monthly_sr = mean_monthly / std_monthly if std_monthly > 0 else np.nan
        annualized_sr = np.sqrt(12.0) * monthly_sr if np.isfinite(monthly_sr) else np.nan

        rows.append({
            "strategy": col,
            "start_date": monthly_returns[DATE_COL].min(),
            "end_date": monthly_returns[DATE_COL].max(),
            "n_months": int(r.shape[0]),
            "mean_monthly_ret": mean_monthly,
            "std_monthly_ret": std_monthly,
            "monthly_sr": monthly_sr,
            "annualized_sr": annualized_sr,
            "annualized_mean_ret": 12.0 * mean_monthly,
            "cumulative_log_ret_final": float(np.log1p(r).sum()),
        })

    return pd.DataFrame(rows)


def plot_figure(cum_log_returns: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))

    strategy_cols = [c for c in cum_log_returns.columns if c != DATE_COL]

    for col in strategy_cols:
        plt.plot(
            cum_log_returns[DATE_COL],
            cum_log_returns[col],
            label=col,
            linewidth=1.5,
        )

    plt.title("Out-of-Sample Performance")
    plt.xlabel("date")
    plt.ylabel("cumulative log return")
    plt.legend(loc="upper left")
    plt.tight_layout()

    fig_png = OUTPUT_DIR / "figure9_vary_ensemble_size.png"
    fig_pdf = OUTPUT_DIR / "figure9_vary_ensemble_size.pdf"

    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Saved figure PNG: {fig_png}")
    print(f"[INFO] Saved figure PDF: {fig_pdf}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    validate_strategy_paths(STRATEGIES)

    monthly_returns = merge_strategy_returns(STRATEGIES)
    cum_log_returns = compute_cumulative_log_returns(monthly_returns)
    summary = compute_summary_stats(monthly_returns)

    monthly_path = OUTPUT_DIR / "figure9_monthly_returns_merged.csv"
    cumlog_path = OUTPUT_DIR / "figure9_cumulative_log_returns.csv"
    summary_path = OUTPUT_DIR / "figure9_summary.csv"

    monthly_returns.to_csv(monthly_path, index=False)
    cum_log_returns.to_csv(cumlog_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("=" * 100)
    print("[INFO] Saved outputs")
    print(f"[INFO] Monthly returns       : {monthly_path}")
    print(f"[INFO] Cumulative log return: {cumlog_path}")
    print(f"[INFO] Summary               : {summary_path}")

    print("=" * 100)
    print("[INFO] Summary")
    print(summary.to_string(index=False))

    plot_figure(cum_log_returns)


if __name__ == "__main__":
    main()