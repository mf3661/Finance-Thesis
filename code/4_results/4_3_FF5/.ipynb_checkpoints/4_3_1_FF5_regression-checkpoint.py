from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================================================
# Paths
# =========================================================
FF5_PATH = Path("../../../data/common/FF5.csv")

PNL_BASE = Path("../../../data/4_results/4_1_pnl")

OUTPUT_DIR = Path("../../../data/4_results/4_3_FF5/4_3_1_FF5_regression/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "date"
RETURN_COL_CANDIDATES = ["ensemble_ret", "strategy_ret", "ls_ret"]


# =========================================================
# Regression settings
# =========================================================
STRATEGY_RET_IS_EXCESS = True

COV_TYPE = "nonrobust"
HAC_LAGS = 6


# =========================================================
# Strategy input files
# =========================================================
STRATEGIES = {
    "AP forest I": [
        PNL_BASE / "4_1_1_ap_forestI" / "output" / "ensemble_monthly_returns.csv",
    ],
    "AP forest EW": [
        PNL_BASE / "4_1_2_ap_forest_ew" / "output" / "ensemble_monthly_returns.csv",
    ],
    "AP forest Impute": [
        PNL_BASE / "4_1_3_ap_forest_impute" / "output" / "ensemble_monthly_returns.csv",
    ],
    "AP tree": [
        PNL_BASE / "4_1_4_ap_forest_benchmark" / "output" / "ensemble_monthly_returns.csv",
    ],
    "AP forest II": [
        PNL_BASE / "4_1_5_ap_forestII" / "output" / "ensemble_monthly_returns.csv",
    ],
    "IPCA": [
        PNL_BASE / "4_1_6_IPCA" / "output" / "ensemble_monthly_returns.csv",
    ],
    "Ridge": [
        PNL_BASE / "4_1_7_ridge" / "output" / "ensemble_monthly_returns.csv",
    ],
}


# =========================================================
# Helpers
# =========================================================
def find_existing_file(candidates: list[Path], strategy_name: str) -> Path | None:
    for fp in candidates:
        if fp.exists():
            return fp

    print(f"[WARNING] No input file found for {strategy_name}.")
    for fp in candidates:
        print(f"          checked: {fp}")

    return None


def find_return_column(df: pd.DataFrame, fp: Path) -> str:
    for col in RETURN_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        f"No valid return column found in {fp}. "
        f"Expected one of {RETURN_COL_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def add_significance_stars(pval: float) -> str:
    if not np.isfinite(pval):
        return ""

    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    if pval < 0.10:
        return "+"
    return ""


def read_ff5_factors(ff5_path: Path) -> pd.DataFrame:
    """
    Read Fama-French 5 factors from a non-standard CSV.

    Expected original FF5 monthly format:
        descriptive lines
        ,Mkt-RF,SMB,HML,RMW,CMA,RF
        196307, -0.39, -0.44, -0.89, 0.68, -1.23, 0.27
        ...
        Annual Factors: January-December
        ...

    Output columns:
        ym, Mkt_RF, SMB, HML, RMW, CMA, RF

    All factor columns are divided by 100.
    """
    if not ff5_path.exists():
        raise FileNotFoundError(f"Missing FF5 file: {ff5_path}")

    with open(ff5_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if (
            "Mkt-RF" in line
            and "SMB" in line
            and "HML" in line
            and "RMW" in line
            and "CMA" in line
            and "RF" in line
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find FF5 header line in {ff5_path}")

    raw = pd.read_csv(ff5_path, skiprows=header_idx)

    first_col = raw.columns[0]
    raw = raw.rename(columns={first_col: "yyyymm"})
    raw["yyyymm"] = raw["yyyymm"].astype(str).str.strip()

    raw = raw[raw["yyyymm"].str.fullmatch(r"\d{6}", na=False)].copy()

    rename_map = {
        "Mkt-RF": "Mkt_RF",
        "Mkt_RF": "Mkt_RF",
        "SMB": "SMB",
        "HML": "HML",
        "RMW": "RMW",
        "CMA": "CMA",
        "RF": "RF",
    }
    raw = raw.rename(columns=rename_map)

    required_cols = ["yyyymm", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in parsed FF5 file. "
            f"Available columns: {list(raw.columns)}"
        )

    for col in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce") / 100.0

    raw = raw.dropna(subset=["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]).copy()
    raw["ym"] = pd.PeriodIndex(raw["yyyymm"], freq="M")

    out = raw[["ym", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()
    out = out.drop_duplicates(subset=["ym"])
    out = out.sort_values("ym").reset_index(drop=True)

    print("=" * 100)
    print("[INFO] Loaded FF5 factors")
    print(f"[INFO] FF5 path       : {ff5_path}")
    print(f"[INFO] rows           : {len(out)}")
    print(f"[INFO] date range     : {out['ym'].min()} -> {out['ym'].max()}")
    print("[INFO] factors scaled : divided by 100")

    return out


def load_strategy_returns(strategy_name: str, candidates: list[Path]) -> pd.DataFrame | None:
    fp = find_existing_file(candidates, strategy_name)

    if fp is None:
        return None

    df = pd.read_csv(fp)

    if DATE_COL not in df.columns:
        raise ValueError(f"Missing date column in {fp}. Available columns: {list(df.columns)}")

    ret_col = find_return_column(df, fp)

    out = df[[DATE_COL, ret_col]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[DATE_COL, ret_col]).copy()

    out = out.rename(columns={ret_col: "strategy_ret"})
    out["ym"] = out[DATE_COL].dt.to_period("M")
    out["strategy"] = strategy_name

    out = out.sort_values(DATE_COL).reset_index(drop=True)

    print("=" * 100)
    print(f"[INFO] Loaded strategy returns: {strategy_name}")
    print(f"[INFO] file       : {fp}")
    print(f"[INFO] return col : {ret_col}")
    print(f"[INFO] rows       : {len(out)}")
    print(f"[INFO] date range : {out[DATE_COL].min()} -> {out[DATE_COL].max()}")

    return out


# =========================================================
# Regression function
# =========================================================
def run_ff5_regression(
    strategy_ret_df: pd.DataFrame,
    ff5_df: pd.DataFrame,
    strategy_name: str,
    strategy_ret_is_excess: bool = True,
    cov_type: str = "nonrobust",
    hac_lags: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run FF5 regression for one strategy.

    If strategy_ret_is_excess is True:
        y_t = strategy_ret_t

    If strategy_ret_is_excess is False:
        y_t = strategy_ret_t - RF_t

    Regression:
        y_t = alpha
            + b_mkt * Mkt_RF_t
            + b_smb * SMB_t
            + b_hml * HML_t
            + b_rmw * RMW_t
            + b_cma * CMA_t
            + eps_t

    Returns:
        summary_df, regression_data
    """
    required_ret_cols = ["ym", "strategy_ret"]
    missing_ret = [c for c in required_ret_cols if c not in strategy_ret_df.columns]
    if missing_ret:
        raise ValueError(f"Missing strategy return columns: {missing_ret}")

    required_ff_cols = ["ym", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing_ff = [c for c in required_ff_cols if c not in ff5_df.columns]
    if missing_ff:
        raise ValueError(f"Missing FF5 columns: {missing_ff}")

    reg_df = strategy_ret_df[[DATE_COL, "ym", "strategy_ret"]].copy()
    reg_df = reg_df.merge(ff5_df, on="ym", how="inner")

    reg_df = reg_df.replace([np.inf, -np.inf], np.nan)
    reg_df = reg_df.dropna(
        subset=["strategy_ret", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    ).copy()
    reg_df = reg_df.sort_values(DATE_COL).reset_index(drop=True)

    if reg_df.empty:
        raise ValueError(f"No overlapping monthly data for {strategy_name}")

    if strategy_ret_is_excess:
        reg_df["y"] = reg_df["strategy_ret"]
    else:
        reg_df["y"] = reg_df["strategy_ret"] - reg_df["RF"]

    X = reg_df[["Mkt_RF", "SMB", "HML", "RMW", "CMA"]].copy()
    X = sm.add_constant(X)
    y = reg_df["y"].astype(float)

    model = sm.OLS(y, X, missing="drop")

    if cov_type.upper() == "HAC":
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        se_label = f"HAC({hac_lags})"
    else:
        res = model.fit()
        se_label = "OLS"

    alpha = float(res.params.get("const", np.nan))
    alpha_t = float(res.tvalues.get("const", np.nan))
    alpha_p = float(res.pvalues.get("const", np.nan))

    beta_mkt = float(res.params.get("Mkt_RF", np.nan))
    beta_smb = float(res.params.get("SMB", np.nan))
    beta_hml = float(res.params.get("HML", np.nan))
    beta_rmw = float(res.params.get("RMW", np.nan))
    beta_cma = float(res.params.get("CMA", np.nan))

    t_mkt = float(res.tvalues.get("Mkt_RF", np.nan))
    t_smb = float(res.tvalues.get("SMB", np.nan))
    t_hml = float(res.tvalues.get("HML", np.nan))
    t_rmw = float(res.tvalues.get("RMW", np.nan))
    t_cma = float(res.tvalues.get("CMA", np.nan))

    mean_y = float(reg_df["y"].mean())
    std_y = float(reg_df["y"].std(ddof=1))
    monthly_sr = mean_y / std_y if std_y > 0 else np.nan
    annualized_sr = np.sqrt(12.0) * monthly_sr if np.isfinite(monthly_sr) else np.nan

    summary = pd.DataFrame([{
        "strategy": strategy_name,
        "start_month": str(reg_df["ym"].min()),
        "end_month": str(reg_df["ym"].max()),
        "n_obs": int(res.nobs),
        "ret_is_excess": strategy_ret_is_excess,
        "se_type": se_label,
        "alpha_monthly": alpha,
        "alpha_monthly_pct": 100.0 * alpha,
        "alpha_annualized": 12.0 * alpha,
        "alpha_annualized_pct": 100.0 * 12.0 * alpha,
        "alpha_t": alpha_t,
        "alpha_p": alpha_p,
        "alpha_stars": add_significance_stars(alpha_p),
        "beta_mkt": beta_mkt,
        "beta_smb": beta_smb,
        "beta_hml": beta_hml,
        "beta_rmw": beta_rmw,
        "beta_cma": beta_cma,
        "t_mkt": t_mkt,
        "t_smb": t_smb,
        "t_hml": t_hml,
        "t_rmw": t_rmw,
        "t_cma": t_cma,
        "r_squared": float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "mean_monthly_y": mean_y,
        "std_monthly_y": std_y,
        "monthly_sr": monthly_sr,
        "annualized_sr": annualized_sr,
    }])

    reg_df["fitted"] = res.fittedvalues
    reg_df["resid"] = res.resid
    reg_df["strategy"] = strategy_name

    return summary, reg_df


# =========================================================
# Main
# =========================================================
def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ff5_df = read_ff5_factors(FF5_PATH)

    selected_strategies = STRATEGIES

    if args.strategies.lower() != "all":
        names = [x.strip() for x in args.strategies.split(",") if x.strip()]
        selected_strategies = {k: v for k, v in STRATEGIES.items() if k in names}

        if len(selected_strategies) == 0:
            raise ValueError(f"No requested strategies found. Requested: {names}")

    all_summary = []
    all_reg_data = []

    for strategy_name, candidates in selected_strategies.items():
        strategy_df = load_strategy_returns(strategy_name, candidates)

        if strategy_df is None:
            continue

        summary, reg_data = run_ff5_regression(
            strategy_ret_df=strategy_df,
            ff5_df=ff5_df,
            strategy_name=strategy_name,
            strategy_ret_is_excess=args.strategy_ret_is_excess,
            cov_type=args.cov_type,
            hac_lags=args.hac_lags,
        )

        all_summary.append(summary)
        all_reg_data.append(reg_data)

    if len(all_summary) == 0:
        raise ValueError("No FF5 regressions were run. Please check input paths.")

    summary_df = pd.concat(all_summary, axis=0, ignore_index=True)
    reg_data_df = pd.concat(all_reg_data, axis=0, ignore_index=True)

    summary_path = OUTPUT_DIR / "FF5_regression_summary.csv"
    reg_data_path = OUTPUT_DIR / "FF5_regression_data.csv"

    summary_df.to_csv(summary_path, index=False)
    reg_data_df.to_csv(reg_data_path, index=False)

    print("=" * 100)
    print("[INFO] FF5 regression finished")
    print(f"[INFO] Saved summary data to   : {summary_path}")
    print(f"[INFO] Saved regression data to: {reg_data_path}")

    display_cols = [
        "strategy",
        "n_obs",
        "alpha_monthly_pct",
        "alpha_t",
        "alpha_stars",
        "beta_mkt",
        "beta_smb",
        "beta_hml",
        "beta_rmw",
        "beta_cma",
        "r_squared",
        "annualized_sr",
    ]

    print("=" * 100)
    print("[INFO] Summary")
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help='Use "all" or comma-separated names, e.g. "AP forest I,IPCA,Ridge".',
    )

    parser.add_argument(
        "--strategy_ret_is_excess",
        action=argparse.BooleanOptionalAction,
        default=STRATEGY_RET_IS_EXCESS,
        help="If true, regress strategy_ret directly. If false, regress strategy_ret - RF.",
    )

    parser.add_argument(
        "--cov_type",
        type=str,
        default=COV_TYPE,
        choices=["nonrobust", "HAC"],
        help='Use "nonrobust" for plain OLS SE or "HAC" for Newey-West SE.',
    )

    parser.add_argument(
        "--hac_lags",
        type=int,
        default=HAC_LAGS,
        help="Number of lags for HAC/Newey-West standard errors.",
    )

    args = parser.parse_args()
    main(args)