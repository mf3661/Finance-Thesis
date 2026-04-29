from __future__ import annotations

import numpy as np
import pandas as pd


def _add_industry_adjusted_mean(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    industry_col: str = "ffi49",
    date_col: str = "datadate",
) -> pd.DataFrame:
    """
    Add industry-adjusted variable using within-date industry mean.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        date_col,
        industry_col,
        value_col,
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for {output_col}: {missing_cols}"
        )

    industry_mean_col = f"{value_col}_ind"

    industry_temp_df = (
        output_df.groupby([date_col, industry_col], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: industry_mean_col})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=[date_col, industry_col],
    ).reset_index(drop=True)

    output_df[output_col] = output_df[value_col] - output_df[industry_mean_col]

    return output_df


def recompute_annual_monthly_me_features(
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recompute annual variables that depend on monthly market equity.
    """

    output_df: pd.DataFrame = annual_df.copy()

    required_cols: list[str] = [
        "datadate",
        "date",
        "ffi49",
        "me",
        "be",
        "dp",
        "ib",
        "sale",
        "sale_l1",
        "lt",
        "xrd",
        "dvt",
        "dltt",
        "at",
        "che",
        "mom12m",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for annual monthly-ME recomputation: {missing_cols}"
        )

    output_df["bm"] = output_df["be"] / output_df["me"]

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="bm",
        output_col="bm_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="me",
        output_col="me_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    condition_list = [
        output_df["dp"].isna(),
        output_df["ib"].isna(),
    ]

    choice_list = [
        output_df["ib"] / output_df["me"],
        np.nan,
    ]

    output_df["cfp"] = np.select(
        condition_list,
        choice_list,
        default=(output_df["ib"] + output_df["dp"]) / output_df["me"],
    )

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="cfp",
        output_col="cfp_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    output_df["ep"] = output_df["ib"] / output_df["me"]
    output_df["rsup"] = (output_df["sale"] - output_df["sale_l1"]) / output_df["me"]
    output_df["lev"] = output_df["lt"] / output_df["me"]
    output_df["sp"] = output_df["sale"] / output_df["me"]
    output_df["rdm"] = output_df["xrd"] / output_df["me"]

    output_df["dy"] = output_df["dvt"] / output_df["me"]

    output_df["cashpr"] = (
        output_df["me"]
        + output_df["dltt"]
        - output_df["at"]
    ) / output_df["che"]

    industry_temp_df = (
        output_df.groupby(["date", "ffi49"], as_index=False)["mom12m"]
        .mean()
        .rename(columns={"mom12m": "indmom"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["date", "ffi49"],
    ).reset_index(drop=True)

    return output_df


def recompute_quarterly_monthly_me_features(
    quarterly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recompute quarterly variables that depend on monthly market equity.
    """

    output_df: pd.DataFrame = quarterly_df.copy()

    required_cols: list[str] = [
        "datadate",
        "date",
        "ffi49",
        "me",
        "beq",
        "dpq",
        "ibq4",
        "dpq4",
        "ltq",
        "xrdq4",
        "saleq4",
        "dlttq",
        "atq",
        "cheq",
        "mom12m",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly monthly-ME recomputation: {missing_cols}"
        )

    output_df["bm"] = output_df["beq"] / output_df["me"]

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="bm",
        output_col="bm_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="me",
        output_col="me_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    output_df["cfp"] = np.where(
        output_df["dpq"].isna(),
        output_df["ibq4"] / output_df["me"],
        (output_df["ibq4"] + output_df["dpq4"]) / output_df["me"],
    )

    output_df = _add_industry_adjusted_mean(
        df=output_df,
        value_col="cfp",
        output_col="cfp_ia",
        industry_col="ffi49",
        date_col="datadate",
    )

    output_df["ep"] = output_df["ibq4"] / output_df["me"]
    output_df["lev"] = output_df["ltq"] / output_df["me"]
    output_df["rdm"] = output_df["xrdq4"] / output_df["me"]
    output_df["sp"] = output_df["saleq4"] / output_df["me"]

    output_df["cashpr"] = (
        output_df["me"]
        + output_df["dlttq"]
        - output_df["atq"]
    ) / output_df["cheq"]

    industry_temp_df = (
        output_df.groupby(["date", "ffi49"], as_index=False)["mom12m"]
        .mean()
        .rename(columns={"mom12m": "indmom"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["date", "ffi49"],
    ).reset_index(drop=True)

    return output_df