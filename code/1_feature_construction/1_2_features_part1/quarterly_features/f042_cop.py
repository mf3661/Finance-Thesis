from __future__ import annotations

import numpy as np
import pandas as pd


def add_cop(
    quarterly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add quarterly cash-based operating profitability variable.
    """

    output_df: pd.DataFrame = quarterly_df.copy()
    annual_input_df: pd.DataFrame = annual_df.copy()

    required_quarterly_cols: list[str] = [
        "permno",
        "jdate",
        "opa",
        "rectq",
        "rectq_l4",
        "invtq",
        "invtq_l4",
        "drcq",
        "drltq",
        "apq",
        "apq_l4",
        "xaccq",
        "atq",
    ]

    missing_quarterly_cols: list[str] = [
        col for col in required_quarterly_cols if col not in output_df.columns
    ]

    if missing_quarterly_cols:
        raise KeyError(
            f"Missing required quarterly columns for cop: {missing_quarterly_cols}"
        )

    required_annual_cols: list[str] = [
        "permno",
        "jdate",
        "xpp",
    ]

    missing_annual_cols: list[str] = [
        col for col in required_annual_cols if col not in annual_input_df.columns
    ]

    if missing_annual_cols:
        raise KeyError(
            f"Missing required annual columns for quarterly cop: {missing_annual_cols}"
        )

    output_df["jdate"] = pd.to_datetime(output_df["jdate"])
    annual_input_df["jdate"] = pd.to_datetime(annual_input_df["jdate"])

    output_df["year"] = output_df["jdate"].dt.year
    annual_input_df["year"] = annual_input_df["jdate"].dt.year
    annual_input_df["jdate_a"] = annual_input_df["jdate"].copy()

    annual_input_df = annual_input_df.sort_values(
        by=["permno", "jdate"]
    ).reset_index(drop=True)

    annual_input_df["xpp_l1"] = (
        annual_input_df.groupby("permno")["xpp"].shift(1)
    )

    annual_xpp_df = annual_input_df[
        [
            "permno",
            "year",
            "jdate_a",
            "xpp",
            "xpp_l1",
        ]
    ].copy()

    output_df = pd.merge(
        output_df,
        annual_xpp_df,
        how="left",
        on=["permno", "year"],
    ).reset_index(drop=True)

    output_df["xpp"] = np.where(
        output_df["jdate"] > output_df["jdate_a"],
        output_df["xpp"],
        output_df["xpp_l1"],
    )

    output_df["xpp_l4"] = output_df.groupby("permno")["xpp"].shift(4)
    output_df["xaccq_l4"] = output_df.groupby("permno")["xaccq"].shift(4)

    output_df["cop"] = (
        output_df["opa"]
        - (output_df["rectq"] - output_df["rectq_l4"])
        - (output_df["invtq"] - output_df["invtq_l4"])
        - (output_df["xpp"] - output_df["xpp_l4"])
        + (output_df["drcq"] + output_df["drltq"])
        + (output_df["apq"] - output_df["apq_l4"])
        + (output_df["xaccq"] - output_df["xaccq_l4"])
    ) / output_df["atq"]

    return output_df