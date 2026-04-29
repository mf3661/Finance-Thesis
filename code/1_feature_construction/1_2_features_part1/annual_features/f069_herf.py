from __future__ import annotations

import pandas as pd


def add_herf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add industry sales concentration variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "sale",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for herf: {missing_cols}")

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["sale"]
        .sum()
        .rename(columns={"sale": "indsale"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["herf"] = (
        output_df["sale"] / output_df["indsale"]
    ) * (
        output_df["sale"] / output_df["indsale"]
    )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["herf"]
        .sum()
    )

    output_df = output_df.drop(["herf"], axis=1)

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    return output_df