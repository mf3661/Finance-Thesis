from __future__ import annotations

import pandas as pd


def add_chpmia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add industry-adjusted change in profit margin variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "chpm",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chpmia: {missing_cols}")

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["chpm"]
        .mean()
        .rename(columns={"chpm": "chpm_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["chpmia"] = output_df["chpm"] - output_df["chpm_ind"]

    return output_df