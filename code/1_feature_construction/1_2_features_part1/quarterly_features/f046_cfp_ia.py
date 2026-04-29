from __future__ import annotations

import pandas as pd


def add_cfp_ia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly industry-adjusted cash-flow-to-price variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "cfp",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly cfp_ia: {missing_cols}"
        )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["cfp"]
        .mean()
        .rename(columns={"cfp": "cfp_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["cfp_ia"] = output_df["cfp"] - output_df["cfp_ind"]

    return output_df