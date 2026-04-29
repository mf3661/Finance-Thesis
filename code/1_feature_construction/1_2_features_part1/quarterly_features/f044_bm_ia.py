from __future__ import annotations

import pandas as pd


def add_bm_ia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly industry-adjusted book-to-market variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "bm",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly bm_ia: {missing_cols}"
        )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["bm"]
        .mean()
        .rename(columns={"bm": "bm_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["bm_ia"] = output_df["bm"] - output_df["bm_ind"]

    return output_df