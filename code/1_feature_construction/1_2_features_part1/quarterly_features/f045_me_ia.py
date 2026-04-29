from __future__ import annotations

import pandas as pd


def add_me_ia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly industry-adjusted market equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly me_ia: {missing_cols}"
        )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["me"]
        .mean()
        .rename(columns={"me": "me_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["me_ia"] = output_df["me"] - output_df["me_ind"]

    return output_df