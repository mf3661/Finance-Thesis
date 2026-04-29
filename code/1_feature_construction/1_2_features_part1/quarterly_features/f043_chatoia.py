from __future__ import annotations

import pandas as pd


def add_chatoia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly industry-adjusted change in asset turnover variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "chato",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly chatoia: {missing_cols}"
        )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["chato"]
        .mean()
        .rename(columns={"chato": "chato_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["chatoia"] = output_df["chato"] - output_df["chato_ind"]

    return output_df