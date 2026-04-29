from __future__ import annotations

import pandas as pd


def add_chempia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add industry-adjusted employee growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "datadate",
        "ffi49",
        "hire",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chempia: {missing_cols}")

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["hire"]
        .mean()
        .rename(columns={"hire": "hire_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["chempia"] = output_df["hire"] - output_df["hire_ind"]

    return output_df