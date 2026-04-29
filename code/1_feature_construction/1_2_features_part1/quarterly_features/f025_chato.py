from __future__ import annotations

import pandas as pd


def add_chato(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly change in asset turnover variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "saleq4",
        "saleq4_l4",
        "atq",
        "atq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly chato: {missing_cols}")

    output_df["atq_l8"] = output_df.groupby("permno")["atq"].shift(8)

    output_df["chato"] = (
        output_df["saleq4"] / ((output_df["atq"] + output_df["atq_l4"]) / 2)
    ) - (
        output_df["saleq4_l4"]
        / ((output_df["atq_l4"] + output_df["atq_l8"]) / 2)
    )

    return output_df