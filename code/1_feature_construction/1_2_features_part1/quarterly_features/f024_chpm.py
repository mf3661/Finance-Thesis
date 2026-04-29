from __future__ import annotations

import pandas as pd


def add_chpm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly change in profit margin variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ibq4",
        "saleq4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly chpm: {missing_cols}")

    output_df["ibq4_l1"] = output_df.groupby("permno")["ibq4"].shift(1)
    output_df["saleq4_l1"] = output_df.groupby("permno")["saleq4"].shift(1)

    output_df["chpm"] = (
        output_df["ibq4"] / output_df["saleq4"]
    ) - (
        output_df["ibq4_l1"] / output_df["saleq4_l1"]
    )

    return output_df