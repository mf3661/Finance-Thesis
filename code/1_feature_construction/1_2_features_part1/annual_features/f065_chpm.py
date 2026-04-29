from __future__ import annotations

import pandas as pd


def add_chpm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in profit margin variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ib",
        "sale",
        "sale_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chpm: {missing_cols}")

    output_df["ib_l1"] = output_df.groupby("permno")["ib"].shift(1)

    output_df["chpm"] = (
        output_df["ib"] / output_df["sale"]
    ) - (
        output_df["ib_l1"] / output_df["sale_l1"]
    )

    return output_df