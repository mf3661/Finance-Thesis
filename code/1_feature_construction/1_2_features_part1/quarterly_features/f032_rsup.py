from __future__ import annotations

import pandas as pd


def add_rsup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly revenue surprise variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "saleq",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly rsup: {missing_cols}")

    output_df["saleq_l4"] = output_df.groupby("permno")["saleq"].shift(4)

    output_df["rsup"] = (
        output_df["saleq"] - output_df["saleq_l4"]
    ) / output_df["me"]

    return output_df