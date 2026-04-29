from __future__ import annotations

import pandas as pd


def add_pm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly profit margin variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "oiadpq",
        "saleq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly pm: {missing_cols}")

    output_df["pm"] = output_df["oiadpq"] / output_df["saleq"]

    return output_df