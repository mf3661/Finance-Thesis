from __future__ import annotations

import pandas as pd


def add_pm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add profit margin variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "oiadp",
        "sale",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pm: {missing_cols}")

    output_df["pm"] = output_df["oiadp"] / output_df["sale"]

    return output_df