from __future__ import annotations

import pandas as pd


def add_salerec(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales-to-receivables variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sale",
        "rect",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for salerec: {missing_cols}")

    output_df["salerec"] = output_df["sale"] / output_df["rect"]

    return output_df