from __future__ import annotations

import pandas as pd


def add_cash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cash-to-assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "che",
        "at",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for cash: {missing_cols}")

    output_df["cash"] = output_df["che"] / output_df["at"]

    return output_df