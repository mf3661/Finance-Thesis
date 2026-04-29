from __future__ import annotations

import pandas as pd


def add_sp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly sales-to-price variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "saleq4",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly sp: {missing_cols}")

    output_df["sp"] = output_df["saleq4"] / output_df["me"]

    return output_df