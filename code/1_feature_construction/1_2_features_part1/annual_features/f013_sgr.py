from __future__ import annotations

import pandas as pd


def add_sgr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sale",
        "sale_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for sgr: {missing_cols}")

    output_df["sgr"] = (
        output_df["sale"] / output_df["sale_l1"]
    ) - 1

    return output_df