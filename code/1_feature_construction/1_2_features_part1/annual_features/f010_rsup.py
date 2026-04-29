from __future__ import annotations

import pandas as pd


def add_rsup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add revenue surprise variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "sale",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rsup: {missing_cols}")

    output_df["sale_l1"] = output_df.groupby("permno")["sale"].shift(1)

    output_df["rsup"] = (
        output_df["sale"] - output_df["sale_l1"]
    ) / output_df["me"]

    return output_df