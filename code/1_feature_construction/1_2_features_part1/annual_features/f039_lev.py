from __future__ import annotations

import pandas as pd


def add_lev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leverage variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "lt",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for lev: {missing_cols}")

    output_df["lev"] = output_df["lt"] / output_df["me"]

    return output_df