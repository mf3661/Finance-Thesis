from __future__ import annotations

import pandas as pd


def add_currat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add current ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "act",
        "lct",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for currat: {missing_cols}")

    output_df["currat"] = output_df["act"] / output_df["lct"]

    return output_df