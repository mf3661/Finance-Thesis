from __future__ import annotations

import pandas as pd


def add_quick(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quick ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "act",
        "invt",
        "lct",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quick: {missing_cols}")

    output_df["quick"] = (
        output_df["act"] - output_df["invt"]
    ) / output_df["lct"]

    return output_df