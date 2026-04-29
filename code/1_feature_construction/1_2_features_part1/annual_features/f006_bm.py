from __future__ import annotations

import pandas as pd


def add_bm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add book-to-market variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "be",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for bm: {missing_cols}")

    output_df["bm"] = output_df["be"] / output_df["me"]

    return output_df