from __future__ import annotations

import pandas as pd


def add_secured(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add secured debt ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dm",
        "dltt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for secured: {missing_cols}")

    output_df["secured"] = output_df["dm"] / output_df["dltt"]

    return output_df