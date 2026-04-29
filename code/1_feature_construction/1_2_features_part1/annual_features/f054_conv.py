from __future__ import annotations

import pandas as pd


def add_conv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add convertible debt ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dc",
        "dltt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for conv: {missing_cols}")

    output_df["conv"] = output_df["dc"] / output_df["dltt"]

    return output_df