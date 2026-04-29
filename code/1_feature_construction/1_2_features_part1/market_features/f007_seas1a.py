from __future__ import annotations

import pandas as pd


def add_seas1a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add annual seasonality variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ret",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for seas1a: {missing_cols}")

    output_df["seas1a"] = output_df.groupby("permno")["ret"].shift(11)

    return output_df