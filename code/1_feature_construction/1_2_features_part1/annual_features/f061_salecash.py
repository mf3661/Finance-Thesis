from __future__ import annotations

import pandas as pd


def add_salecash(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales-to-cash variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sale",
        "che",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for salecash: {missing_cols}")

    output_df["salecash"] = output_df["sale"] / output_df["che"]

    return output_df