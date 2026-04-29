from __future__ import annotations

import pandas as pd


def add_gma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gross profitability variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "revt",
        "cogs",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for gma: {missing_cols}")

    output_df["gma"] = (
        output_df["revt"] - output_df["cogs"]
    ) / output_df["at_l1"]

    return output_df