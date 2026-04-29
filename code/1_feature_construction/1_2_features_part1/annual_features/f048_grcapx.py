from __future__ import annotations

import pandas as pd


def add_grcapx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two-year capital expenditure growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "capx",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for grcapx: {missing_cols}")

    output_df["capx_l2"] = output_df.groupby("permno")["capx"].shift(2)

    output_df["grcapx"] = (
        output_df["capx"] - output_df["capx_l2"]
    ) / output_df["capx_l2"]

    return output_df