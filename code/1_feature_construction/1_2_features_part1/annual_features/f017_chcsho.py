from __future__ import annotations

import pandas as pd


def add_chcsho(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in shares outstanding variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "csho",
        "csho_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chcsho: {missing_cols}")

    output_df["chcsho"] = (
        output_df["csho"] / output_df["csho_l1"]
    ) - 1

    return output_df