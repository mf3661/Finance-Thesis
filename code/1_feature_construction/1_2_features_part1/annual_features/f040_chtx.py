from __future__ import annotations

import pandas as pd


def add_chtx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in tax expense variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "txt",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chtx: {missing_cols}")

    output_df["txt_l1"] = output_df.groupby("permno")["txt"].shift(1)

    output_df["chtx"] = (
        output_df["txt"] - output_df["txt_l1"]
    ) / output_df["at_l1"]

    return output_df