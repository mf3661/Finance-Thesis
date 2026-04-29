from __future__ import annotations

import pandas as pd


def add_chtx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly change in tax expense variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "txtq",
        "atq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly chtx: {missing_cols}")

    output_df["txtq_l4"] = output_df.groupby("permno")["txtq"].shift(4)
    output_df["atq_l4"] = output_df.groupby("permno")["atq"].shift(4)

    output_df["chtx"] = (
        output_df["txtq"] - output_df["txtq_l4"]
    ) / output_df["atq_l4"]

    return output_df