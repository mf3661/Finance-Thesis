from __future__ import annotations

import pandas as pd


def add_capxint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add capital expenditure intensity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "capx",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for capxint: {missing_cols}")

    output_df["capxint"] = output_df["capx"] / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df