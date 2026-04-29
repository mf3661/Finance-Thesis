from __future__ import annotations

import pandas as pd


def add_rdm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R&D expense-to-market equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "xrd",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rdm: {missing_cols}")

    output_df["rdm"] = output_df["xrd"] / output_df["me"]

    return output_df