from __future__ import annotations

import pandas as pd


def add_adm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advertising expense-to-market equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "xad",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for adm: {missing_cols}")

    output_df["adm"] = output_df["xad"] / output_df["me"]

    return output_df