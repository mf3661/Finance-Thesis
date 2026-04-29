from __future__ import annotations

import pandas as pd


def add_chato(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in asset turnover variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "sale",
        "sale_l1",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chato: {missing_cols}")

    output_df["at_l2"] = output_df.groupby("permno")["at"].shift(2)

    output_df["chato"] = (
        output_df["sale"] / ((output_df["at"] + output_df["at_l1"]) / 2)
    ) - (
        output_df["sale_l1"] / ((output_df["at"] + output_df["at_l2"]) / 2)
    )

    return output_df