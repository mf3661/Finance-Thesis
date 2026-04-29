from __future__ import annotations

import pandas as pd


def add_cashdebt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cash-flow-to-debt variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ib",
        "dp",
        "lt",
        "lt_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for cashdebt: {missing_cols}")

    output_df["cashdebt"] = (
        output_df["ib"] + output_df["dp"]
    ) / (
        (output_df["lt"] + output_df["lt_l1"]) / 2
    )

    return output_df