from __future__ import annotations

import pandas as pd


def add_lgr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add liability growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "lt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for lgr: {missing_cols}")

    output_df["lt_l1"] = output_df.groupby("permno")["lt"].shift(1)

    output_df["lgr"] = (
        output_df["lt"] / output_df["lt_l1"]
    ) - 1

    return output_df