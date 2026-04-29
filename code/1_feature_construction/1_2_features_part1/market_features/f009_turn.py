from __future__ import annotations

import pandas as pd


def add_turn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add share turnover variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "vol",
        "shrout",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for turn: {missing_cols}")

    output_df["vol_l1"] = output_df.groupby("permno")["vol"].shift(1)
    output_df["vol_l2"] = output_df.groupby("permno")["vol"].shift(2)
    output_df["vol_l3"] = output_df.groupby("permno")["vol"].shift(3)

    output_df["turn"] = (
        (
            output_df["vol_l1"]
            + output_df["vol_l2"]
            + output_df["vol_l3"]
        )
        / 3
        / 10
    ) / output_df["shrout"]

    return output_df