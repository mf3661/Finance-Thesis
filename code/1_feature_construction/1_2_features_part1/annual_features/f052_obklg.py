from __future__ import annotations

import pandas as pd


def add_obklg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add off-balance-sheet obligations variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ob",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for obklg: {missing_cols}")

    output_df["obklg"] = output_df["ob"] / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df