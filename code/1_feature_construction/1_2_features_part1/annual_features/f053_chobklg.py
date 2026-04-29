from __future__ import annotations

import pandas as pd


def add_chobklg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in off-balance-sheet obligations variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ob",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chobklg: {missing_cols}")

    output_df["ob_l1"] = output_df.groupby("permno")["ob"].shift(1)

    output_df["chobklg"] = (
        output_df["ob"] - output_df["ob_l1"]
    ) / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df