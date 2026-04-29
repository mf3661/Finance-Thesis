from __future__ import annotations

import pandas as pd


def add_chdrc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in deferred revenue variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "dr",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chdrc: {missing_cols}")

    output_df["dr_l1"] = output_df.groupby("permno")["dr"].shift(1)

    output_df["chdrc"] = (
        output_df["dr"] - output_df["dr_l1"]
    ) / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df