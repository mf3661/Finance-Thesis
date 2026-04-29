from __future__ import annotations

import pandas as pd


def add_chinv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in inventory variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "invt",
        "invt_l1",
        "at",
        "at_l2",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chinv: {missing_cols}")

    output_df["chinv"] = (
        output_df["invt"] - output_df["invt_l1"]
    ) / (
        (output_df["at"] + output_df["at_l2"]) / 2
    )

    return output_df