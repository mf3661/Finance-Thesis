from __future__ import annotations

import pandas as pd


def add_xadint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advertising expense intensity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "xad",
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for xadint: {missing_cols}")

    output_df["xadint"] = output_df["xad"] / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df