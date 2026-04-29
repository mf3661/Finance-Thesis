from __future__ import annotations

import pandas as pd


def add_roa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return on assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ib",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for roa: {missing_cols}")

    output_df["roa"] = output_df["ib"] / output_df["at_l1"]

    return output_df