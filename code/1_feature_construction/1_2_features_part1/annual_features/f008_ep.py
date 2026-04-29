from __future__ import annotations

import pandas as pd


def add_ep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add earnings-to-price variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ib",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for ep: {missing_cols}")

    output_df["ep"] = output_df["ib"] / output_df["me"]

    return output_df