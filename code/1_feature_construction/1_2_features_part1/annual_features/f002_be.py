# f002_be.py

from __future__ import annotations

import numpy as np
import pandas as pd


def add_be(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add book equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "seq",
        "txditc",
        "ps",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for be: {missing_cols}")

    output_df["txditc0"] = output_df["txditc"].fillna(0)

    output_df["be"] = (
        output_df["seq"]
        + output_df["txditc0"]
        - output_df["ps"]
    )

    output_df["be"] = np.where(
        output_df["be"] > 0,
        output_df["be"],
        np.nan,
    )

    return output_df