from __future__ import annotations

import numpy as np
import pandas as pd


def add_bm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly book-to-market variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "seqq",
        "txditcq",
        "pstkq",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly bm: {missing_cols}")

    output_df["beq"] = np.where(
        output_df["seqq"] > 0,
        output_df["seqq"] + output_df["txditcq"] - output_df["pstkq"],
        np.nan,
    )

    output_df["beq"] = np.where(
        output_df["beq"] <= 0,
        np.nan,
        output_df["beq"],
    )

    output_df["bm"] = output_df["beq"] / output_df["me"]

    return output_df