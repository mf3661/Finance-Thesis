from __future__ import annotations

import numpy as np
import pandas as pd


def add_cfp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cash-flow-to-price variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dp",
        "ib",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for cfp: {missing_cols}")

    condition_list = [
        output_df["dp"].isna(),
        output_df["ib"].isna(),
    ]

    choice_list = [
        output_df["ib"] / output_df["me"],
        np.nan,
    ]

    output_df["cfp"] = np.select(
        condition_list,
        choice_list,
        default=(output_df["ib"] + output_df["dp"]) / output_df["me"],
    )

    return output_df