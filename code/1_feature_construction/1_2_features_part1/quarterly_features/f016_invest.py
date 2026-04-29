from __future__ import annotations

import numpy as np
import pandas as pd


def add_invest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly investment variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ppentq",
        "invtq",
        "ppegtq",
        "atq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly invest: {missing_cols}"
        )

    output_df["ppentq_l4"] = output_df.groupby("permno")["ppentq"].shift(4)
    output_df["invtq_l4"] = output_df.groupby("permno")["invtq"].shift(4)
    output_df["ppegtq_l4"] = output_df.groupby("permno")["ppegtq"].shift(4)

    output_df["invest"] = np.where(
        output_df["ppegtq"].isna(),
        (
            (output_df["ppentq"] - output_df["ppentq_l4"])
            + (output_df["invtq"] - output_df["invtq_l4"])
        ) / output_df["atq_l4"],
        (
            (output_df["ppegtq"] - output_df["ppegtq_l4"])
            + (output_df["invtq"] - output_df["invtq_l4"])
        ) / output_df["atq_l4"],
    )

    return output_df