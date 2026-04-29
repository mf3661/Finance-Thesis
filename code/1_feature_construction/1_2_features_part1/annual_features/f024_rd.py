from __future__ import annotations

import numpy as np
import pandas as pd


def add_rd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R&D increase indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "xrd0",
        "at_l1",
        "at",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rd: {missing_cols}")

    output_df["xrd/at_l1"] = output_df["xrd0"] / output_df["at_l1"]

    output_df["xrd/at_l1_l1"] = (
        output_df.groupby("permno")["xrd/at_l1"].shift(1)
    )

    output_df["rd"] = np.where(
        (
            (output_df["xrd0"] / output_df["at"])
            - output_df["xrd/at_l1_l1"]
        )
        / output_df["xrd/at_l1_l1"]
        > 0.05,
        1,
        0,
    )

    return output_df