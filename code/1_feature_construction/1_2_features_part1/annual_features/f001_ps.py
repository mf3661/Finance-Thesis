from __future__ import annotations

import numpy as np
import pandas as pd


def add_ps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add preferred stock variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "pstkrv",
        "pstkl",
        "pstk",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for ps: {missing_cols}")

    output_df["ps"] = np.where(
        output_df["pstkrv"].isna(),
        output_df["pstkl"],
        output_df["pstkrv"],
    )

    output_df["ps"] = np.where(
        output_df["ps"].isna(),
        output_df["pstk"],
        output_df["ps"],
    )

    output_df["ps"] = np.where(
        output_df["ps"].isna(),
        0,
        output_df["ps"],
    )

    return output_df