from __future__ import annotations

import numpy as np
import pandas as pd


def add_divo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dividend omission indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dvt",
        "dvt_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for divo: {missing_cols}")

    output_df["divo"] = np.where(
        (
            (output_df["dvt"].isna() | (output_df["dvt"] == 0))
            & (output_df["dvt_l1"] > 0)
        ),
        1,
        0,
    )

    return output_df