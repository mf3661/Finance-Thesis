from __future__ import annotations

import numpy as np
import pandas as pd


def add_securedind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add secured debt indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dm",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for securedind: {missing_cols}")

    output_df["securedind"] = np.where(
        output_df["dm"].notna() & (output_df["dm"] != 0),
        1,
        0,
    )

    return output_df