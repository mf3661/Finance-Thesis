from __future__ import annotations

import numpy as np
import pandas as pd


def add_convind(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add convertible debt indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dc",
        "cshrc",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for convind: {missing_cols}")

    output_df["convind"] = np.where(
        (
            output_df["dc"].notna()
            & (output_df["dc"] != 0)
        )
        | (
            output_df["cshrc"].notna()
            & (output_df["cshrc"] != 0)
        ),
        1,
        0,
    )

    return output_df