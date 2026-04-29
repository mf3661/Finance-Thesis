from __future__ import annotations

import numpy as np
import pandas as pd


def add_divi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dividend initiation indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "dvt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for divi: {missing_cols}")

    output_df["dvt_l1"] = output_df.groupby("permno")["dvt"].shift(1)

    output_df["divi"] = np.where(
        (
            output_df["dvt"].notna()
            & (output_df["dvt"] > 0)
            & (
                (output_df["dvt_l1"] == 0)
                | output_df["dvt_l1"].isna()
            )
        ),
        1,
        0,
    )

    return output_df