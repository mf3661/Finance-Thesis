from __future__ import annotations

import numpy as np
import pandas as pd


def add_dolvol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dollar trading volume variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "vol",
        "prc",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for dolvol: {missing_cols}")

    output_df["vol_l2"] = output_df.groupby("permno")["vol"].shift(2)
    output_df["prc_l2"] = output_df.groupby("permno")["prc"].shift(2)

    output_df["dolvol"] = np.log(
        (output_df["vol_l2"] * 100) * output_df["prc_l2"]
    ).replace([np.inf, -np.inf], np.nan)

    return output_df