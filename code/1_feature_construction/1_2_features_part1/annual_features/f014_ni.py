from __future__ import annotations

import numpy as np
import pandas as pd


def add_ni(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add net equity issuance variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "gvkey",
        "csho",
        "ajex",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for ni: {missing_cols}")

    output_df["csho_l1"] = output_df.groupby("permno")["csho"].shift(1)
    output_df["ajex_l1"] = output_df.groupby("permno")["ajex"].shift(1)

    output_df["ni"] = np.where(
        output_df["gvkey"] != output_df["gvkey"].shift(1),
        np.nan,
        np.log(output_df["csho"] * output_df["ajex"]).replace(-np.inf, 0)
        - np.log(output_df["csho_l1"] * output_df["ajex_l1"]).replace(-np.inf, 0),
    )

    return output_df