from __future__ import annotations

import numpy as np
import pandas as pd


def add_ni(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly net equity issuance variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "cshoq",
        "ajexq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly ni: {missing_cols}")

    output_df["cshoq_l4"] = output_df.groupby("permno")["cshoq"].shift(4)
    output_df["ajexq_l4"] = output_df.groupby("permno")["ajexq"].shift(4)

    output_df["ni"] = np.where(
        output_df["cshoq"].isna(),
        np.nan,
        np.log(output_df["cshoq"] * output_df["ajexq"]).replace(-np.inf, 0)
        - np.log(output_df["cshoq_l4"] * output_df["ajexq_l4"]),
    )

    return output_df