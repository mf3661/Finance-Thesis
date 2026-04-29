from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_ttm4(column_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Compute trailing-four-quarter sum.
    """

    return (
        df[column_name]
        + df.groupby("permno")[column_name].shift(1)
        + df.groupby("permno")[column_name].shift(2)
        + df.groupby("permno")[column_name].shift(3)
    )


def add_cfp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly cash-flow-to-price variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ibq",
        "dpq",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly cfp: {missing_cols}")

    output_df["ibq4"] = _compute_ttm4("ibq", output_df)
    output_df["dpq4"] = _compute_ttm4("dpq", output_df)

    output_df["cfp"] = np.where(
        output_df["dpq"].isna(),
        output_df["ibq4"] / output_df["me"],
        (output_df["ibq4"] + output_df["dpq4"]) / output_df["me"],
    )

    return output_df