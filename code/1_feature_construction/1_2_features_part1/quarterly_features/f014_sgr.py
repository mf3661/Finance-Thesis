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


def add_sgr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly sales growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "saleq",
        "saley",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly sgr: {missing_cols}")

    output_df["saleq4"] = _compute_ttm4("saleq", output_df)

    output_df["saleq4"] = np.where(
        output_df["saleq4"].isna(),
        output_df["saley"],
        output_df["saleq4"],
    )

    output_df["saleq4_l4"] = output_df.groupby("permno")["saleq4"].shift(4)

    output_df["sgr"] = (
        output_df["saleq4"] / output_df["saleq4_l4"]
    ) - 1

    return output_df