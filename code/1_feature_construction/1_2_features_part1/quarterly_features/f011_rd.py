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


def add_rd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly R&D increase indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "xrdq",
        "xrdy",
        "atq",
        "atq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly rd: {missing_cols}")

    output_df["xrdq4"] = _compute_ttm4("xrdq", output_df)

    output_df["xrdq4"] = np.where(
        output_df["xrdq4"].isna(),
        output_df["xrdy"],
        output_df["xrdq4"],
    )

    output_df["xrdq4/atq_l4"] = (
        output_df["xrdq4"] / output_df["atq_l4"]
    )

    output_df["xrdq4/atq_l4_l4"] = (
        output_df.groupby("permno")["xrdq4/atq_l4"].shift(4)
    )

    output_df["rd"] = np.where(
        (
            (output_df["xrdq4"] / output_df["atq"])
            - output_df["xrdq4/atq_l4_l4"]
        )
        / output_df["xrdq4/atq_l4_l4"]
        > 0.05,
        1,
        0,
    )

    return output_df