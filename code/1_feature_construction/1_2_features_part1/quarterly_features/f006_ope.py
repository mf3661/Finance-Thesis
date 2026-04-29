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


def add_ope(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly operating profitability over book equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "revtq",
        "cogsq",
        "xsgaq",
        "xintq",
        "beq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly ope: {missing_cols}")

    output_df["xintq0"] = np.where(
        output_df["xintq"].isna(),
        0,
        output_df["xintq"],
    )

    output_df["xsgaq0"] = np.where(
        output_df["xsgaq"].isna(),
        0,
        output_df["xsgaq"],
    )

    output_df["cogsq0"] = np.where(
        output_df["cogsq"].isna(),
        0,
        output_df["cogsq"],
    )

    output_df["beq_l4"] = output_df.groupby("permno")["beq"].shift(4)

    output_df["ope"] = (
        _compute_ttm4("revtq", output_df)
        - _compute_ttm4("cogsq0", output_df)
        - _compute_ttm4("xsgaq0", output_df)
        - _compute_ttm4("xintq0", output_df)
    ) / output_df["beq_l4"]

    return output_df