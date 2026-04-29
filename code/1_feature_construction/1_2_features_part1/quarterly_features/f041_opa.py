from __future__ import annotations

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


def add_opa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly operating profitability over assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "revtq",
        "cogsq0",
        "xsgaq0",
        "xrdq4",
        "atq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly opa: {missing_cols}")

    output_df["opa"] = (
        _compute_ttm4("revtq", output_df)
        - _compute_ttm4("cogsq0", output_df)
        - _compute_ttm4("xsgaq0", output_df)
        + output_df["xrdq4"]
    ) / output_df["atq_l4"]

    return output_df