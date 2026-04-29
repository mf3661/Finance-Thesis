from __future__ import annotations

import pandas as pd


def add_egr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly equity growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ceqq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly egr: {missing_cols}")

    output_df["ceqq_l4"] = output_df.groupby("permno")["ceqq"].shift(4)

    output_df["egr"] = (
        output_df["ceqq"] - output_df["ceqq_l4"]
    ) / output_df["ceqq_l4"]

    return output_df