from __future__ import annotations

import pandas as pd


def add_depr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add depreciation-to-PPE variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dp",
        "ppent",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for depr: {missing_cols}")

    output_df["depr"] = output_df["dp"] / output_df["ppent"]

    return output_df