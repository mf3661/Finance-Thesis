from __future__ import annotations

import pandas as pd


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add firm age variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "count",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for age: {missing_cols}")

    output_df["age"] = output_df["count"].copy()

    return output_df