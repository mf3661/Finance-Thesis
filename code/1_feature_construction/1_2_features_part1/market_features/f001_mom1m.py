from __future__ import annotations

import pandas as pd


def add_mom1m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one-month momentum variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ret",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for mom1m: {missing_cols}")

    output_df["mom1m"] = output_df["ret"]

    return output_df