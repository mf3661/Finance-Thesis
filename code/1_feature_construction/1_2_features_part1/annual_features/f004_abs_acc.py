from __future__ import annotations

import pandas as pd


def add_absacc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add absolute accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "acc",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for absacc: {missing_cols}")

    output_df["absacc"] = output_df["acc"].abs()

    return output_df