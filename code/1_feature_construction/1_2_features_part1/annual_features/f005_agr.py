from __future__ import annotations

import pandas as pd


def add_agr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add asset growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "at",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for agr: {missing_cols}")

    output_df["agr"] = (
        output_df["at"] - output_df["at_l1"]
    ) / output_df["at_l1"]

    return output_df