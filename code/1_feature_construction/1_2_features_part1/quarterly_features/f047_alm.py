from __future__ import annotations

import pandas as pd


def add_alm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly asset liquidity over market assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ala",
        "atq",
        "me",
        "ceqq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly alm: {missing_cols}"
        )

    output_df["alm"] = output_df["ala"] / (
        output_df["atq"]
        + output_df["me"]
        - output_df["ceqq"]
    )

    return output_df