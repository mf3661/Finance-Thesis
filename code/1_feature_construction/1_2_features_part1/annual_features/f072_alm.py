from __future__ import annotations

import pandas as pd


def add_alm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add asset liquidity over market assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ala",
        "at",
        "prcc_f",
        "csho",
        "ceq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for alm: {missing_cols}")

    output_df["alm"] = output_df["ala"] / (
        output_df["at"]
        + output_df["prcc_f"] * output_df["csho"]
        - output_df["ceq"]
    )

    return output_df