from __future__ import annotations

import pandas as pd


def add_ala(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add asset liquidity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "che",
        "act",
        "at",
        "gdwl",
        "intan",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for ala: {missing_cols}")

    gdwl0 = output_df["gdwl"].fillna(0)
    intan0 = output_df["intan"].fillna(0)

    output_df["ala"] = (
        output_df["che"]
        + 0.75 * (output_df["act"] - output_df["che"])
        - 0.5 * (output_df["at"] - output_df["act"] - gdwl0 - intan0)
    )

    return output_df