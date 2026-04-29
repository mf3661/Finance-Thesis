from __future__ import annotations

import pandas as pd


def add_ala(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly asset liquidity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "cheq",
        "actq",
        "atq",
        "gdwlq",
        "intanq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly ala: {missing_cols}")

    gdwlq0 = output_df["gdwlq"].fillna(0)
    intanq0 = output_df["intanq"].fillna(0)

    output_df["ala"] = (
        output_df["cheq"]
        + 0.75 * (output_df["actq"] - output_df["cheq"])
        + 0.5 * (
            output_df["atq"]
            - output_df["actq"]
            - gdwlq0
            - intanq0
        )
    )

    return output_df