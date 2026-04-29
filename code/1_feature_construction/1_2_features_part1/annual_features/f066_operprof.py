from __future__ import annotations

import pandas as pd


def add_operprof(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add operating profitability variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "revt",
        "cogs",
        "xsga0",
        "xint0",
        "ceq_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for operprof: {missing_cols}")

    output_df["operprof"] = (
        output_df["revt"]
        - output_df["cogs"]
        - output_df["xsga0"]
        - output_df["xint0"]
    ) / output_df["ceq_l1"]

    return output_df