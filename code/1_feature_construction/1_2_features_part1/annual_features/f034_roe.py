from __future__ import annotations

import pandas as pd


def add_roe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return on equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ib",
        "ceq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for roe: {missing_cols}")

    output_df["ceq_l1"] = output_df.groupby("permno")["ceq"].shift(1)

    output_df["roe"] = output_df["ib"] / output_df["ceq_l1"]

    return output_df