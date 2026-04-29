from __future__ import annotations

import pandas as pd


def add_egr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add equity growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ceq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for egr: {missing_cols}")

    output_df["ceq_l1"] = output_df.groupby("permno")["ceq"].shift(1)

    output_df["egr"] = (
        output_df["ceq"] - output_df["ceq_l1"]
    ) / output_df["ceq_l1"]

    return output_df