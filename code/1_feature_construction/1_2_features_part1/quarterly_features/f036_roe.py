from __future__ import annotations

import pandas as pd


def add_roe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly return on equity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ibq",
        "ceqq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly roe: {missing_cols}")

    output_df["ceqq_l1"] = output_df.groupby("permno")["ceqq"].shift(1)

    output_df["roe"] = output_df["ibq"] / output_df["ceqq_l1"]

    return output_df