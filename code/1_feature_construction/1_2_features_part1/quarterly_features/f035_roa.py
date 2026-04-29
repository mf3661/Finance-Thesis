from __future__ import annotations

import pandas as pd


def add_roa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly return on assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ibq",
        "atq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly roa: {missing_cols}")

    output_df["atq_l1"] = output_df.groupby("permno")["atq"].shift(1)

    output_df["roa"] = output_df["ibq"] / output_df["atq_l1"]

    return output_df