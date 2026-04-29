from __future__ import annotations

import pandas as pd


def add_rna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return on net operating assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "noa",
        "oiadp",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rna: {missing_cols}")

    output_df["noa_l1"] = output_df.groupby("permno")["noa"].shift(1)

    output_df["rna"] = output_df["oiadp"] / output_df["noa_l1"]

    return output_df