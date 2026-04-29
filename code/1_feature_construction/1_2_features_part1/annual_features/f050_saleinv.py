from __future__ import annotations

import pandas as pd


def add_saleinv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales-to-inventory variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sale",
        "invt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for saleinv: {missing_cols}")

    output_df["saleinv"] = output_df["sale"] / output_df["invt"]

    return output_df