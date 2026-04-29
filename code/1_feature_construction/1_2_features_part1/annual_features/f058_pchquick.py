from __future__ import annotations

import pandas as pd


def add_pchquick(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent change in quick ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "act",
        "invt",
        "lct",
        "act_l1",
        "invt_l1",
        "lct_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pchquick: {missing_cols}")

    quick_current = (
        output_df["act"] - output_df["invt"]
    ) / output_df["lct"]

    quick_lagged = (
        output_df["act_l1"] - output_df["invt_l1"]
    ) / output_df["lct_l1"]

    output_df["pchquick"] = (
        quick_current - quick_lagged
    ) / quick_lagged

    return output_df