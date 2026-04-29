from __future__ import annotations

import pandas as pd


def add_pchsaleinv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent change in sales-to-inventory variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sale",
        "sale_l1",
        "invt",
        "invt_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pchsaleinv: {missing_cols}")

    output_df["pchsaleinv"] = (
        (output_df["sale"] / output_df["invt"])
        - (output_df["sale_l1"] / output_df["invt_l1"])
    ) / (
        output_df["sale_l1"] / output_df["invt_l1"]
    )

    return output_df