from __future__ import annotations

import pandas as pd


def add_pchsale_pchinvt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales growth minus inventory growth variable.
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
        raise KeyError(
            f"Missing required columns for pchsale_pchinvt: {missing_cols}"
        )

    output_df["pchsale_pchinvt"] = (
        (output_df["sale"] - output_df["sale_l1"]) / output_df["sale_l1"]
    ) - (
        (output_df["invt"] - output_df["invt_l1"]) / output_df["invt_l1"]
    )

    return output_df