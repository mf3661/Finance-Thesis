from __future__ import annotations

import pandas as pd


def add_pchsale_pchxsga(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sales growth minus SG&A growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "sale",
        "sale_l1",
        "xsga",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for pchsale_pchxsga: {missing_cols}"
        )

    output_df["xsga_l1"] = output_df.groupby("permno")["xsga"].shift(1)

    output_df["pchsale_pchxsga"] = (
        (output_df["sale"] - output_df["sale_l1"]) / output_df["sale_l1"]
    ) - (
        (output_df["xsga"] - output_df["xsga_l1"]) / output_df["xsga_l1"]
    )

    return output_df