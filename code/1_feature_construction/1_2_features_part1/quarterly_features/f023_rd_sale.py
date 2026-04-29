from __future__ import annotations

import pandas as pd


def add_rd_sale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly R&D-to-sales variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "xrdq4",
        "saleq4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly rd_sale: {missing_cols}"
        )

    output_df["rd_sale"] = output_df["xrdq4"] / output_df["saleq4"]

    return output_df