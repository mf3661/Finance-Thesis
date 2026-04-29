from __future__ import annotations

import numpy as np
import pandas as pd


def add_rd_sale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R&D-to-sales variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "xrd",
        "sale",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rd_sale: {missing_cols}")

    output_df["xrd0"] = np.where(
        output_df["xrd"].isna(),
        0,
        output_df["xrd"],
    )

    output_df["rd_sale"] = output_df["xrd0"] / output_df["sale"]

    return output_df