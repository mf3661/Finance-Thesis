from __future__ import annotations

import numpy as np
import pandas as pd


def add_sin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sin stock indicator.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "sic",
        "naics",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for sin: {missing_cols}")

    sic = pd.to_numeric(output_df["sic"], errors="coerce")
    naics = output_df["naics"].astype("string")

    sin_condition = (
        ((2100 <= sic) & (sic <= 2199))
        | ((2080 <= sic) & (sic <= 2085))
        | (naics == "7132")
        | (naics == "71312")
        | (naics == "713210")
        | (naics == "71329")
        | (naics == "713290")
        | (naics == "72112")
        | (naics == "721120")
    )

    output_df["sin"] = np.where(
        sin_condition.fillna(False),
        1,
        0,
    )

    return output_df