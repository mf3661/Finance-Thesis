from __future__ import annotations

import pandas as pd


def add_pchcapx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent change in capital expenditures variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "capx",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pchcapx: {missing_cols}")

    output_df["capx_l1"] = output_df.groupby("permno")["capx"].shift(1)

    output_df["pchcapx"] = (
        output_df["capx"] - output_df["capx_l1"]
    ) / output_df["capx_l1"]

    return output_df