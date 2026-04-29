from __future__ import annotations

import pandas as pd


def add_ato(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly asset turnover variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "saleq",
        "noa_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly ato: {missing_cols}")

    output_df["ato"] = output_df["saleq"] / output_df["noa_l4"]

    return output_df