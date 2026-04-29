from __future__ import annotations

import pandas as pd


def add_chcsho(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly change in shares outstanding variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "cshoq",
        "cshoq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly chcsho: {missing_cols}"
        )

    output_df["chcsho"] = (
        output_df["cshoq"] / output_df["cshoq_l4"]
    ) - 1

    return output_df