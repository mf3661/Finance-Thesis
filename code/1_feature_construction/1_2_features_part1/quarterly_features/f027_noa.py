from __future__ import annotations

import pandas as pd


def add_noa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly net operating assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "atq",
        "cheq",
        "ivaoq",
        "dlcq",
        "dlttq",
        "mibq",
        "pstkq",
        "ceqq",
        "atq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly noa: {missing_cols}")

    ivaoq0 = output_df["ivaoq"].fillna(0)
    dlcq0 = output_df["dlcq"].fillna(0)
    dlttq0 = output_df["dlttq"].fillna(0)
    mibq0 = output_df["mibq"].fillna(0)
    pstkq0 = output_df["pstkq"].fillna(0)

    operating_assets = (
        output_df["atq"]
        - output_df["cheq"]
        - ivaoq0
    )

    operating_liabilities = (
        output_df["atq"]
        - dlcq0
        - dlttq0
        - mibq0
        - pstkq0
        - output_df["ceqq"]
    )

    output_df["noa"] = (
        operating_assets - operating_liabilities
    ) / output_df["atq_l4"]

    return output_df