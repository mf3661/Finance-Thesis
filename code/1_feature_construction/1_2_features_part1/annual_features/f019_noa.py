from __future__ import annotations

import pandas as pd


def add_noa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add net operating assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "at",
        "che",
        "ivao",
        "dlc",
        "dltt",
        "mib",
        "pstk",
        "ceq",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for noa: {missing_cols}")

    operating_assets = (
        output_df["at"]
        - output_df["che"]
        - output_df["ivao"].fillna(0)
    )

    operating_liabilities = (
        output_df["at"]
        - output_df["dlc"].fillna(0)
        - output_df["dltt"].fillna(0)
        - output_df["mib"].fillna(0)
        - output_df["pstk"].fillna(0)
        - output_df["ceq"]
    )

    output_df["noa"] = (
        operating_assets - operating_liabilities
    ) / output_df["at_l1"]

    return output_df