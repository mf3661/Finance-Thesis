from __future__ import annotations

import pandas as pd


def add_pchcurrat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent change in current ratio variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "act",
        "lct",
        "act_l1",
        "lct_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pchcurrat: {missing_cols}")

    output_df["pchcurrat"] = (
        (output_df["act"] / output_df["lct"])
        - (output_df["act_l1"] / output_df["lct_l1"])
    ) / (
        output_df["act_l1"] / output_df["lct_l1"]
    )

    return output_df