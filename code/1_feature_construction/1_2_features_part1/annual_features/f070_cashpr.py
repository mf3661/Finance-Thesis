from __future__ import annotations

import pandas as pd


def add_cashpr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cash productivity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "me",
        "dltt",
        "at",
        "che",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for cashpr: {missing_cols}")

    output_df["cashpr"] = (
        output_df["me"]
        + output_df["dltt"]
        - output_df["at"]
    ) / output_df["che"]

    return output_df