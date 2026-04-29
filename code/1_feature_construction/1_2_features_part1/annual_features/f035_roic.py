from __future__ import annotations

import pandas as pd


def add_roic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add return on invested capital variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ebit",
        "nopi",
        "ceq",
        "lt",
        "che",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for roic: {missing_cols}")

    output_df["roic"] = (
        output_df["ebit"] - output_df["nopi"]
    ) / (
        output_df["ceq"]
        + output_df["lt"]
        - output_df["che"]
    )

    return output_df