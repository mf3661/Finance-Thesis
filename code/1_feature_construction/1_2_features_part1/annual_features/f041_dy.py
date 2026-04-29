from __future__ import annotations

import pandas as pd


def add_dy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dividend yield variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dvt",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for dy: {missing_cols}")

    output_df["dy"] = output_df["dvt"] / output_df["me"]

    return output_df