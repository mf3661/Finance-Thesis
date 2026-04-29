from __future__ import annotations

import pandas as pd


def add_depr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly depreciation-to-PPE variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "dpq4",
        "ppentq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly depr: {missing_cols}")

    output_df["depr"] = output_df["dpq4"] / output_df["ppentq"]

    return output_df