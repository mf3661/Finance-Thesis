from __future__ import annotations

import pandas as pd


def add_lgr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly liability growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ltq",
        "ltq_l4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly lgr: {missing_cols}")

    output_df["lgr"] = (
        output_df["ltq"] / output_df["ltq_l4"]
    ) - 1

    return output_df