from __future__ import annotations

import pandas as pd


def add_mom6m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add six-month momentum variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ret",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for mom6m: {missing_cols}")

    mom6m = 1

    for lag in range(1, 6):
        lagged_ret = output_df.groupby("permno")["ret"].shift(lag)
        mom6m = mom6m * (1 + lagged_ret)

    output_df["mom6m"] = mom6m - 1

    return output_df