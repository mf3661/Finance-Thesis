from __future__ import annotations

import pandas as pd


def add_mom12m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add twelve-month momentum variable.
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
        raise KeyError(f"Missing required columns for mom12m: {missing_cols}")

    mom12m = 1

    for lag in range(1, 12):
        lagged_ret = output_df.groupby("permno")["ret"].shift(lag)
        mom12m = mom12m * (1 + lagged_ret)

    output_df["mom12m"] = mom12m - 1

    return output_df