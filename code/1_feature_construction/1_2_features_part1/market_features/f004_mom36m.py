from __future__ import annotations

import pandas as pd


def add_mom36m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add thirty-six-month momentum variable.
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
        raise KeyError(f"Missing required columns for mom36m: {missing_cols}")

    mom36m = 1

    for lag in range(12, 36):
        lagged_ret = output_df.groupby("permno")["ret"].shift(lag)
        mom36m = mom36m * (1 + lagged_ret)

    output_df["mom36m"] = mom36m - 1

    return output_df