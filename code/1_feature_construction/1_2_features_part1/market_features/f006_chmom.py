from __future__ import annotations

import pandas as pd


def add_chmom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in momentum variable.
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
        raise KeyError(f"Missing required columns for chmom: {missing_cols}")

    result_first_half = 1

    for lag in range(1, 12):
        lagged_ret = output_df.groupby("permno")["ret"].shift(lag)
        result_first_half = result_first_half * (1 + lagged_ret)

    result_second_half = 1

    for lag in range(7, 18):
        lagged_ret = output_df.groupby("permno")["ret"].shift(lag)
        result_second_half = result_second_half * (1 + lagged_ret)

    result_first_half = result_first_half - 1
    result_second_half = result_second_half - 1

    output_df["chmom"] = result_first_half - result_second_half

    return output_df