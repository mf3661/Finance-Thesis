from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_chars_std(
    start_lag: int,
    end_lag: int,
    df: pd.DataFrame,
    characteristic_name: str,
) -> pd.Series:
    """
    Compute row-wise standard deviation of lagged characteristics.
    """

    lagged_df = pd.DataFrame(index=df.index)
    lag_column_list: list[str] = []

    for i in range(start_lag, end_lag):
        lag_col = f"chars_l{i}"
        lagged_df[lag_col] = (
            df.groupby("permno")[characteristic_name].shift(i)
        )
        lag_column_list.append(lag_col)

    computed_result = lagged_df[lag_column_list].std(axis=1)

    return computed_result


def add_stdacc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly standard deviation of scaled accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "actq",
        "cheq",
        "lctq",
        "dlcq",
        "saleq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly stdacc: {missing_cols}"
        )

    output_df["actq_l1"] = output_df.groupby("permno")["actq"].shift(1)
    output_df["cheq_l1"] = output_df.groupby("permno")["cheq"].shift(1)
    output_df["lctq_l1"] = output_df.groupby("permno")["lctq"].shift(1)
    output_df["dlcq_l1"] = output_df.groupby("permno")["dlcq"].shift(1)

    scaled_accrual_numerator = (
        (
            output_df["actq"]
            - output_df["actq_l1"]
            - (output_df["cheq"] - output_df["cheq_l1"])
        )
        - (
            (output_df["lctq"] - output_df["lctq_l1"])
            - (output_df["dlcq"] - output_df["dlcq_l1"])
        )
    )

    output_df["sacc"] = scaled_accrual_numerator / output_df["saleq"]

    output_df["sacc"] = np.where(
        output_df["saleq"] <= 0,
        scaled_accrual_numerator / 0.01,
        output_df["sacc"],
    )

    output_df["stdacc"] = _compute_chars_std(
        start_lag=0,
        end_lag=16,
        df=output_df,
        characteristic_name="sacc",
    )

    return output_df