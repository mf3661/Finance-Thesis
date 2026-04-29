from __future__ import annotations

import pandas as pd


def _compute_ttm12(column_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Compute trailing-twelve-month sum.
    """

    result = df[column_name].copy()

    for lag in range(1, 12):
        result = result + df.groupby("permno")[column_name].shift(lag)

    return result


def add_dy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dividend yield variable from CRSP monthly returns.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "me",
        "ret",
        "retx",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for dy: {missing_cols}")

    output_df["me_l1"] = output_df.groupby("permno")["me"].shift(1)
    output_df["retdy"] = output_df["ret"] - output_df["retx"]
    output_df["mdivpay"] = output_df["retdy"] * output_df["me_l1"]

    output_df["dy"] = (
        _compute_ttm12("mdivpay", output_df)
        / output_df["me"]
    )

    return output_df