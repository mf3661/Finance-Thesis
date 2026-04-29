from __future__ import annotations

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


def add_roavol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly return-on-assets volatility variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "roa",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly roavol: {missing_cols}"
        )

    output_df["roavol"] = _compute_chars_std(
        start_lag=0,
        end_lag=16,
        df=output_df,
        characteristic_name="roa",
    )

    return output_df