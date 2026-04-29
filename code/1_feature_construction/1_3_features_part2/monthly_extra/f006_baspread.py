from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_baspread_daily_data(
    conn: Any,
    start_date: str = "01/01/1959",
) -> pd.DataFrame:
    """
    Load daily CRSP data for bid-ask spread.
    """

    daily_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            a.permno,
            a.date,
            a.ret,
            a.askhi,
            a.bidlo,
            a.vol
        from crsp.dsf as a
        where a.date > '{start_date}'
        """
    )

    daily_df["permno"] = daily_df["permno"].astype(int)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    numeric_cols: list[str] = [
        "ret",
        "askhi",
        "bidlo",
        "vol",
    ]

    for col in numeric_cols:
        daily_df[col] = pd.to_numeric(
            daily_df[col],
            errors="coerce",
        )

    daily_df = (
        daily_df.sort_values(by=["permno", "date"])
        .reset_index(drop=True)
    )

    return daily_df


def prepare_monthly_sequence(
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each daily observation to a firm-month sequence.
    """

    output_df: pd.DataFrame = daily_df.copy()

    required_cols: list[str] = [
        "permno",
        "date",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for baspread monthly sequence: {missing_cols}"
        )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["date"] = pd.to_datetime(output_df["date"])

    output_df = (
        output_df.sort_values(by=["permno", "date"])
        .reset_index(drop=True)
    )

    output_df["monthend"] = output_df["date"] + MonthEnd(0)
    output_df["date_diff"] = output_df["monthend"] - output_df["date"]

    month_end_diff_df = (
        output_df.groupby(["permno", "monthend"], as_index=False)["date_diff"]
        .min()
        .rename(columns={"date_diff": "min_diff"})
    )

    output_df = pd.merge(
        output_df,
        month_end_diff_df,
        how="left",
        on=["permno", "monthend"],
    ).reset_index(drop=True)

    output_df["month_end_flag"] = np.where(
        output_df["date_diff"] == output_df["min_diff"],
        1,
        np.nan,
    )

    output_df["monthly_sequence"] = np.nan

    month_end_mask = output_df["month_end_flag"] == 1

    output_df.loc[month_end_mask, "monthly_sequence"] = (
        output_df.loc[month_end_mask]
        .groupby("permno")
        .cumcount()
    )

    output_df["monthly_sequence"] = (
        output_df.groupby("permno")["monthly_sequence"]
        .bfill()
    )

    return output_df


def _estimate_baspread_for_window(
    window_df: pd.DataFrame,
    min_obs: int,
) -> float | None:
    """
    Estimate bid-ask spread from one rolling daily window.
    """

    if len(window_df) < min_obs:
        return None

    if window_df["vol"].notna().sum() < min_obs:
        return None

    spread_df = window_df.dropna(
        subset=[
            "askhi",
            "bidlo",
        ]
    )

    if len(spread_df) < min_obs:
        return None

    midpoint = (spread_df["askhi"] + spread_df["bidlo"]) / 2

    daily_spread = (
        spread_df["askhi"] - spread_df["bidlo"]
    ) / midpoint

    daily_spread = daily_spread.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    baspread = daily_spread.mean()

    if pd.isna(baspread):
        return None

    return float(baspread)


def _compute_baspread_for_permnos(
    args: tuple[pd.DataFrame, list[int], int],
) -> pd.DataFrame:
    """
    Compute baspread for one chunk of permnos.
    """

    stock_data_df, permno_list, min_obs = args

    result_rows: list[dict[str, object]] = []

    for permno in permno_list:
        firm_df = stock_data_df[
            stock_data_df["permno"] == permno
        ].copy()

        firm_df = firm_df.sort_values("date")

        valid_months = (
            firm_df.loc[
                firm_df["month_end_flag"] == 1,
                "monthly_sequence",
            ]
            .dropna()
            .astype(int)
            .unique()
        )

        for month_idx in valid_months:
            window_df = firm_df[
                (firm_df["monthly_sequence"] >= month_idx - 2)
                & (firm_df["monthly_sequence"] <= month_idx)
            ]

            baspread = _estimate_baspread_for_window(
                window_df=window_df,
                min_obs=min_obs,
            )

            if baspread is None:
                continue

            latest_row = (
                window_df.sort_values("date")
                .tail(1)
                .iloc[0]
            )

            result_rows.append(
                {
                    "permno": permno,
                    "date": latest_row["date"],
                    "jdate": latest_row["monthend"],
                    "baspread": baspread,
                }
            )

    result_df = pd.DataFrame(result_rows)

    return result_df


def add_baspread(
    daily_df: pd.DataFrame,
    n_processes: int = 20,
    min_obs: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling three-month daily bid-ask spread.
    """

    input_df: pd.DataFrame = daily_df.copy()

    required_cols: list[str] = [
        "permno",
        "date",
        "askhi",
        "bidlo",
        "vol",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in input_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for baspread calculation: {missing_cols}"
        )

    stock_data_df: pd.DataFrame = prepare_monthly_sequence(
        daily_df=input_df,
    )

    permno_list: list[int] = (
        stock_data_df["permno"]
        .drop_duplicates()
        .astype(int)
        .tolist()
    )

    if not permno_list:
        return pd.DataFrame(
            columns=[
                "permno",
                "date",
                "jdate",
                "baspread",
            ]
        )

    if n_processes <= 1:
        baspread_df = _compute_baspread_for_permnos(
            (
                stock_data_df,
                permno_list,
                min_obs,
            )
        )
    else:
        n_chunks = min(
            n_processes,
            len(permno_list),
        )

        permno_chunks = [
            chunk.astype(int).tolist()
            for chunk in np.array_split(
                np.array(permno_list),
                n_chunks,
            )
            if len(chunk) > 0
        ]

        pool_args = [
            (
                stock_data_df,
                chunk,
                min_obs,
            )
            for chunk in permno_chunks
        ]

        with Pool(processes=n_chunks) as pool:
            result_list = pool.map(
                _compute_baspread_for_permnos,
                pool_args,
            )

        baspread_df = pd.concat(
            result_list,
            axis=0,
            ignore_index=True,
        )

    baspread_df = baspread_df.dropna(
        subset=["baspread"]
    ).reset_index(drop=True)

    baspread_df["permno"] = baspread_df["permno"].astype(int)
    baspread_df["date"] = pd.to_datetime(baspread_df["date"])
    baspread_df["jdate"] = pd.to_datetime(baspread_df["jdate"])

    baspread_df = baspread_df[
        [
            "permno",
            "date",
            "jdate",
            "baspread",
        ]
    ].copy()

    return baspread_df