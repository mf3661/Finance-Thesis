from __future__ import annotations

from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_beta_daily_data(
    conn: Any,
    start_date: str = "01/01/1950",
) -> pd.DataFrame:
    """
    Load daily CRSP returns, daily Fama-French factors, and daily delisting returns.
    """

    stock_returns_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            a.permno,
            a.date,
            a.ret,
            a.vol,
            b.rf,
            b.mktrf,
            b.smb,
            b.hml
        from crsp.dsf as a
        left join ff.factors_daily as b
        on a.date = b.date
        where a.date > '{start_date}'
        """
    )

    stock_returns_df["permno"] = stock_returns_df["permno"].astype(int)
    stock_returns_df["date"] = pd.to_datetime(stock_returns_df["date"])

    numeric_cols: list[str] = [
        "ret",
        "vol",
        "rf",
        "mktrf",
        "smb",
        "hml",
    ]

    for col in numeric_cols:
        stock_returns_df[col] = pd.to_numeric(
            stock_returns_df[col],
            errors="coerce",
        )

    delisting_returns_df: pd.DataFrame = conn.raw_sql(
        """
        select
            permno,
            dlret,
            dlstdt
        from crsp.dsedelist
        """
    )

    delisting_returns_df["permno"] = delisting_returns_df["permno"].astype(int)
    delisting_returns_df["dlstdt"] = pd.to_datetime(
        delisting_returns_df["dlstdt"]
    )
    delisting_returns_df["date"] = delisting_returns_df["dlstdt"]
    delisting_returns_df["dlret"] = pd.to_numeric(
        delisting_returns_df["dlret"],
        errors="coerce",
    )

    stock_returns_df = pd.merge(
        stock_returns_df,
        delisting_returns_df[["permno", "date", "dlret"]],
        how="left",
        on=["permno", "date"],
    ).reset_index(drop=True)

    stock_returns_df["dlret"] = stock_returns_df["dlret"].fillna(0)
    stock_returns_df["ret"] = stock_returns_df["ret"].fillna(0)

    stock_returns_df["retadj"] = (
        (1 + stock_returns_df["ret"])
        * (1 + stock_returns_df["dlret"])
        - 1
    )

    stock_returns_df["exret"] = (
        stock_returns_df["retadj"] - stock_returns_df["rf"]
    )

    stock_returns_df = stock_returns_df.sort_values(
        by=["permno", "date"]
    ).reset_index(drop=True)

    return stock_returns_df


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
            f"Missing required columns for beta monthly sequence: {missing_cols}"
        )

    output_df["date"] = pd.to_datetime(output_df["date"])
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


def _estimate_beta_for_window(
    window_df: pd.DataFrame,
    min_obs: int,
) -> float | None:
    """
    Estimate CAPM beta from one rolling daily window.
    """

    reg_df = window_df.dropna(
        subset=[
            "exret",
            "mktrf",
        ]
    )

    if len(reg_df) < min_obs:
        return None

    if reg_df["vol"].notna().sum() < min_obs:
        return None

    x = reg_df["mktrf"].to_numpy(dtype=float)
    y = reg_df["exret"].to_numpy(dtype=float)

    if np.nanvar(x) <= 0:
        return None

    x_matrix = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    try:
        coef = np.linalg.lstsq(
            x_matrix,
            y,
            rcond=None,
        )[0]
    except np.linalg.LinAlgError:
        return None

    beta = float(coef[1])

    return beta


def _compute_beta_for_permnos(
    args: tuple[pd.DataFrame, list[int], int],
) -> pd.DataFrame:
    """
    Compute beta for one chunk of permnos.
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

            beta = _estimate_beta_for_window(
                window_df=window_df,
                min_obs=min_obs,
            )

            if beta is None:
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
                    "beta": beta,
                }
            )

    result_df = pd.DataFrame(result_rows)

    return result_df


def add_beta(
    daily_df: pd.DataFrame,
    n_processes: int = 20,
    min_obs: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling three-month daily CAPM beta for each firm-month.
    """

    stock_data_df: pd.DataFrame = prepare_monthly_sequence(
        daily_df=daily_df,
    )

    required_cols: list[str] = [
        "permno",
        "date",
        "monthend",
        "monthly_sequence",
        "month_end_flag",
        "exret",
        "mktrf",
        "vol",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in stock_data_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for beta calculation: {missing_cols}"
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
                "beta",
            ]
        )

    if n_processes <= 1:
        beta_df = _compute_beta_for_permnos(
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
                _compute_beta_for_permnos,
                pool_args,
            )

        beta_df = pd.concat(
            result_list,
            axis=0,
            ignore_index=True,
        )

    beta_df = beta_df.dropna(
        subset=["beta"]
    ).reset_index(drop=True)

    beta_df["permno"] = beta_df["permno"].astype(int)
    beta_df["date"] = pd.to_datetime(beta_df["date"])
    beta_df["jdate"] = pd.to_datetime(beta_df["jdate"])

    beta_df = beta_df[
        [
            "permno",
            "date",
            "jdate",
            "beta",
        ]
    ].copy()

    return beta_df