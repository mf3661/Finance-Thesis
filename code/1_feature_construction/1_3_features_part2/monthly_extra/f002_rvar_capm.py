from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


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
            f"Missing required columns for rvar_capm monthly sequence: {missing_cols}"
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


def _estimate_capm_residual_variance(
    window_df: pd.DataFrame,
    min_obs: int,
) -> float | None:
    """
    Estimate CAPM residual variance from one rolling daily window.
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

    y = reg_df["exret"].to_numpy(dtype=float)
    x = reg_df["mktrf"].to_numpy(dtype=float)

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

    residual = y - x_matrix @ coef

    if len(residual) <= 1:
        return None

    rvar_capm = float(np.var(residual, ddof=1))

    return rvar_capm


def _compute_rvar_capm_for_permnos(
    args: tuple[pd.DataFrame, list[int], int],
) -> pd.DataFrame:
    """
    Compute CAPM residual variance for one chunk of permnos.
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

            rvar_capm = _estimate_capm_residual_variance(
                window_df=window_df,
                min_obs=min_obs,
            )

            if rvar_capm is None:
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
                    "rvar_capm": rvar_capm,
                }
            )

    result_df = pd.DataFrame(result_rows)

    return result_df


def add_rvar_capm(
    daily_df: pd.DataFrame,
    n_processes: int = 20,
    min_obs: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling three-month daily CAPM residual variance for each firm-month.
    """

    input_df: pd.DataFrame = daily_df.copy()

    required_cols: list[str] = [
        "permno",
        "date",
        "exret",
        "mktrf",
        "vol",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in input_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for rvar_capm calculation: {missing_cols}"
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
                "rvar_capm",
            ]
        )

    if n_processes <= 1:
        rvar_capm_df = _compute_rvar_capm_for_permnos(
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
                _compute_rvar_capm_for_permnos,
                pool_args,
            )

        rvar_capm_df = pd.concat(
            result_list,
            axis=0,
            ignore_index=True,
        )

    rvar_capm_df = rvar_capm_df.dropna(
        subset=["rvar_capm"]
    ).reset_index(drop=True)

    rvar_capm_df["permno"] = rvar_capm_df["permno"].astype(int)
    rvar_capm_df["date"] = pd.to_datetime(rvar_capm_df["date"])
    rvar_capm_df["jdate"] = pd.to_datetime(rvar_capm_df["jdate"])

    rvar_capm_df = rvar_capm_df[
        [
            "permno",
            "date",
            "jdate",
            "rvar_capm",
        ]
    ].copy()

    return rvar_capm_df