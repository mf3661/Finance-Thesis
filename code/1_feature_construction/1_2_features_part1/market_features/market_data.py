from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_crsp_delisting_return(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load CRSP monthly delisting returns.
    """

    print("Loading CRSP delisting returns...")

    dlret_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            permno, dlret, dlstdt
        from crsp_df.msedelist
        where dlstdt >= '{start_date}'
        and dlstdt <= '{end_date}'
        """
    )

    dlret_df["permno"] = dlret_df["permno"].astype(int)
    dlret_df["dlret"] = pd.to_numeric(dlret_df["dlret"], errors="coerce")
    dlret_df["dlstdt"] = pd.to_datetime(dlret_df["dlstdt"])
    dlret_df["jdate"] = dlret_df["dlstdt"] + MonthEnd(0)

    return dlret_df


def prepare_market_data(
    crsp_monthly_df: pd.DataFrame,
    dlret_df: pd.DataFrame,
    scale_me_to_millions: bool = True,
) -> pd.DataFrame:
    """
    Prepare CRSP monthly data for market features.
    """

    output_df: pd.DataFrame = crsp_monthly_df.copy()
    delist_df: pd.DataFrame = dlret_df.copy()

    required_cols: list[str] = [
        "permno",
        "date",
        "ret",
        "retx",
        "prc",
        "shrout",
        "vol",
        "me",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for market data: {missing_cols}")

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["date"] = pd.to_datetime(output_df["date"])

    if "monthend" in output_df.columns:
        output_df["jdate"] = pd.to_datetime(output_df["monthend"])
    else:
        output_df["jdate"] = output_df["date"] + MonthEnd(0)

    numeric_cols: list[str] = [
        "ret",
        "retx",
        "prc",
        "shrout",
        "vol",
        "me",
    ]

    for col in numeric_cols:
        output_df[col] = pd.to_numeric(output_df[col], errors="coerce")

    output_df = (
        output_df.dropna(subset=["ret", "retx", "prc"])
        .sort_values(by=["permno", "date"])
        .reset_index(drop=True)
    )

    if scale_me_to_millions:
        output_df["me"] = output_df["me"] / 1000

    delist_df = delist_df[["permno", "jdate", "dlret"]].copy()
    delist_df["permno"] = delist_df["permno"].astype(int)
    delist_df["jdate"] = pd.to_datetime(delist_df["jdate"])
    delist_df["dlret"] = pd.to_numeric(delist_df["dlret"], errors="coerce")

    output_df = pd.merge(
        output_df,
        delist_df,
        how="left",
        on=["permno", "jdate"],
    ).reset_index(drop=True)

    output_df["dlret"] = output_df["dlret"].fillna(0)
    output_df["ret"] = output_df["ret"].fillna(0)

    output_df["retadj"] = (
        (1 + output_df["ret"])
        * (1 + output_df["dlret"])
        - 1
    )

    return output_df