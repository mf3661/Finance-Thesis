from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_crsp_monthly(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load raw CRSP monthly stock and name data from WRDS.
    """

    print("Loading CRSP Monthly data...")

    crsp_df: pd.DataFrame = conn.raw_sql(
        f"""
        select

        /* monthly stock file */
        a.cfacpr, a.cfacshr, a.date, a.permco, a.permno,
        a.prc, a.ret, a.retx, a.shrout, a.vol,

        /* name history */
        b.comnam, b.exchcd, b.ncusip, b.shrcd, b.ticker

        from crsp_df.msf as a
        left join crsp_df.msenames as b
        on a.permno = b.permno
        and b.namedt <= a.date
        and a.date <= b.nameendt

        where a.date >= '{start_date}'
        and a.date <= '{end_date}'
        and b.exchcd between 1 and 3
        """
    )

    return crsp_df


def clean_crsp_monthly(
    crsp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean raw CRSP monthly data and construct stock-level market equity.
    """

    df: pd.DataFrame = crsp_df.copy()

    # CRSP identifiers and exchange/share codes should be integer-valued.
    # Explicit casting makes later filters and merges more stable.
    df[["permco", "permno", "shrcd", "exchcd"]] = df[
        ["permco", "permno", "shrcd", "exchcd"]
    ].astype(int)

    # Convert CRSP monthly date to datetime and align it to calendar month-end.
    # The month-end date is the key used later for matching CRSP with accounting data.
    df["date"] = pd.to_datetime(df["date"])
    df["monthend"] = df["date"] + MonthEnd(0)

    # Price is required for market equity.
    # Observations without price cannot produce a valid CRSP market equity.
    df = df.dropna(subset=["prc"]).reset_index(drop=True)

    # CRSP prices can be negative because negative prices represent bid-ask average prices.
    # Market equity should use absolute price times shares outstanding.
    df["me"] = df["prc"].abs() * df["shrout"]

    # If market equity is missing, the return observation is not usable for value-weighted
    # market calculations. The original code sets ret and retx to zero in these cases.
    df["ret"] = np.where(df["me"].isna(), 0, df["ret"])
    df["retx"] = np.where(df["me"].isna(), 0, df["retx"])

    # Sort before filling and aggregation so the panel order is stable.
    df = (
        df.sort_values(by=["permno", "date"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Forward-fill market equity within each security only.
    # This avoids losing observations when shares or price are temporarily missing.
    df["me"] = df.groupby("permno")["me"].ffill()

    print("Finished cleaning CRSP Monthly data.")

    return df


def aggregate_permco_market_equity(
    crsp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate CRSP market equity from security level to firm level.
    """

    df: pd.DataFrame = crsp_df.copy()

    # Some firms have multiple securities under the same permco at the same month-end.
    # Firm-level market equity is the sum across securities.
    market_equity_sum_df = (
        df.groupby(["monthend", "permco"], as_index=False)["me"]
        .sum()
        .rename(columns={"me": "me"})
    )

    # Identify the largest security within each firm-month.
    # The aggregated firm-level ME will be assigned to this representative permno.
    market_equity_max_df = (
        df.groupby(["monthend", "permco"], as_index=False)["me"]
        .max()
        .rename(columns={"me": "me"})
    )

    largest_security_df = pd.merge(
        df,
        market_equity_max_df,
        how="inner",
        on=["monthend", "permco", "me"],
    ).reset_index(drop=True)

    # Drop security-level ME before replacing it with firm-level aggregated ME.
    largest_security_df = largest_security_df.drop(columns=["me"])

    aligned_market_df = pd.merge(
        largest_security_df,
        market_equity_sum_df,
        how="inner",
        on=["monthend", "permco"],
    ).reset_index(drop=True)

    # If ties exist in the largest-security step, keep a unique permno-monthend record.
    aligned_market_df = (
        aligned_market_df.sort_values(by=["permno", "monthend"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Finished aggregating CRSP market equity.")

    return aligned_market_df


def load_and_clean_crsp_monthly(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load, clean, and aggregate CRSP monthly data.
    """

    raw_df: pd.DataFrame = load_crsp_monthly(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )

    clean_df: pd.DataFrame = clean_crsp_monthly(raw_df)

    aligned_market_df: pd.DataFrame = aggregate_permco_market_equity(clean_df)

    return aligned_market_df