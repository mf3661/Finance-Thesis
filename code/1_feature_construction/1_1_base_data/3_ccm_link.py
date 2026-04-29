from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, YearEnd

from functions import ffi49


def load_ccm_linktable(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """Load CRSP-Compustat link table from WRDS."""

    print("Loading CCM link table...")

    ccm_link_df: pd.DataFrame = conn.raw_sql(
        f"""
        select gvkey, lpermno as permno, linkdt, linkenddt,
               linkprim, linktype

        from crsp_df.ccmxpf_linktable

        where substr(linktype, 1, 1) = 'L'
        and (linkprim = 'C' or linkprim = 'P')
        and linkdt <= '{end_date}'
        and (linkenddt >= '{start_date}' or linkenddt is null)
        """
    )

    return ccm_link_df


def clean_ccm_linktable(
    ccm_link_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean CCM link table while preserving raw missing link dates."""

    df: pd.DataFrame = ccm_link_df.copy()

    # Convert link dates to datetime so they can be compared with accounting jdate.
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])

    # Keep the original missing linkenddt.
    # In CCM, missing linkenddt usually means the link is still active.
    # Use a separate helper only for link-window filtering.
    df["linkenddt_for_filter"] = df["linkenddt"].fillna(pd.Timestamp.today().normalize())

    # Do not fill other missing values here.
    # Invalid or unmatched links will be handled naturally by date filters and inner merges.
    df = df.sort_values(by=["gvkey", "permno", "linkdt"]).reset_index(drop=True)

    print("Finished cleaning CCM link table.")

    return df


def build_annual_base(
    compustat_annual_df: pd.DataFrame,
    crsp_monthly_df: pd.DataFrame,
    ccm_link_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build annual CRSP-Compustat linked base dataframe."""

    comp_df: pd.DataFrame = compustat_annual_df.copy()
    crsp_df: pd.DataFrame = crsp_monthly_df.copy()
    link_df: pd.DataFrame = ccm_link_df.copy()

    # Merge Compustat annual accounting records with CCM links by gvkey.
    # We keep this as a left merge first, then remove invalid links by date bounds.
    merged_df = pd.merge(
        comp_df,
        link_df,
        how="left",
        on=["gvkey"],
    ).reset_index(drop=True)

    # Annual accounting information is assumed to become available four months
    # after fiscal year-end, following the original code's lag rule.
    merged_df["yearend"] = merged_df["datadate"] + YearEnd(0)
    merged_df["jdate"] = merged_df["datadate"] + MonthEnd(4)

    # Keep only links that are active when the accounting data becomes available.
    linked_df = merged_df[
        (merged_df["jdate"] >= merged_df["linkdt"])
        & (merged_df["jdate"] <= merged_df["linkenddt_for_filter"])
    ].reset_index(drop=True)

    # Match linked Compustat records to CRSP monthly records by permno and jdate.
    crsp_df = crsp_df.rename(columns={"monthend": "jdate"})

    annual_base_df = pd.merge(
        crsp_df,
        linked_df,
        how="inner",
        on=["permno", "jdate"],
    ).reset_index(drop=True)

    # Keep common stocks listed on NYSE, AMEX, or NASDAQ.
    annual_base_df = annual_base_df[
        annual_base_df["exchcd"].isin([1, 2, 3])
        & annual_base_df["shrcd"].isin([10, 11])
    ].reset_index(drop=True)

    # Use CRSP market equity and convert it from thousands to millions.
    # Compustat market equity mve_f is kept as a separate reference variable.
    annual_base_df["me"] = annual_base_df["me"] / 1000

    # Zero market equity is not economically meaningful for accounting ratios.
    annual_base_df["me"] = np.where(
        annual_base_df["me"] == 0,
        np.nan,
        annual_base_df["me"],
    )
    annual_base_df = annual_base_df.dropna(subset=["me"]).reset_index(drop=True)

    # Count observations within each gvkey after the CRSP-Compustat link is formed.
    annual_base_df["count"] = annual_base_df.groupby("gvkey").cumcount() + 1

    # Resolve duplicate links in the same spirit as the original code.
    annual_base_df = _keep_first_within_group(
        annual_base_df,
        group_cols=["datadate", "permno", "linkprim"],
    )

    annual_base_df = _keep_last_within_group(
        annual_base_df,
        group_cols=["permno", "yearend", "datadate"],
    )

    annual_base_df = (
        annual_base_df.sort_values(by=["permno", "jdate"])
        .reset_index(drop=True)
    )

    annual_base_df = add_ffi49(
        annual_base_df,
        drop_missing_sic=False,
    )

    print("Finished building annual linked base.")

    return annual_base_df


def build_quarterly_base(
    compustat_quarterly_df: pd.DataFrame,
    crsp_monthly_df: pd.DataFrame,
    ccm_link_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build quarterly CRSP-Compustat linked base dataframe."""

    comp_df: pd.DataFrame = compustat_quarterly_df.copy()
    crsp_df: pd.DataFrame = crsp_monthly_df.copy()
    link_df: pd.DataFrame = ccm_link_df.copy()

    # Merge Compustat quarterly accounting records with CCM links by gvkey.
    merged_df = pd.merge(
        comp_df,
        link_df,
        how="left",
        on=["gvkey"],
    ).reset_index(drop=True)

    # Quarterly accounting data are first aligned using a four-month lag.
    merged_df["yearend"] = merged_df["datadate"] + YearEnd(0)
    merged_df["jdate"] = merged_df["datadate"] + MonthEnd(4)

    # Use rdq to update ibq when the next quarter's earnings announcement is
    # already available by the current jdate. Keep raw rdq and use helper columns.
    merged_df = merged_df.sort_values(by=["permno", "datadate"]).reset_index(drop=True)

    merged_df["rdq"] = pd.to_datetime(merged_df["rdq"]) + MonthEnd(0)
    merged_df["rdq_for_filter"] = merged_df["rdq"].fillna(merged_df["jdate"])

    merged_df["rdq_next"] = merged_df.groupby("permno")["rdq_for_filter"].shift(-1)
    merged_df["rdq_next"] = merged_df["rdq_next"].fillna(merged_df["jdate"])

    merged_df["ibq_diff"] = (merged_df["jdate"] - merged_df["rdq_next"]).dt.days
    merged_df["ibq_next"] = merged_df.groupby("permno")["ibq"].shift(-1)

    merged_df = merged_df.rename(columns={"ibq": "ibq_old"})

    merged_df["ibq"] = np.where(
        merged_df["ibq_diff"] >= 0,
        merged_df["ibq_next"],
        merged_df["ibq_old"],
    )

    # If the updated next-quarter value is missing, fall back to the original ibq.
    merged_df["ibq"] = np.where(
        merged_df["ibq"].isna(),
        merged_df["ibq_old"],
        merged_df["ibq"],
    )

    # Keep only links that are active when the accounting data becomes available.
    linked_df = merged_df[
        (merged_df["jdate"] >= merged_df["linkdt"])
        & (merged_df["jdate"] <= merged_df["linkenddt_for_filter"])
    ].reset_index(drop=True)

    # Match linked Compustat records to CRSP monthly records by permno and jdate.
    crsp_df = crsp_df.rename(columns={"monthend": "jdate"})

    quarterly_base_df = pd.merge(
        crsp_df,
        linked_df,
        how="inner",
        on=["permno", "jdate"],
    ).reset_index(drop=True)

    # Keep common stocks listed on NYSE, AMEX, or NASDAQ.
    quarterly_base_df = quarterly_base_df[
        quarterly_base_df["exchcd"].isin([1, 2, 3])
        & quarterly_base_df["shrcd"].isin([10, 11])
    ].reset_index(drop=True)

    # Use CRSP market equity and convert it from thousands to millions.
    quarterly_base_df["me"] = quarterly_base_df["me"] / 1000

    # Zero market equity is not economically meaningful for accounting ratios.
    quarterly_base_df["me"] = np.where(
        quarterly_base_df["me"] == 0,
        np.nan,
        quarterly_base_df["me"],
    )
    quarterly_base_df = quarterly_base_df.dropna(subset=["me"]).reset_index(drop=True)

    # Resolve duplicate links in the same spirit as the original code.
    quarterly_base_df = _keep_first_within_group(
        quarterly_base_df,
        group_cols=["datadate", "permno", "linkprim"],
    )

    quarterly_base_df = _keep_last_within_group(
        quarterly_base_df,
        group_cols=["permno", "yearend", "datadate"],
    )

    quarterly_base_df = (
        quarterly_base_df.sort_values(by=["permno", "jdate"])
        .reset_index(drop=True)
    )

    # The original quarterly block drops missing sic before assigning ffi49.
    quarterly_base_df = add_ffi49(
        quarterly_base_df,
        drop_missing_sic=True,
    )

    print("Finished building quarterly linked base.")

    return quarterly_base_df


def build_monthly_base(
    crsp_monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build monthly CRSP-only base dataframe."""

    monthly_base_df: pd.DataFrame = crsp_monthly_df.copy()

    monthly_base_df = (
        monthly_base_df.sort_values(by=["permno", "monthend"])
        .reset_index(drop=True)
    )

    print("Finished building monthly base.")

    return monthly_base_df


def add_ffi49(
    df: pd.DataFrame,
    drop_missing_sic: bool,
) -> pd.DataFrame:
    """Add Fama-French 49 industry classification."""

    output_df: pd.DataFrame = df.copy()

    output_df["sic"] = pd.to_numeric(output_df["sic"], errors="coerce")

    if drop_missing_sic:
        output_df = output_df.dropna(subset=["sic"]).reset_index(drop=True)

    # ffi49 expects a numeric sic column and returns NaN for unmatched ranges.
    output_df["ffi49"] = ffi49(output_df)
    output_df["ffi49"] = output_df["ffi49"].fillna(49).astype(int)

    # Use regular int when possible; otherwise preserve nullable integer sic.
    if output_df["sic"].isna().any():
        output_df["sic"] = output_df["sic"].astype("Int64")
    else:
        output_df["sic"] = output_df["sic"].astype(int)

    return output_df


def _keep_first_within_group(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Keep the first row within each duplicate group."""

    return (
        df.groupby(group_cols, as_index=False, group_keys=False)
        .nth(0)
        .reset_index(drop=True)
    )


def _keep_last_within_group(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Keep the last row within each duplicate group."""

    return (
        df.groupby(group_cols, as_index=False, group_keys=False)
        .nth(-1)
        .reset_index(drop=True)
    )


def load_and_clean_ccm_linktable(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """Load and clean CCM link table."""

    raw_df: pd.DataFrame = load_ccm_linktable(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )
    clean_df: pd.DataFrame = clean_ccm_linktable(raw_df)

    return clean_df