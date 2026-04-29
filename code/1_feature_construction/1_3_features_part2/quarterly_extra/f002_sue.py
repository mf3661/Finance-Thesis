from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


SUE_OUTPUT_COLUMNS: list[str] = [
    "gvkey",
    "permno",
    "datadate",
    "date",
    "jdate",
    "sue",
]


def load_sue_source_data(
    conn: Any,
    start_date: str = "01/01/1925",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load source data for SUE.
    """

    comp_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            gvkey,
            datadate,
            fyearq,
            fqtr,
            epspxq,
            ajexq
        from comp.fundq
        where indfmt = 'INDL'
        and datafmt = 'STD'
        and popsrc = 'D'
        and consol = 'C'
        and datadate >= '{start_date}'
        """
    )

    ccm_df: pd.DataFrame = conn.raw_sql(
        """
        select
            gvkey,
            lpermno as permno,
            linktype,
            linkprim,
            linkdt,
            linkenddt
        from crsp.ccmxpf_linktable
        where linktype in ('LU', 'LC')
        """
    )

    crsp_months_df: pd.DataFrame = conn.raw_sql(
        f"""
        select distinct
            date
        from crsp.msf
        where date >= '{start_date}'
        """
    )

    return comp_df, ccm_df, crsp_months_df


def _calculate_std(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.Series:
    """
    Calculate row-wise standard deviation with the original SUE constraints.
    """

    valid_counts = df[cols].notna().sum(axis=1) >= 6
    all_equal = df[cols].round(4).nunique(axis=1) == 1

    std_values = df[cols].std(axis=1)
    std_values[~valid_counts] = np.nan
    std_values[all_equal] = 0.0

    return std_values


def _format_dates_for_sql(
    df: pd.DataFrame,
    date_cols: list[str],
) -> pd.DataFrame:
    """
    Convert datetime columns to ISO date strings for SQLite range joins.
    """

    output_df: pd.DataFrame = df.copy()

    for col in date_cols:
        output_df[col] = pd.to_datetime(output_df[col])
        output_df[col] = output_df[col].dt.strftime("%Y-%m-%d")

    return output_df


def _link_compustat_to_crsp(
    comp_df: pd.DataFrame,
    ccm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Link Compustat quarterly records to CRSP permno through CCM.
    """

    comp_input_df: pd.DataFrame = comp_df.copy()
    ccm_input_df: pd.DataFrame = ccm_df.copy()

    required_comp_cols: list[str] = [
        "gvkey",
        "datadate",
        "fyearq",
        "fqtr",
        "epspxq",
        "ajexq",
    ]

    required_ccm_cols: list[str] = [
        "gvkey",
        "permno",
        "linkdt",
        "linkenddt",
    ]

    missing_comp_cols: list[str] = [
        col for col in required_comp_cols if col not in comp_input_df.columns
    ]

    if missing_comp_cols:
        raise KeyError(f"Missing required Compustat columns for sue: {missing_comp_cols}")

    missing_ccm_cols: list[str] = [
        col for col in required_ccm_cols if col not in ccm_input_df.columns
    ]

    if missing_ccm_cols:
        raise KeyError(f"Missing required CCM columns for sue: {missing_ccm_cols}")

    comp_input_df["datadate"] = pd.to_datetime(comp_input_df["datadate"])

    for col in ["epspxq", "ajexq"]:
        comp_input_df[col] = pd.to_numeric(
            comp_input_df[col],
            errors="coerce",
        )

    ccm_input_df["linkdt"] = pd.to_datetime(ccm_input_df["linkdt"])
    ccm_input_df["linkenddt"] = pd.to_datetime(ccm_input_df["linkenddt"])
    ccm_input_df["linkenddt"] = ccm_input_df["linkenddt"].fillna(
        pd.Timestamp.today().normalize()
    )

    merged_df = pd.merge(
        comp_input_df,
        ccm_input_df,
        how="left",
        on="gvkey",
    )

    merged_df = merged_df[
        (merged_df["datadate"] >= merged_df["linkdt"])
        & (merged_df["datadate"] <= merged_df["linkenddt"])
    ].copy()

    merged_df = merged_df[
        [
            "gvkey",
            "permno",
            "datadate",
            "fyearq",
            "fqtr",
            "epspxq",
            "ajexq",
        ]
    ].copy()

    merged_df = merged_df.dropna(subset=["permno"])
    merged_df["permno"] = merged_df["permno"].astype(int)

    return merged_df


def _compute_quarterly_sue(
    linked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute quarterly standardized unexpected earnings.
    """

    output_df: pd.DataFrame = linked_df.copy()

    output_df["eps"] = output_df["epspxq"] / output_df["ajexq"]

    output_df = output_df.drop_duplicates(
        subset=["permno", "datadate"],
    ).copy()

    output_df = output_df.sort_values(
        by=["permno", "datadate"]
    ).reset_index(drop=True)

    output_df["eps_diff"] = output_df.groupby("permno")["eps"].diff(4)

    output_df = output_df[
        output_df["eps"].notna()
        & output_df["eps_diff"].notna()
    ].reset_index(drop=True)

    output_df["count"] = (
        output_df.groupby("permno")
        .cumcount()
        + 1
    )

    for lag in range(1, 9):
        output_df[f"e{lag}"] = (
            output_df.groupby("permno")["eps"]
            .shift(lag)
        )

        output_df[f"e{lag}_diff"] = (
            output_df.groupby("permno")[f"e{lag}"]
            .diff(4)
        )

    condlist = [
        output_df["count"] <= 6,
        output_df["count"] == 7,
        output_df["count"] == 8,
        output_df["count"] >= 9,
    ]

    choicelist = [
        np.nan,
        _calculate_std(
            output_df,
            [f"{col}_diff" for col in ["e6", "e5", "e4", "e3", "e2", "e1"]],
        ),
        _calculate_std(
            output_df,
            [f"{col}_diff" for col in ["e7", "e6", "e5", "e4", "e3", "e2", "e1"]],
        ),
        _calculate_std(
            output_df,
            [f"{col}_diff" for col in ["e8", "e7", "e6", "e5", "e4", "e3", "e2", "e1"]],
        ),
    ]

    output_df["sue_std"] = np.select(
        condlist,
        choicelist,
        default=np.nan,
    )

    output_df["sue"] = (
        output_df["eps"] - output_df["e4"]
    ) / output_df["sue_std"]

    output_df = output_df[
        [
            "gvkey",
            "permno",
            "datadate",
            "sue",
        ]
    ].copy()

    return output_df


def _populate_sue_to_monthly(
    sue_quarterly_df: pd.DataFrame,
    crsp_months_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Populate quarterly SUE to monthly dates.
    """

    sue_input_df: pd.DataFrame = sue_quarterly_df.copy()
    months_df: pd.DataFrame = crsp_months_df.copy()

    sue_input_df["datadate"] = pd.to_datetime(sue_input_df["datadate"])

    sue_input_df["plus12m"] = (
        sue_input_df["datadate"]
        + pd.DateOffset(months=12)
        + MonthEnd(0)
    )

    months_df["date"] = pd.to_datetime(months_df["date"]) + MonthEnd(0)
    months_df = months_df[["date"]].drop_duplicates().copy()

    sue_sql_df = _format_dates_for_sql(
        df=sue_input_df,
        date_cols=[
            "datadate",
            "plus12m",
        ],
    )

    months_sql_df = _format_dates_for_sql(
        df=months_df,
        date_cols=["date"],
    )

    sql = sqlite3.connect(":memory:")

    try:
        sue_sql_df.to_sql("sue_quarterly", sql, index=False)
        months_sql_df.to_sql("crsp_msf", sql, index=False)

        query = """
            select
                a.*,
                b.date
            from sue_quarterly as a
            left join crsp_msf as b
            on a.datadate <= b.date
            and a.plus12m >= b.date
            order by
                a.permno,
                b.date,
                a.datadate desc;
        """

        populated_df = pd.read_sql_query(query, sql)

    finally:
        sql.close()

    populated_df = populated_df.drop_duplicates(
        subset=["permno", "date"]
    ).copy()

    populated_df = populated_df.dropna(subset=["date"]).copy()

    populated_df["datadate"] = pd.to_datetime(populated_df["datadate"])
    populated_df["date"] = pd.to_datetime(populated_df["date"])
    populated_df["jdate"] = populated_df["date"] + MonthEnd(0)

    populated_df["permno"] = populated_df["permno"].astype(int)

    output_df: pd.DataFrame = populated_df[SUE_OUTPUT_COLUMNS].copy()

    return output_df


def add_sue(
    comp_df: pd.DataFrame,
    ccm_df: pd.DataFrame,
    crsp_months_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly-populated quarterly SUE feature.
    """

    linked_df: pd.DataFrame = _link_compustat_to_crsp(
        comp_df=comp_df,
        ccm_df=ccm_df,
    )

    sue_quarterly_df: pd.DataFrame = _compute_quarterly_sue(
        linked_df=linked_df,
    )

    sue_df: pd.DataFrame = _populate_sue_to_monthly(
        sue_quarterly_df=sue_quarterly_df,
        crsp_months_df=crsp_months_df,
    )

    return sue_df


def build_sue_from_wrds(
    conn: Any,
    start_date: str = "01/01/1925",
) -> pd.DataFrame:
    """
    Load source data from WRDS and build SUE.
    """

    comp_df, ccm_df, crsp_months_df = load_sue_source_data(
        conn=conn,
        start_date=start_date,
    )

    sue_df: pd.DataFrame = add_sue(
        comp_df=comp_df,
        ccm_df=ccm_df,
        crsp_months_df=crsp_months_df,
    )

    return sue_df