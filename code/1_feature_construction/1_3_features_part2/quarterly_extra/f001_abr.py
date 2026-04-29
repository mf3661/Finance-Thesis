from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


ABR_OUTPUT_COLUMNS: list[str] = [
    "gvkey",
    "permno",
    "datadate",
    "rdq",
    "rdq_plus_1d",
    "abr",
    "date",
    "jdate",
]


def load_abr_source_data(
    conn: Any,
    start_date: str = "01/01/1925",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load source data for ABR.
    """

    comp_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            gvkey,
            datadate,
            rdq,
            fyearq,
            fqtr
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

    crsp_trading_days_df: pd.DataFrame = conn.raw_sql(
        f"""
        select distinct
            date
        from crsp.dsi
        where date >= '{start_date}'
        """
    )

    crsp_daily_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            a.prc,
            a.ret,
            a.shrout,
            a.vol,
            a.cfacpr,
            a.cfacshr,
            a.permno,
            a.permco,
            a.date,
            b.siccd,
            b.ncusip,
            b.shrcd,
            b.exchcd
        from crsp.dsf as a
        left join crsp.dsenames as b
        on a.permno = b.permno
        and b.namedt <= a.date
        and a.date <= b.nameendt
        where a.date >= '{start_date}'
        and b.exchcd between 1 and 3
        and b.shrcd in (10, 11)
        """
    )

    daily_delist_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            permno,
            dlret,
            dlstdt
        from crsp.dsedelist
        where dlstdt >= '{start_date}'
        """
    )

    sp500_daily_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            date,
            sprtrn
        from crsp.dsi
        where date >= '{start_date}'
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

    return (
        comp_df,
        ccm_df,
        crsp_trading_days_df,
        crsp_daily_df,
        daily_delist_df,
        sp500_daily_df,
        crsp_months_df,
    )


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
        "rdq",
        "fyearq",
        "fqtr",
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
        raise KeyError(f"Missing required Compustat columns for abr: {missing_comp_cols}")

    missing_ccm_cols: list[str] = [
        col for col in required_ccm_cols if col not in ccm_input_df.columns
    ]

    if missing_ccm_cols:
        raise KeyError(f"Missing required CCM columns for abr: {missing_ccm_cols}")

    comp_input_df["datadate"] = pd.to_datetime(comp_input_df["datadate"])
    comp_input_df["rdq"] = pd.to_datetime(comp_input_df["rdq"])

    ccm_input_df["linkdt"] = pd.to_datetime(ccm_input_df["linkdt"])
    ccm_input_df["linkenddt"] = pd.to_datetime(ccm_input_df["linkenddt"])
    ccm_input_df["linkenddt"] = ccm_input_df["linkenddt"].fillna(
        pd.Timestamp.today().normalize()
    )

    ccm_merged_df = pd.merge(
        comp_input_df,
        ccm_input_df,
        how="left",
        on="gvkey",
    )

    ccm_merged_df = ccm_merged_df[
        (ccm_merged_df["datadate"] >= ccm_merged_df["linkdt"])
        & (ccm_merged_df["datadate"] <= ccm_merged_df["linkenddt"])
    ].copy()

    ccm_merged_df = ccm_merged_df[
        [
            "gvkey",
            "datadate",
            "rdq",
            "fyearq",
            "fqtr",
            "permno",
        ]
    ].copy()

    ccm_merged_df["permno"] = ccm_merged_df["permno"].astype(int)

    return ccm_merged_df


def _add_rdq_trading_day(
    linked_df: pd.DataFrame,
    crsp_trading_days_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert rdq to first CRSP trading day on or after rdq within 5 calendar days.
    """

    output_df: pd.DataFrame = linked_df.copy()
    trading_days_df: pd.DataFrame = crsp_trading_days_df.copy()

    trading_days_df["rdq_trad"] = pd.to_datetime(trading_days_df["date"])
    trading_days_df = (
        trading_days_df[["rdq_trad"]]
        .drop_duplicates()
        .sort_values("rdq_trad")
        .reset_index(drop=True)
    )

    output_df = output_df.dropna(subset=["rdq"]).copy()
    output_df = output_df.sort_values("rdq").reset_index(drop=True)

    output_df = pd.merge_asof(
        output_df,
        trading_days_df,
        left_on="rdq",
        right_on="rdq_trad",
        direction="forward",
        tolerance=pd.Timedelta(days=5),
    )

    output_df = output_df.dropna(subset=["rdq_trad"]).copy()

    output_df = output_df[
        [
            "gvkey",
            "permno",
            "datadate",
            "fyearq",
            "fqtr",
            "rdq",
            "rdq_trad",
        ]
    ].copy()

    return output_df


def _prepare_daily_abnormal_return(
    crsp_daily_df: pd.DataFrame,
    daily_delist_df: pd.DataFrame,
    sp500_daily_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare daily abnormal returns adjusted for delisting returns.
    """

    crsp_input_df: pd.DataFrame = crsp_daily_df.copy()
    delist_input_df: pd.DataFrame = daily_delist_df.copy()
    sp500_input_df: pd.DataFrame = sp500_daily_df.copy()

    required_crsp_cols: list[str] = [
        "permno",
        "permco",
        "date",
        "ret",
        "prc",
        "shrout",
        "shrcd",
        "exchcd",
    ]

    missing_crsp_cols: list[str] = [
        col for col in required_crsp_cols if col not in crsp_input_df.columns
    ]

    if missing_crsp_cols:
        raise KeyError(f"Missing required CRSP daily columns for abr: {missing_crsp_cols}")

    crsp_input_df[["permco", "permno", "shrcd", "exchcd"]] = crsp_input_df[
        ["permco", "permno", "shrcd", "exchcd"]
    ].astype(int)

    crsp_input_df["date"] = pd.to_datetime(crsp_input_df["date"])

    for col in ["ret", "prc", "shrout"]:
        crsp_input_df[col] = pd.to_numeric(crsp_input_df[col], errors="coerce")

    delist_input_df["permno"] = delist_input_df["permno"].astype(int)
    delist_input_df["dlstdt"] = pd.to_datetime(delist_input_df["dlstdt"])
    delist_input_df["dlret"] = pd.to_numeric(
        delist_input_df["dlret"],
        errors="coerce",
    )

    crsp_input_df = pd.merge(
        crsp_input_df,
        delist_input_df[["permno", "dlstdt", "dlret"]],
        how="left",
        left_on=["permno", "date"],
        right_on=["permno", "dlstdt"],
    ).reset_index(drop=True)

    crsp_input_df["retadj"] = np.where(
        crsp_input_df["dlret"].notna(),
        (1 + crsp_input_df["ret"]) * (1 + crsp_input_df["dlret"]) - 1,
        crsp_input_df["ret"],
    )

    crsp_input_df["meq"] = (
        crsp_input_df["prc"].abs() * crsp_input_df["shrout"]
    )

    crsp_input_df = crsp_input_df.sort_values(
        by=["date", "permno", "meq"]
    ).reset_index(drop=True)

    sp500_input_df["date"] = pd.to_datetime(sp500_input_df["date"])
    sp500_input_df["sprtrn"] = pd.to_numeric(
        sp500_input_df["sprtrn"],
        errors="coerce",
    )

    crsp_input_df = pd.merge(
        crsp_input_df,
        sp500_input_df,
        how="left",
        on="date",
    ).reset_index(drop=True)

    crsp_input_df["abrd"] = crsp_input_df["retadj"] - crsp_input_df["sprtrn"]

    crsp_input_df = crsp_input_df[
        [
            "date",
            "permno",
            "ret",
            "retadj",
            "sprtrn",
            "abrd",
        ]
    ].copy()

    return crsp_input_df


def _compute_quarterly_abr_events(
    linked_df: pd.DataFrame,
    crsp_abnormal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute quarterly ABR around earnings announcement.
    """

    event_df: pd.DataFrame = linked_df.copy()
    abnormal_df: pd.DataFrame = crsp_abnormal_df.copy()

    event_df["minus10d"] = event_df["rdq_trad"] - pd.Timedelta(days=10)
    event_df["plus5d"] = event_df["rdq_trad"] + pd.Timedelta(days=5)

    event_sql_df = _format_dates_for_sql(
        df=event_df,
        date_cols=[
            "datadate",
            "rdq",
            "rdq_trad",
            "minus10d",
            "plus5d",
        ],
    )

    abnormal_sql_df = _format_dates_for_sql(
        df=abnormal_df,
        date_cols=["date"],
    )

    sql = sqlite3.connect(":memory:")

    try:
        event_sql_df.to_sql("ccm3", sql, index=False)
        abnormal_sql_df.to_sql("crsp_d", sql, index=False)

        query = """
            select
                a.*,
                b.date,
                b.abrd
            from ccm3 as a
            left join crsp_d as b
            on a.permno = b.permno
            and a.minus10d <= b.date
            and b.date <= a.plus5d
            order by
                a.permno,
                a.rdq_trad,
                b.date;
        """

        joined_df = pd.read_sql_query(query, sql)

    finally:
        sql.close()

    joined_df = joined_df.drop(columns=["plus5d", "minus10d"])

    joined_df = joined_df[joined_df["abrd"].notna()].copy()

    date_cols = [
        "datadate",
        "rdq",
        "rdq_trad",
        "date",
    ]

    for col in date_cols:
        joined_df[col] = pd.to_datetime(joined_df[col])

    joined_df = joined_df.sort_values(
        by=["permno", "rdq_trad", "date"]
    ).reset_index(drop=True)

    condlist = [
        joined_df["date"] == joined_df["rdq_trad"],
        joined_df["date"] > joined_df["rdq_trad"],
        joined_df["date"] < joined_df["rdq_trad"],
    ]

    choicelist = [
        0,
        1,
        -1,
    ]

    joined_df["c_1"] = np.select(
        condlist,
        choicelist,
        default=np.nan,
    )

    before_df = joined_df[joined_df["c_1"] == -1].copy()
    before_df["count"] = (
        before_df.groupby(["permno", "rdq_trad"])["date"]
        .cumcount(ascending=False)
        + 1
    ) * -1

    after_df = joined_df[joined_df["c_1"] >= 0].copy()
    after_df["count"] = (
        after_df.groupby(["permno", "rdq_trad"])["date"]
        .cumcount()
    )

    joined_df = pd.concat(
        [
            before_df,
            after_df,
        ],
        axis=0,
        ignore_index=True,
    )

    joined_df = joined_df[
        (joined_df["count"] >= -2)
        & (joined_df["count"] <= 1)
    ].copy()

    abr_temp_df = (
        joined_df.groupby(["permno", "rdq_trad"], as_index=False)["abrd"]
        .sum()
        .rename(columns={"abrd": "abr"})
    )

    joined_df = pd.merge(
        joined_df,
        abr_temp_df,
        how="left",
        on=["permno", "rdq_trad"],
    )

    joined_df = joined_df[joined_df["count"] == 1].copy()

    joined_df = joined_df.rename(columns={"date": "rdq_plus_1d"})

    abr_event_df = joined_df[
        [
            "gvkey",
            "permno",
            "datadate",
            "rdq",
            "rdq_plus_1d",
            "abr",
        ]
    ].copy()

    return abr_event_df


def _populate_abr_to_monthly(
    abr_event_df: pd.DataFrame,
    crsp_months_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Populate quarterly ABR events to monthly dates.
    """

    event_df: pd.DataFrame = abr_event_df.copy()
    months_df: pd.DataFrame = crsp_months_df.copy()

    event_df["datadate"] = pd.to_datetime(event_df["datadate"])
    event_df["rdq"] = pd.to_datetime(event_df["rdq"])
    event_df["rdq_plus_1d"] = pd.to_datetime(event_df["rdq_plus_1d"])

    event_df["plus12m"] = (
        event_df["datadate"]
        + pd.DateOffset(months=12)
        + MonthEnd(0)
    )

    months_df["date"] = pd.to_datetime(months_df["date"]) + MonthEnd(0)
    months_df = months_df[["date"]].drop_duplicates().copy()

    event_sql_df = _format_dates_for_sql(
        df=event_df,
        date_cols=[
            "datadate",
            "rdq",
            "rdq_plus_1d",
            "plus12m",
        ],
    )

    months_sql_df = _format_dates_for_sql(
        df=months_df,
        date_cols=["date"],
    )

    sql = sqlite3.connect(":memory:")

    try:
        event_sql_df.to_sql("abr_event", sql, index=False)
        months_sql_df.to_sql("crsp_msf", sql, index=False)

        query = """
            select
                a.*,
                b.date
            from abr_event as a
            left join crsp_msf as b
            on a.rdq_plus_1d < b.date
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

    for col in [
        "datadate",
        "rdq",
        "rdq_plus_1d",
        "date",
    ]:
        populated_df[col] = pd.to_datetime(populated_df[col])

    populated_df["jdate"] = populated_df["date"] + MonthEnd(0)

    populated_df["permno"] = populated_df["permno"].astype(int)

    output_df = populated_df[ABR_OUTPUT_COLUMNS].copy()

    return output_df


def add_abr(
    comp_df: pd.DataFrame,
    ccm_df: pd.DataFrame,
    crsp_trading_days_df: pd.DataFrame,
    crsp_daily_df: pd.DataFrame,
    daily_delist_df: pd.DataFrame,
    sp500_daily_df: pd.DataFrame,
    crsp_months_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ABR monthly-populated quarterly feature.
    """

    linked_df: pd.DataFrame = _link_compustat_to_crsp(
        comp_df=comp_df,
        ccm_df=ccm_df,
    )

    linked_df = _add_rdq_trading_day(
        linked_df=linked_df,
        crsp_trading_days_df=crsp_trading_days_df,
    )

    crsp_abnormal_df: pd.DataFrame = _prepare_daily_abnormal_return(
        crsp_daily_df=crsp_daily_df,
        daily_delist_df=daily_delist_df,
        sp500_daily_df=sp500_daily_df,
    )

    abr_event_df: pd.DataFrame = _compute_quarterly_abr_events(
        linked_df=linked_df,
        crsp_abnormal_df=crsp_abnormal_df,
    )

    abr_df: pd.DataFrame = _populate_abr_to_monthly(
        abr_event_df=abr_event_df,
        crsp_months_df=crsp_months_df,
    )

    return abr_df


def build_abr_from_wrds(
    conn: Any,
    start_date: str = "01/01/1925",
) -> pd.DataFrame:
    """
    Load source data from WRDS and build ABR.
    """

    (
        comp_df,
        ccm_df,
        crsp_trading_days_df,
        crsp_daily_df,
        daily_delist_df,
        sp500_daily_df,
        crsp_months_df,
    ) = load_abr_source_data(
        conn=conn,
        start_date=start_date,
    )

    abr_df: pd.DataFrame = add_abr(
        comp_df=comp_df,
        ccm_df=ccm_df,
        crsp_trading_days_df=crsp_trading_days_df,
        crsp_daily_df=crsp_daily_df,
        daily_delist_df=daily_delist_df,
        sp500_daily_df=sp500_daily_df,
        crsp_months_df=crsp_months_df,
    )

    return abr_df