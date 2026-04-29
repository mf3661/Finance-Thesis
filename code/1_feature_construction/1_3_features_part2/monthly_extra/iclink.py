from __future__ import annotations

from typing import Any

import pandas as pd
from fuzzywuzzy import fuzz


def load_iclink_source_data(
    conn: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw IBES and CRSP data needed to build the ICLINK table.
    """

    ibes_cusip_df: pd.DataFrame = conn.raw_sql(
        """
        select
            ticker,
            cusip,
            cname,
            sdates
        from ibes.id
        where usfirm = 1
        and cusip != ''
        """
    )

    crsp_cusip_df: pd.DataFrame = conn.raw_sql(
        """
        select
            permno,
            ncusip,
            comnam,
            namedt,
            nameenddt
        from crsp.stocknames
        where ncusip != ''
        """
    )

    ibes_ticker_df: pd.DataFrame = conn.raw_sql(
        """
        select
            ticker,
            cname,
            oftic,
            sdates,
            cusip
        from ibes.id
        """
    )

    crsp_ticker_df: pd.DataFrame = conn.raw_sql(
        """
        select
            ticker,
            comnam,
            permno,
            ncusip,
            namedt,
            nameenddt
        from crsp.stocknames
        """
    )

    return (
        ibes_cusip_df,
        crsp_cusip_df,
        ibes_ticker_df,
        crsp_ticker_df,
    )


def _prepare_ibes_cusip_data(
    ibes_cusip_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare IBES ticker-cusip-date data for CUSIP linking.
    """

    output_df: pd.DataFrame = ibes_cusip_df.copy()

    required_cols: list[str] = [
        "ticker",
        "cusip",
        "cname",
        "sdates",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required IBES CUSIP columns for iclink: {missing_cols}"
        )

    output_df["sdates"] = pd.to_datetime(output_df["sdates"])

    ibes_date_df = (
        output_df.groupby(["ticker", "cusip"])["sdates"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "fdate", "max": "ldate"})
    )

    output_df = pd.merge(
        output_df,
        ibes_date_df,
        how="left",
        on=["ticker", "cusip"],
    )

    output_df = output_df.sort_values(
        by=["ticker", "cusip", "sdates"]
    ).reset_index(drop=True)

    output_df = (
        output_df.loc[output_df["sdates"] == output_df["ldate"]]
        .drop(columns=["sdates"])
        .reset_index(drop=True)
    )

    return output_df


def _prepare_crsp_cusip_data(
    crsp_cusip_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare CRSP permno-ncusip-date data for CUSIP linking.
    """

    output_df: pd.DataFrame = crsp_cusip_df.copy()

    required_cols: list[str] = [
        "permno",
        "ncusip",
        "comnam",
        "namedt",
        "nameenddt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required CRSP CUSIP columns for iclink: {missing_cols}"
        )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["namedt"] = pd.to_datetime(output_df["namedt"])
    output_df["nameenddt"] = pd.to_datetime(output_df["nameenddt"])

    first_namedt_df = (
        output_df.groupby(["permno", "ncusip"])["namedt"]
        .min()
        .reset_index()
    )

    last_nameenddt_df = (
        output_df.groupby(["permno", "ncusip"])["nameenddt"]
        .max()
        .reset_index()
    )

    date_range_df = pd.merge(
        first_namedt_df,
        last_nameenddt_df,
        how="inner",
        on=["permno", "ncusip"],
    )

    output_df = output_df.drop(columns=["namedt"]).rename(
        columns={"nameenddt": "enddt"}
    )

    output_df = pd.merge(
        output_df,
        date_range_df,
        how="inner",
        on=["permno", "ncusip"],
    )

    output_df = (
        output_df.loc[output_df["enddt"] == output_df["nameenddt"]]
        .drop(columns=["enddt"])
        .reset_index(drop=True)
    )

    return output_df


def _score_cusip_link(
    row: pd.Series,
    name_ratio_cutoff: float,
) -> int:
    """
    Score CUSIP-based IBES-CRSP link.
    """

    date_match = (
        (row["fdate"] <= row["nameenddt"])
        and (row["ldate"] >= row["namedt"])
    )

    name_match = row["name_ratio"] >= name_ratio_cutoff

    if date_match and name_match:
        return 0

    if date_match:
        return 1

    if name_match:
        return 2

    return 3


def _build_cusip_link(
    ibes_prepared_df: pd.DataFrame,
    crsp_prepared_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Build ICLINK matches using full CUSIP.
    """

    link_df = pd.merge(
        ibes_prepared_df,
        crsp_prepared_df,
        how="inner",
        left_on="cusip",
        right_on="ncusip",
    ).sort_values(
        by=["ticker", "permno", "ldate"]
    )

    latest_link_df = (
        link_df.groupby(["ticker", "permno"])["ldate"]
        .max()
        .reset_index()
    )

    link_df = pd.merge(
        link_df,
        latest_link_df,
        how="inner",
        on=["ticker", "permno", "ldate"],
    )

    link_df["name_ratio"] = link_df.apply(
        lambda row: fuzz.token_set_ratio(row["comnam"], row["cname"]),
        axis=1,
    )

    name_ratio_cutoff = link_df["name_ratio"].quantile(0.10)

    link_df["score"] = link_df.apply(
        lambda row: _score_cusip_link(
            row=row,
            name_ratio_cutoff=name_ratio_cutoff,
        ),
        axis=1,
    )

    link_df = link_df[
        [
            "ticker",
            "permno",
            "cname",
            "comnam",
            "name_ratio",
            "score",
        ]
    ].drop_duplicates()

    return link_df, float(name_ratio_cutoff)


def _prepare_ibes_ticker_data(
    ibes_ticker_df: pd.DataFrame,
    unmatched_tickers_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare unmatched IBES tickers for exchange-ticker linking.
    """

    output_df: pd.DataFrame = ibes_ticker_df.copy()

    required_cols: list[str] = [
        "ticker",
        "cname",
        "oftic",
        "sdates",
        "cusip",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required IBES ticker columns for iclink: {missing_cols}"
        )

    output_df = output_df.loc[output_df["oftic"].notna()].copy()
    output_df["sdates"] = pd.to_datetime(output_df["sdates"])

    output_df = pd.merge(
        unmatched_tickers_df,
        output_df,
        how="inner",
        on="ticker",
    )

    date_range_df = (
        output_df.groupby(["ticker", "oftic"])["sdates"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "fdate", "max": "ldate"})
    )

    output_df = pd.merge(
        output_df,
        date_range_df,
        how="left",
        on=["ticker", "oftic"],
    )

    output_df = output_df.loc[
        output_df["sdates"] == output_df["ldate"]
    ].reset_index(drop=True)

    return output_df


def _prepare_crsp_ticker_data(
    crsp_ticker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare CRSP exchange ticker data for ticker linking.
    """

    output_df: pd.DataFrame = crsp_ticker_df.copy()

    required_cols: list[str] = [
        "ticker",
        "comnam",
        "permno",
        "ncusip",
        "namedt",
        "nameenddt",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required CRSP ticker columns for iclink: {missing_cols}"
        )

    output_df = output_df.loc[output_df["ticker"].notna()].copy()

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["namedt"] = pd.to_datetime(output_df["namedt"])
    output_df["nameenddt"] = pd.to_datetime(output_df["nameenddt"])

    output_df = output_df.sort_values(
        by=["permno", "ticker", "namedt"]
    ).reset_index(drop=True)

    first_namedt_df = (
        output_df.groupby(["permno", "ticker"])["namedt"]
        .min()
        .reset_index()
    )

    last_nameenddt_df = (
        output_df.groupby(["permno", "ticker"])["nameenddt"]
        .max()
        .reset_index()
    )

    date_range_df = pd.merge(
        first_namedt_df,
        last_nameenddt_df,
        how="inner",
        on=["permno", "ticker"],
    )

    output_df = output_df.rename(
        columns={
            "namedt": "namedt_ind",
            "nameenddt": "nameenddt_ind",
        }
    )

    output_df = pd.merge(
        output_df,
        date_range_df,
        how="left",
        on=["permno", "ticker"],
    )

    output_df = output_df.rename(columns={"ticker": "crsp_ticker"})

    output_df = output_df.loc[
        output_df["nameenddt_ind"] == output_df["nameenddt"]
    ].drop(
        columns=["namedt_ind", "nameenddt_ind"]
    ).reset_index(drop=True)

    return output_df


def _score_ticker_link(
    row: pd.Series,
    name_ratio_cutoff: float,
) -> int:
    """
    Score exchange-ticker-based IBES-CRSP link.
    """

    cusip_match = row["cusip6"] == row["ncusip6"]
    name_match = row["name_ratio"] >= name_ratio_cutoff

    if cusip_match and name_match:
        return 0

    if cusip_match:
        return 4

    if name_match:
        return 5

    return 6


def _build_ticker_link(
    ibes_prepared_df: pd.DataFrame,
    crsp_prepared_df: pd.DataFrame,
    cusip_link_df: pd.DataFrame,
    name_ratio_cutoff: float,
) -> pd.DataFrame:
    """
    Build ICLINK matches for unmatched tickers using exchange ticker.
    """

    unmatched_tickers_df = pd.merge(
        ibes_prepared_df[["ticker"]],
        cusip_link_df[["permno", "ticker"]],
        how="left",
        on="ticker",
    )

    unmatched_tickers_df = (
        unmatched_tickers_df.loc[unmatched_tickers_df["permno"].isna()]
        .drop(columns=["permno"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if unmatched_tickers_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "permno",
                "cname",
                "comnam",
                "score",
            ]
        )

    ticker_link_df = pd.merge(
        ibes_prepared_df,
        crsp_prepared_df,
        how="inner",
        left_on="oftic",
        right_on="crsp_ticker",
    )

    ticker_link_df = ticker_link_df.loc[
        (ticker_link_df["ldate"] >= ticker_link_df["namedt"])
        & (ticker_link_df["fdate"] <= ticker_link_df["nameenddt"])
    ].copy()

    if ticker_link_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "permno",
                "cname",
                "comnam",
                "score",
            ]
        )

    ticker_link_df["name_ratio"] = ticker_link_df.apply(
        lambda row: fuzz.token_set_ratio(row["comnam"], row["cname"]),
        axis=1,
    )

    ticker_link_df["cusip6"] = ticker_link_df["cusip"].astype(str).str[:6]
    ticker_link_df["ncusip6"] = ticker_link_df["ncusip"].astype(str).str[:6]

    ticker_link_df["score"] = ticker_link_df.apply(
        lambda row: _score_ticker_link(
            row=row,
            name_ratio_cutoff=name_ratio_cutoff,
        ),
        axis=1,
    )

    ticker_link_df = ticker_link_df[
        [
            "ticker",
            "permno",
            "cname",
            "comnam",
            "name_ratio",
            "score",
        ]
    ].sort_values(
        by=["ticker", "score"]
    )

    best_score_df = (
        ticker_link_df.groupby("ticker")["score"]
        .min()
        .reset_index()
    )

    ticker_link_df = pd.merge(
        ticker_link_df,
        best_score_df,
        how="inner",
        on=["ticker", "score"],
    )

    ticker_link_df = ticker_link_df[
        [
            "ticker",
            "permno",
            "cname",
            "comnam",
            "score",
        ]
    ].drop_duplicates()

    return ticker_link_df


def build_iclink(
    ibes_cusip_df: pd.DataFrame,
    crsp_cusip_df: pd.DataFrame,
    ibes_ticker_df: pd.DataFrame,
    crsp_ticker_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ICLINK table linking IBES tickers to CRSP permnos.
    """

    ibes_cusip_prepared_df = _prepare_ibes_cusip_data(
        ibes_cusip_df=ibes_cusip_df,
    )

    crsp_cusip_prepared_df = _prepare_crsp_cusip_data(
        crsp_cusip_df=crsp_cusip_df,
    )

    cusip_link_df, name_ratio_cutoff = _build_cusip_link(
        ibes_prepared_df=ibes_cusip_prepared_df,
        crsp_prepared_df=crsp_cusip_prepared_df,
    )

    unmatched_tickers_df = pd.merge(
        ibes_cusip_prepared_df[["ticker"]],
        cusip_link_df[["permno", "ticker"]],
        how="left",
        on="ticker",
    )

    unmatched_tickers_df = (
        unmatched_tickers_df.loc[unmatched_tickers_df["permno"].isna()]
        .drop(columns=["permno"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    ibes_ticker_prepared_df = _prepare_ibes_ticker_data(
        ibes_ticker_df=ibes_ticker_df,
        unmatched_tickers_df=unmatched_tickers_df,
    )

    crsp_ticker_prepared_df = _prepare_crsp_ticker_data(
        crsp_ticker_df=crsp_ticker_df,
    )

    ticker_link_df = _build_ticker_link(
        ibes_prepared_df=ibes_ticker_prepared_df,
        crsp_prepared_df=crsp_ticker_prepared_df,
        cusip_link_df=cusip_link_df,
        name_ratio_cutoff=name_ratio_cutoff,
    )

    iclink_df = pd.concat(
        [
            cusip_link_df,
            ticker_link_df,
        ],
        axis=0,
        ignore_index=True,
    )

    iclink_df["permno"] = iclink_df["permno"].astype(int)
    iclink_df["score"] = iclink_df["score"].astype(int)

    iclink_df = iclink_df.sort_values(
        by=[
            "ticker",
            "score",
            "permno",
        ]
    ).reset_index(drop=True)

    return iclink_df


def build_iclink_from_wrds(
    conn: Any,
) -> pd.DataFrame:
    """
    Load source data from WRDS and build ICLINK.
    """

    (
        ibes_cusip_df,
        crsp_cusip_df,
        ibes_ticker_df,
        crsp_ticker_df,
    ) = load_iclink_source_data(conn=conn)

    iclink_df: pd.DataFrame = build_iclink(
        ibes_cusip_df=ibes_cusip_df,
        crsp_cusip_df=crsp_cusip_df,
        ibes_ticker_df=ibes_ticker_df,
        crsp_ticker_df=crsp_ticker_df,
    )

    return iclink_df