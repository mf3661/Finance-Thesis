from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_compustat_annual(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load raw Compustat annual data from WRDS.
    """

    print("Loading Compustat Annual data...")

    compustat_df: pd.DataFrame = conn.raw_sql(
        f"""
        /* header info */
        select c.gvkey, f.cusip, f.datadate, f.fyear, c.cik,
               c.naics, c.sic, substr(c.sic, 1, 2) as sic2,

        /* income statement */
        f.cogs, f.dp, f.ebit, f.ebitda, f.ib, f.ni, f.nopi, f.pi,
        f.revt, f.sale, f.spi, f.txfed, f.txfo, f.txt, f.txp,
        f.xad, f.xint, f.xrd, f.xsga,

        /* cash flow statement and other items */
        f.capx, f.dvt, f.gdwlia, f.gdwlip, f.gwo, f.ivao,
        f.mib, f.ob, f.oancf, f.oiadp, f.xacc, f.xpp,

        /* assets */
        f.aco, f.act, f.ao, f.at, f.che, f.fatb, f.fatl,
        f.gdwl, f.intan, f.invt, f.ppegt, f.ppent, f.rect,

        /* liabilities */
        f.ap, f.dcpstk, f.dcvt, f.dlc, f.dltt, f.dm, f.drc,
        f.drlt, f.lco, f.lct, f.lo, f.lt, f.pstk, f.txdi,

        /* equity and other items */
        f.ajex, f.ceq, f.conm, f.csho, f.dpc, f.emp, f.np,
        f.pstkl, f.pstkrv, f.scstkc, f.seq, f.txdc, f.txditc,

        /* market */
        abs(f.prcc_f) as prcc_f

        from compustat_df.funda as f
        left join compustat_df.company as c
        on f.gvkey = c.gvkey

        where f.indfmt = 'INDL'
        and f.datafmt = 'STD'
        and f.popsrc = 'D'
        and f.consol = 'C'
        and f.datadate >= '{start_date}'
        and f.datadate <= '{end_date}'
        """
    )

    return compustat_df


def clean_compustat_annual(
    compustat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean raw Compustat annual data.
    """

    # Work on a copy to avoid modifying the raw dataframe in place.
    df: pd.DataFrame = compustat_df.copy()

    # Convert Compustat report date to datetime format.
    # This is needed later for sorting, lag construction, and CCM link-date filtering.
    df["datadate"] = pd.to_datetime(df["datadate"])

    # Sort observations within each firm by fiscal report date.
    # This makes the panel order stable before later lag construction.
    # Exact duplicates are removed to avoid double-counting the same firm-date record.
    df = (
        df.sort_values(by=["gvkey", "datadate"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Shares outstanding equal to zero is not economically meaningful for market equity.
    # Treat it as missing so that Compustat market equity does not become artificially zero.
    df["csho"] = np.where(df["csho"] == 0, np.nan, df["csho"])

    # Compustat market equity. This is kept as a source-level helper.
    # In the linked base table, the main market equity will usually come from CRSP.
    df["mve_f"] = df["csho"] * df["prcc_f"]

    # Combine current and long-term deferred revenue into one deferred-revenue measure.
    # Different firms may report only the current part, only the long-term part, or both.
    dr_conditions = [
        df["drc"].notna() & df["drlt"].notna(),
        df["drc"].notna() & df["drlt"].isna(),
        df["drlt"].notna() & df["drc"].isna(),
    ]

    dr_choices = [
        df["drc"] + df["drlt"],
        df["drc"],
        df["drlt"],
    ]

    df["dr"] = np.select(dr_conditions, dr_choices, default=np.nan)

    # Construct a unified convertible-debt variable.
    # When dcvt is missing, infer convertible debt from convertible preferred stock
    # when the relevant preferred-stock fields are available.
    dc_conditions = [
        df["dcvt"].isna()
        & df["dcpstk"].notna()
        & df["pstk"].notna()
        & (df["dcpstk"] > df["pstk"]),
        df["dcvt"].isna()
        & df["dcpstk"].notna()
        & df["pstk"].isna(),
    ]

    dc_choices = [
        df["dcpstk"] - df["pstk"],
        df["dcpstk"],
    ]

    df["dc"] = np.select(dc_conditions, dc_choices, default=np.nan)

    # If dcvt is available, use it directly; otherwise use the inferred value above.
    df["dc"] = np.where(df["dc"].isna(), df["dcvt"], df["dc"])

    # Interest expense is often missing when it is effectively unavailable or zero.
    # For profitability formulas that subtract interest expense, use a zero-filled helper.
    # Keep the original xint unchanged.
    df["xint0"] = np.where(df["xint"].isna(), 0, df["xint"])

    # Common equity equal to zero is not useful for ratios that divide by equity.
    # Treat zero as missing to avoid invalid or extreme accounting ratios later.
    df["ceq"] = np.where(df["ceq"] == 0, np.nan, df["ceq"])

    # Total assets equal to zero is not economically meaningful and would create
    # invalid denominators for many characteristics, such as roa, agr, acc, and invest.
    df["at"] = np.where(df["at"] == 0, np.nan, df["at"])

    # Total assets are required for most accounting characteristics.
    # Observations without valid assets are removed from the Compustat annual base.
    df = df.dropna(subset=["at"]).reset_index(drop=True)

    print("Finished cleaning Compustat Annual data.")

    return df


def load_compustat_quarterly(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load raw Compustat quarterly data from WRDS.
    """

    print("Loading Compustat Quarterly data...")

    compustat_df: pd.DataFrame = conn.raw_sql(
        f"""
        /* header info */
        select c.gvkey, f.cusip, f.datadate, f.fyearq, f.fqtr,
               f.rdq, c.sic, substr(c.sic, 1, 2) as sic2,

        /* income statement */
        f.cogsq, f.cogsy, f.ibq, f.revtq, f.revty,
        f.saleq, f.saley, f.txtq, f.xsgaq,

        /* balance sheet items */
        f.actq, f.atq, f.cheq, f.dlcq, f.drcq, f.drltq,
        f.lctq, f.ppegtq, f.ppentq, f.txpq, f.xaccq,

        /* other items */
        f.ceqq, f.conm, f.gdwlq, f.intanq, f.ivaoq, f.ltq,
        f.mibq, f.oiadpq, f.pstkq, f.pstkrq, f.seqq,

        /* additional accounting items */
        f.acoq, f.ajexq, f.aoq, f.apq, f.cshoq, f.dlttq,
        f.dpq, f.invtq, f.lcoq, f.loq, f.niq, f.npq,
        f.oancfy, f.rectq, f.scstkcy, f.txditcq, f.xintq,
        f.xrdq, f.xrdy,

        /* market */
        abs(f.prccq) as prccq,
        abs(f.prccq) * f.cshoq as mveq_f

        from compustat_df.fundq as f
        left join compustat_df.company as c
        on f.gvkey = c.gvkey

        where f.indfmt = 'INDL'
        and f.datafmt = 'STD'
        and f.popsrc = 'D'
        and f.consol = 'C'
        and f.datadate >= '{start_date}'
        and f.datadate <= '{end_date}'
        """
    )

    return compustat_df


def clean_compustat_quarterly(
    compustat_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean raw Compustat quarterly data.
    """

    # Work on a copy to avoid modifying the raw dataframe in place.
    df: pd.DataFrame = compustat_df.copy()

    # Convert report date before sorting and later date operations.
    # Quarterly data will later use datadate, rdq, and jdate to determine data availability.
    df["datadate"] = pd.to_datetime(df["datadate"])

    # Quarterly earnings are required for many quarterly accounting signals.
    # The original code drops observations with missing ibq before further processing.
    df = df.dropna(subset=["ibq"]).reset_index(drop=True)

    # Sort observations within each firm by quarterly report date.
    # This gives a stable panel order for later quarterly lags, such as shift(1) and shift(4).
    # Exact duplicates are removed to avoid duplicate firm-quarter records.
    df = (
        df.sort_values(by=["gvkey", "datadate"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Quarterly shares outstanding equal to zero is not economically meaningful.
    # Treat it as missing so that mveq_f or other market-equity helpers are not forced to zero.
    df["cshoq"] = np.where(df["cshoq"] == 0, np.nan, df["cshoq"])

    # Quarterly common equity equal to zero is not useful for equity-based ratios.
    # Treat zero as missing to avoid invalid denominators later.
    df["ceqq"] = np.where(df["ceqq"] == 0, np.nan, df["ceqq"])

    # Quarterly total assets equal to zero is not economically meaningful.
    # Many quarterly characteristics use atq or lagged atq as denominators,
    # so zero assets would create invalid ratios. We convert zero to NaN first.
    df["atq"] = np.where(df["atq"] == 0, np.nan, df["atq"])

    # Total assets are required for most quarterly accounting characteristics.
    # Observations without valid atq are removed from the quarterly base.
    df = df.dropna(subset=["atq"]).reset_index(drop=True)

    print("Finished cleaning Compustat Quarterly data.")

    return df


def load_and_clean_compustat_annual(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load and clean Compustat annual data.
    """

    raw_df: pd.DataFrame = load_compustat_annual(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )
    clean_df: pd.DataFrame = clean_compustat_annual(raw_df)

    return clean_df


def load_and_clean_compustat_quarterly(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str = "03/31/2025",
) -> pd.DataFrame:
    """
    Load and clean Compustat quarterly data.
    """

    raw_df: pd.DataFrame = load_compustat_quarterly(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )
    clean_df: pd.DataFrame = clean_compustat_quarterly(raw_df)

    return clean_df