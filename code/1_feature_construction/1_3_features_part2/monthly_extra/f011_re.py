from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


def load_re_source_data(
    conn: Any,
    start_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load IBES summary forecasts and CRSP monthly price data for RE.
    """

    date_filter = ""

    if start_date is not None:
        date_filter = f"and statpers >= '{start_date}'"

    ibes_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            ticker,
            statpers,
            meanest,
            fpedats,
            anndats_act,
            curr_act,
            fpi,
            medest
        from ibes.statsum_epsus
        where
            statpers < anndats_act
            and measure = 'EPS'
            and (fpedats - statpers) >= 0
            and curcode = 'USD'
            and fpi in ('1', '2')
            {date_filter}
        """
    )

    crsp_msf_df: pd.DataFrame = conn.raw_sql(
        """
        select
            permno,
            date,
            prc,
            cfacpr
        from crsp.msf
        """
    )

    return ibes_df, crsp_msf_df


def add_re(
    ibes_df: pd.DataFrame,
    crsp_msf_df: pd.DataFrame,
    iclink_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add analysts' earnings forecast revision variable.
    """

    ibes_input_df: pd.DataFrame = ibes_df.copy()
    crsp_input_df: pd.DataFrame = crsp_msf_df.copy()
    iclink_input_df: pd.DataFrame = iclink_df.copy()

    required_ibes_cols: list[str] = [
        "ticker",
        "statpers",
        "meanest",
        "fpedats",
        "anndats_act",
        "curr_act",
        "medest",
    ]

    required_crsp_cols: list[str] = [
        "permno",
        "date",
        "prc",
        "cfacpr",
    ]

    required_iclink_cols: list[str] = [
        "ticker",
        "permno",
    ]

    missing_ibes_cols: list[str] = [
        col for col in required_ibes_cols if col not in ibes_input_df.columns
    ]

    if missing_ibes_cols:
        raise KeyError(f"Missing required IBES columns for re: {missing_ibes_cols}")

    missing_crsp_cols: list[str] = [
        col for col in required_crsp_cols if col not in crsp_input_df.columns
    ]

    if missing_crsp_cols:
        raise KeyError(f"Missing required CRSP columns for re: {missing_crsp_cols}")

    missing_iclink_cols: list[str] = [
        col for col in required_iclink_cols if col not in iclink_input_df.columns
    ]

    if missing_iclink_cols:
        raise KeyError(
            f"Missing required ICLINK columns for re: {missing_iclink_cols}"
        )

    ibes_input_df = ibes_input_df[
        ibes_input_df["medest"].notna()
        & ibes_input_df["fpedats"].notna()
    ].copy()

    ibes_input_df = ibes_input_df[
        (ibes_input_df["curr_act"] == "USD")
        | (ibes_input_df["curr_act"].isna())
    ].copy()

    ibes_input_df["statpers"] = pd.to_datetime(ibes_input_df["statpers"])
    ibes_input_df["fpedats"] = pd.to_datetime(ibes_input_df["fpedats"])
    ibes_input_df["anndats_act"] = pd.to_datetime(ibes_input_df["anndats_act"])
    ibes_input_df["merge_date"] = ibes_input_df["statpers"] + MonthEnd(0)

    crsp_input_df["permno"] = crsp_input_df["permno"].astype(int)
    crsp_input_df["date"] = pd.to_datetime(crsp_input_df["date"])
    crsp_input_df["date"] = crsp_input_df["date"] + MonthEnd(0)

    crsp_input_df["merge_date"] = crsp_input_df["date"] + MonthEnd(1)

    numeric_crsp_cols: list[str] = [
        "prc",
        "cfacpr",
    ]

    for col in numeric_crsp_cols:
        crsp_input_df[col] = pd.to_numeric(
            crsp_input_df[col],
            errors="coerce",
        )

    iclink_input_df["permno"] = iclink_input_df["permno"].astype(int)

    ibes_iclink_df = pd.merge(
        ibes_input_df,
        iclink_input_df,
        how="left",
        on="ticker",
    )

    ibes_crsp_df = pd.merge(
        ibes_iclink_df,
        crsp_input_df,
        how="inner",
        on=[
            "permno",
            "merge_date",
        ],
    )

    ibes_crsp_df = (
        ibes_crsp_df.sort_values(
            by=[
                "ticker",
                "fpedats",
                "statpers",
            ]
        )
        .reset_index(drop=True)
    )

    same_forecast_group = (
        (ibes_crsp_df["ticker"] == ibes_crsp_df["ticker"].shift(1))
        & (ibes_crsp_df["permno"] == ibes_crsp_df["permno"].shift(1))
        & (ibes_crsp_df["fpedats"] == ibes_crsp_df["fpedats"].shift(1))
    )

    ibes_crsp_df["statpers_last_month"] = np.where(
        same_forecast_group,
        ibes_crsp_df["statpers"].shift(1).astype(str),
        np.nan,
    )

    ibes_crsp_df["meanest_last_month"] = np.where(
        same_forecast_group,
        ibes_crsp_df["meanest"].shift(1),
        np.nan,
    )

    ibes_crsp_df = (
        ibes_crsp_df.sort_values(
            by=[
                "ticker",
                "permno",
                "fpedats",
                "statpers",
            ]
        )
        .reset_index(drop=True)
    )

    ibes_crsp_df = ibes_crsp_df[
        ibes_crsp_df["statpers_last_month"].notna()
    ].copy()

    ibes_crsp_df["prc_adj"] = (
        ibes_crsp_df["prc"] / ibes_crsp_df["cfacpr"]
    )

    ibes_crsp_df = ibes_crsp_df[
        ibes_crsp_df["prc_adj"] > 0
    ].copy()

    ibes_crsp_df["monthly_revision"] = (
        ibes_crsp_df["meanest"]
        - ibes_crsp_df["meanest_last_month"]
    ) / ibes_crsp_df["prc_adj"]

    ibes_crsp_df["permno_fpedats"] = (
        ibes_crsp_df["permno"].astype(str)
        + "-"
        + ibes_crsp_df["fpedats"].astype(str)
    )

    ibes_crsp_df = ibes_crsp_df.drop_duplicates(
        [
            "permno_fpedats",
            "statpers",
        ]
    ).copy()

    ibes_crsp_df["count"] = (
        ibes_crsp_df.groupby("permno_fpedats")
        .cumcount()
        + 1
    )

    for lag in range(1, 7):
        ibes_crsp_df[f"monthly_revision_l{lag}"] = (
            ibes_crsp_df.groupby("permno")["monthly_revision"]
            .shift(lag)
        )

    condlist = [
        ibes_crsp_df["count"] == 4,
        ibes_crsp_df["count"] == 5,
        ibes_crsp_df["count"] == 6,
        ibes_crsp_df["count"] >= 7,
    ]

    choicelist = [
        (
            ibes_crsp_df["monthly_revision_l1"]
            + ibes_crsp_df["monthly_revision_l2"]
            + ibes_crsp_df["monthly_revision_l3"]
        )
        / 3,
        (
            ibes_crsp_df["monthly_revision_l1"]
            + ibes_crsp_df["monthly_revision_l2"]
            + ibes_crsp_df["monthly_revision_l3"]
            + ibes_crsp_df["monthly_revision_l4"]
        )
        / 4,
        (
            ibes_crsp_df["monthly_revision_l1"]
            + ibes_crsp_df["monthly_revision_l2"]
            + ibes_crsp_df["monthly_revision_l3"]
            + ibes_crsp_df["monthly_revision_l4"]
            + ibes_crsp_df["monthly_revision_l5"]
        )
        / 5,
        (
            ibes_crsp_df["monthly_revision_l1"]
            + ibes_crsp_df["monthly_revision_l2"]
            + ibes_crsp_df["monthly_revision_l3"]
            + ibes_crsp_df["monthly_revision_l4"]
            + ibes_crsp_df["monthly_revision_l5"]
            + ibes_crsp_df["monthly_revision_l6"]
        )
        / 6,
    ]

    ibes_crsp_df["re"] = np.select(
        condlist,
        choicelist,
        default=np.nan,
    )

    ibes_crsp_df = ibes_crsp_df[
        ibes_crsp_df["count"] >= 4
    ].copy()

    ibes_crsp_df = (
        ibes_crsp_df.sort_values(
            by=[
                "ticker",
                "statpers",
                "fpedats",
            ]
        )
        .drop_duplicates(
            [
                "ticker",
                "statpers",
            ]
        )
        .reset_index(drop=True)
    )

    output_df: pd.DataFrame = ibes_crsp_df[
        [
            "ticker",
            "statpers",
            "fpedats",
            "anndats_act",
            "curr_act",
            "permno",
            "re",
        ]
    ].copy()

    output_df = output_df.rename(
        columns={
            "statpers": "date",
        }
    )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["date"] = pd.to_datetime(output_df["date"])
    output_df["jdate"] = output_df["date"] + MonthEnd(0)

    output_df = output_df[
        [
            "permno",
            "date",
            "jdate",
            "ticker",
            "fpedats",
            "anndats_act",
            "curr_act",
            "re",
        ]
    ].copy()

    return output_df