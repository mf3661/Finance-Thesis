from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pandas.tseries.offsets import MonthEnd


CRSP_FILL_COLUMNS: list[str] = [
    "permno",
    "jdate",
    "ret_fill",
    "retx_fill",
    "retadj_fill",
    "me_fill",
    "shrcd_fill",
    "exchcd_fill",
]


def load_crsp_fill_source_data(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CRSP monthly data and monthly delisting returns for fill variables.
    """

    end_date_filter = ""

    if end_date is not None:
        end_date_filter = f"and a.date <= '{end_date}'"

    crsp_df: pd.DataFrame = conn.raw_sql(
        f"""
        select
            a.prc,
            a.ret,
            a.retx,
            a.shrout,
            a.vol,
            a.date,
            a.permno,
            a.permco,
            b.shrcd,
            b.exchcd
        from crsp.msf as a
        left join crsp.msenames as b
        on a.permno = b.permno
        and b.namedt <= a.date
        and a.date <= b.nameendt
        where a.date >= '{start_date}'
        {end_date_filter}
        and b.exchcd between 1 and 3
        """
    )

    dlret_df: pd.DataFrame = conn.raw_sql(
        """
        select
            permno,
            dlret,
            dlstdt
        from crsp.msedelist
        """
    )

    return crsp_df, dlret_df


def build_crsp_fill(
    crsp_df: pd.DataFrame,
    dlret_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly CRSP fill variables.

    Output columns:
        permno, jdate, ret_fill, retx_fill, retadj_fill,
        me_fill, shrcd_fill, exchcd_fill.
    """

    crsp_input_df: pd.DataFrame = crsp_df.copy()
    dlret_input_df: pd.DataFrame = dlret_df.copy()

    required_crsp_cols: list[str] = [
        "prc",
        "ret",
        "retx",
        "shrout",
        "date",
        "permno",
        "permco",
        "shrcd",
        "exchcd",
    ]

    missing_crsp_cols: list[str] = [
        col for col in required_crsp_cols if col not in crsp_input_df.columns
    ]

    if missing_crsp_cols:
        raise KeyError(
            f"Missing required CRSP columns for crsp fill: {missing_crsp_cols}"
        )

    required_dlret_cols: list[str] = [
        "permno",
        "dlret",
        "dlstdt",
    ]

    missing_dlret_cols: list[str] = [
        col for col in required_dlret_cols if col not in dlret_input_df.columns
    ]

    if missing_dlret_cols:
        raise KeyError(
            f"Missing required delisting return columns for crsp fill: {missing_dlret_cols}"
        )

    crsp_input_df = crsp_input_df.dropna(
        subset=[
            "ret",
            "retx",
            "prc",
        ]
    ).reset_index(drop=True)

    crsp_input_df[["permco", "permno"]] = crsp_input_df[
        ["permco", "permno"]
    ].astype(int)

    crsp_input_df["date"] = pd.to_datetime(crsp_input_df["date"])
    crsp_input_df["jdate"] = crsp_input_df["date"] + MonthEnd(0)

    numeric_cols: list[str] = [
        "prc",
        "ret",
        "retx",
        "shrout",
        "shrcd",
        "exchcd",
    ]

    for col in numeric_cols:
        crsp_input_df[col] = pd.to_numeric(
            crsp_input_df[col],
            errors="coerce",
        )

    crsp_input_df = crsp_input_df.dropna(
        subset=[
            "prc",
        ]
    ).reset_index(drop=True)

    crsp_input_df["me"] = (
        crsp_input_df["prc"].abs()
        * crsp_input_df["shrout"]
    )

    crsp_summe_df = (
        crsp_input_df.groupby(["jdate", "permco"], as_index=False)["me"]
        .sum()
    )

    crsp_maxme_df = (
        crsp_input_df.groupby(["jdate", "permco"], as_index=False)["me"]
        .max()
    )

    crsp_largest_security_df = pd.merge(
        crsp_input_df,
        crsp_maxme_df,
        how="inner",
        on=[
            "jdate",
            "permco",
            "me",
        ],
    )

    crsp_largest_security_df = crsp_largest_security_df.drop(
        columns=[
            "me",
        ]
    )

    crsp_aligned_df = pd.merge(
        crsp_largest_security_df,
        crsp_summe_df,
        how="inner",
        on=[
            "jdate",
            "permco",
        ],
    )

    crsp_aligned_df = (
        crsp_aligned_df.sort_values(by=["permno", "jdate"])
        .drop_duplicates(subset=["permno", "jdate"], keep="last")
        .reset_index(drop=True)
    )

    crsp_aligned_df["me"] = crsp_aligned_df["me"] / 1000

    crsp_aligned_df = crsp_aligned_df.sort_values(
        by=[
            "permno",
            "date",
        ]
    ).reset_index(drop=True)

    dlret_input_df["permno"] = dlret_input_df["permno"].astype(int)
    dlret_input_df["dlstdt"] = pd.to_datetime(dlret_input_df["dlstdt"])
    dlret_input_df["jdate"] = dlret_input_df["dlstdt"] + MonthEnd(0)
    dlret_input_df["dlret"] = pd.to_numeric(
        dlret_input_df["dlret"],
        errors="coerce",
    )

    dlret_input_df = dlret_input_df[
        [
            "permno",
            "jdate",
            "dlret",
        ]
    ].copy()

    crsp_aligned_df = pd.merge(
        crsp_aligned_df,
        dlret_input_df,
        how="left",
        on=[
            "permno",
            "jdate",
        ],
    ).reset_index(drop=True)

    crsp_aligned_df["dlret"] = crsp_aligned_df["dlret"].fillna(0)
    crsp_aligned_df["ret"] = crsp_aligned_df["ret"].fillna(0)

    crsp_aligned_df["retadj"] = (
        (1 + crsp_aligned_df["ret"])
        * (1 + crsp_aligned_df["dlret"])
        - 1
    )

    output_df: pd.DataFrame = crsp_aligned_df[
        [
            "permno",
            "jdate",
            "ret",
            "retx",
            "retadj",
            "me",
            "shrcd",
            "exchcd",
        ]
    ].copy()

    output_df = output_df.rename(
        columns={
            "ret": "ret_fill",
            "retx": "retx_fill",
            "retadj": "retadj_fill",
            "me": "me_fill",
            "shrcd": "shrcd_fill",
            "exchcd": "exchcd_fill",
        }
    )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    output_df = output_df[CRSP_FILL_COLUMNS].copy()

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .drop_duplicates(subset=["permno", "jdate"], keep="last")
        .reset_index(drop=True)
    )

    return output_df


def build_crsp_fill_from_wrds(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load CRSP data from WRDS and build fill variables.
    """

    crsp_df, dlret_df = load_crsp_fill_source_data(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )

    crsp_fill_df: pd.DataFrame = build_crsp_fill(
        crsp_df=crsp_df,
        dlret_df=dlret_df,
    )

    return crsp_fill_df


def apply_crsp_fill(
    df: pd.DataFrame,
    crsp_fill_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing ret, retx, retadj, me, exchcd, and shrcd using CRSP fill data.
    """

    output_df: pd.DataFrame = df.copy()
    fill_df: pd.DataFrame = crsp_fill_df.copy()

    required_cols: list[str] = [
        "permno",
        "jdate",
        "ret",
        "retx",
        "retadj",
        "me",
        "exchcd",
        "shrcd",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns before applying crsp fill: {missing_cols}"
        )

    missing_fill_cols: list[str] = [
        col for col in CRSP_FILL_COLUMNS if col not in fill_df.columns
    ]

    if missing_fill_cols:
        raise KeyError(
            f"Missing required crsp fill columns: {missing_fill_cols}"
        )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    fill_df["permno"] = fill_df["permno"].astype(int)
    fill_df["jdate"] = pd.to_datetime(fill_df["jdate"])

    output_df = pd.merge(
        output_df,
        fill_df,
        how="left",
        on=[
            "permno",
            "jdate",
        ],
    )

    output_df["ret"] = np.where(
        output_df["ret"].isna(),
        output_df["ret_fill"],
        output_df["ret"],
    )

    output_df["retx"] = np.where(
        output_df["retx"].isna(),
        output_df["retx_fill"],
        output_df["retx"],
    )

    output_df["retadj"] = np.where(
        output_df["retadj"].isna(),
        output_df["retadj_fill"],
        output_df["retadj"],
    )

    output_df["me"] = np.where(
        output_df["me"].isna(),
        output_df["me_fill"],
        output_df["me"],
    )

    output_df["exchcd"] = np.where(
        output_df["exchcd"].isna(),
        output_df["exchcd_fill"],
        output_df["exchcd"],
    )

    output_df["shrcd"] = np.where(
        output_df["shrcd"].isna(),
        output_df["shrcd_fill"],
        output_df["shrcd"],
    )

    output_df = output_df.drop(
        columns=[
            col
            for col in CRSP_FILL_COLUMNS
            if col not in ["permno", "jdate"] and col in output_df.columns
        ]
    )

    output_df = output_df.dropna(
        subset=[
            "permno",
            "jdate",
            "ret",
            "retx",
            "retadj",
        ]
    )

    output_df = output_df[
        output_df["exchcd"].isin([1, 2, 3])
        & output_df["shrcd"].isin([10, 11])
    ].reset_index(drop=True)

    return output_df


def save_crsp_fill(
    crsp_fill_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save CRSP fill dataframe.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    feather.write_feather(
        crsp_fill_df.reset_index(drop=True),
        output_path,
    )

    print(f"SAVED: {output_path}")