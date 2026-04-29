from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


ACCOUNTING_MARKET_COLUMNS: list[str] = [
    "date",
    "ret",
    "retx",
    "retadj",
    "me",
    "prc",
    "shrout",
]


def populate_to_monthly(
    market_df: pd.DataFrame,
    accounting_df: pd.DataFrame,
    exclude_market_cols: Sequence[str] | None = ("dy",),
) -> pd.DataFrame:
    """
    Populate accounting features to monthly market panel.
    """

    market_input_df: pd.DataFrame = market_df.copy()
    accounting_input_df: pd.DataFrame = accounting_df.copy()

    required_market_cols: list[str] = [
        "permno",
        "date",
        "jdate",
    ]

    required_accounting_cols: list[str] = [
        "permno",
        "jdate",
        "datadate",
        "exchcd",
        "shrcd",
    ]

    missing_market_cols: list[str] = [
        col for col in required_market_cols if col not in market_input_df.columns
    ]

    if missing_market_cols:
        raise KeyError(
            f"Missing required market columns for monthly population: {missing_market_cols}"
        )

    missing_accounting_cols: list[str] = [
        col for col in required_accounting_cols if col not in accounting_input_df.columns
    ]

    if missing_accounting_cols:
        raise KeyError(
            f"Missing required accounting columns for monthly population: {missing_accounting_cols}"
        )

    market_input_df["permno"] = market_input_df["permno"].astype(int)
    accounting_input_df["permno"] = accounting_input_df["permno"].astype(int)

    market_input_df["date"] = pd.to_datetime(market_input_df["date"])
    market_input_df["jdate"] = pd.to_datetime(market_input_df["jdate"])

    accounting_input_df["jdate"] = pd.to_datetime(accounting_input_df["jdate"])
    accounting_input_df["datadate"] = pd.to_datetime(accounting_input_df["datadate"])

    # Accounting base/features already contain stale market columns from the
    # accounting release month. Drop them so monthly CRSP market columns overwrite them.
    drop_cols: list[str] = [
        col for col in ACCOUNTING_MARKET_COLUMNS if col in accounting_input_df.columns
    ]

    accounting_input_df = accounting_input_df.drop(
        columns=drop_cols,
    )

    # dy is excluded by default because annual dy is accounting-based in our pipeline,
    # and quarterly final output does not use dy.
    if exclude_market_cols is not None:
        market_input_df = market_input_df.drop(
            columns=[col for col in exclude_market_cols if col in market_input_df.columns],
        )

    output_df: pd.DataFrame = pd.merge(
        market_input_df,
        accounting_input_df,
        how="left",
        on=["permno", "jdate"],
    ).reset_index(drop=True)

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .reset_index(drop=True)
    )

    # First carry the latest accounting report date forward within each stock.
    output_df["datadate"] = output_df.groupby("permno")["datadate"].ffill()

    # Then carry all accounting characteristics forward within the same report period.
    fill_cols: list[str] = [
        col for col in output_df.columns if col not in ["permno", "datadate"]
    ]

    output_df[fill_cols] = (
        output_df.groupby(["permno", "datadate"], dropna=False)[fill_cols]
        .ffill()
    )

    output_df = output_df[
        output_df["exchcd"].isin([1, 2, 3])
        & output_df["shrcd"].isin([10, 11])
    ].reset_index(drop=True)

    return output_df