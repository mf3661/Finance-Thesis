from __future__ import annotations

import numpy as np
import pandas as pd


def add_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add annual accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "act",
        "lct",
        "at",
        "che",
        "dlc",
        "txp",
        "oancf",
        "dp",
        "ib",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for acc: {missing_cols}")

    output_df["act_l1"] = output_df.groupby("permno")["act"].shift(1)
    output_df["lct_l1"] = output_df.groupby("permno")["lct"].shift(1)
    output_df["at_l1"] = output_df.groupby("permno")["at"].shift(1)

    output_df["che_l1"] = output_df.groupby("permno")["che"].shift(1)
    output_df["dlc_l1"] = output_df.groupby("permno")["dlc"].shift(1)
    output_df["txp_l1"] = output_df.groupby("permno")["txp"].shift(1)

    avg_at = (output_df["at"] + output_df["at_l1"]) / 2

    balance_sheet_acc = (
        (output_df["act"] - output_df["act_l1"])
        - (output_df["che"] - output_df["che_l1"])
        - (output_df["lct"] - output_df["lct_l1"])
        + (output_df["dlc"] - output_df["dlc_l1"])
        + (output_df["txp"] - output_df["txp_l1"]).fillna(0)
        - output_df["dp"]
    ) / avg_at

    cash_flow_acc = (
        output_df["ib"] - output_df["oancf"]
    ) / avg_at

    output_df["acc"] = np.where(
        output_df["oancf"].isna(),
        balance_sheet_acc,
        cash_flow_acc,
    )

    return output_df