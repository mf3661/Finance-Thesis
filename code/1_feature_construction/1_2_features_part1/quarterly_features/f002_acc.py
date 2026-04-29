from __future__ import annotations

import numpy as np
import pandas as pd


def add_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "actq",
        "lctq",
        "cheq",
        "dlcq",
        "txpq",
        "dpq",
        "atq",
        "atq_l4",
        "oancfy",
        "ibq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for quarterly acc: {missing_cols}")

    output_df["actq_l4"] = output_df.groupby("permno")["actq"].shift(4)
    output_df["lctq_l4"] = output_df.groupby("permno")["lctq"].shift(4)
    output_df["cheq_l4"] = output_df.groupby("permno")["cheq"].shift(4)
    output_df["dlcq_l4"] = output_df.groupby("permno")["dlcq"].shift(4)
    output_df["txpq_l4"] = output_df.groupby("permno")["txpq"].shift(4)

    avg_atq = (output_df["atq"] + output_df["atq_l4"]) / 2

    balance_sheet_acc = (
        (output_df["actq"] - output_df["actq_l4"])
        - (output_df["cheq"] - output_df["cheq_l4"])
        - (output_df["lctq"] - output_df["lctq_l4"])
        + (output_df["dlcq"] - output_df["dlcq_l4"])
        + (output_df["txpq"] - output_df["txpq_l4"]).fillna(0)
        - output_df["dpq"]
    ) / avg_atq

    cash_flow_acc = (
        output_df["ibq"] - output_df["oancfy"]
    ) / avg_atq

    output_df["acc"] = np.where(
        output_df["oancfy"].isna(),
        balance_sheet_acc,
        cash_flow_acc,
    )

    return output_df