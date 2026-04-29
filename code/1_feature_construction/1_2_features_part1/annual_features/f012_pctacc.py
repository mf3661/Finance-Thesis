from __future__ import annotations

import numpy as np
import pandas as pd


def add_pctacc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ib",
        "oancf",
        "act",
        "act_l1",
        "che",
        "che_l1",
        "lct",
        "lct_l1",
        "dlc",
        "dlc_l1",
        "txp",
        "txp_l1",
        "dp",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pctacc: {missing_cols}")

    balance_sheet_accrual_numerator = (
        (output_df["act"] - output_df["act_l1"])
        - (output_df["che"] - output_df["che_l1"])
    ) - (
        (output_df["lct"] - output_df["lct_l1"])
        - output_df["dlc"]
        - output_df["dlc_l1"]
        - (
            (output_df["txp"] - output_df["txp_l1"]).fillna(0)
            - output_df["dp"]
        )
    )

    condition_list = [
        output_df["ib"] == 0,
        output_df["oancf"].isna(),
        output_df["oancf"].isna() & (output_df["ib"] == 0),
    ]

    choice_list = [
        (output_df["ib"] - output_df["oancf"]) / 0.01,
        balance_sheet_accrual_numerator / output_df["ib"].abs(),
        balance_sheet_accrual_numerator / 0.01,
    ]

    output_df["pctacc"] = np.select(
        condition_list,
        choice_list,
        default=(
            output_df["ib"] - output_df["oancf"]
        ) / output_df["ib"].abs(),
    )

    return output_df