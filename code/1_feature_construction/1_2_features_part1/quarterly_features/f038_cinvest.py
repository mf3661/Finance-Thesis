from __future__ import annotations

import numpy as np
import pandas as pd


def add_cinvest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly abnormal corporate investment variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ppentq",
        "saleq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly cinvest: {missing_cols}"
        )

    output_df["ppentq_l1"] = output_df.groupby("permno")["ppentq"].shift(1)
    output_df["ppentq_l2"] = output_df.groupby("permno")["ppentq"].shift(2)
    output_df["ppentq_l3"] = output_df.groupby("permno")["ppentq"].shift(3)
    output_df["ppentq_l4"] = output_df.groupby("permno")["ppentq"].shift(4)

    output_df["saleq_l1"] = output_df.groupby("permno")["saleq"].shift(1)
    output_df["saleq_l2"] = output_df.groupby("permno")["saleq"].shift(2)
    output_df["saleq_l3"] = output_df.groupby("permno")["saleq"].shift(3)

    output_df["c_temp1"] = (
        output_df["ppentq_l1"] - output_df["ppentq_l2"]
    ) / output_df["saleq_l1"]

    output_df["c_temp2"] = (
        output_df["ppentq_l2"] - output_df["ppentq_l3"]
    ) / output_df["saleq_l2"]

    output_df["c_temp3"] = (
        output_df["ppentq_l3"] - output_df["ppentq_l4"]
    ) / output_df["saleq_l3"]

    output_df["cinvest"] = (
        (output_df["ppentq"] - output_df["ppentq_l1"]) / output_df["saleq"]
    ) - output_df[["c_temp1", "c_temp2", "c_temp3"]].mean(axis=1)

    output_df["c_temp1"] = (
        output_df["ppentq_l1"] - output_df["ppentq_l2"]
    ) / 0.01

    output_df["c_temp2"] = (
        output_df["ppentq_l2"] - output_df["ppentq_l3"]
    ) / 0.01

    output_df["c_temp3"] = (
        output_df["ppentq_l3"] - output_df["ppentq_l4"]
    ) / 0.01

    output_df["cinvest"] = np.where(
        output_df["saleq"] <= 0,
        (
            (output_df["ppentq"] - output_df["ppentq_l1"]) / 0.01
        )
        - output_df[["c_temp1", "c_temp2", "c_temp3"]].mean(axis=1),
        output_df["cinvest"],
    )

    output_df = output_df.drop(
        ["c_temp1", "c_temp2", "c_temp3"],
        axis=1,
    )

    return output_df