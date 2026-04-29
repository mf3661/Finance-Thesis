from __future__ import annotations

import numpy as np
import pandas as pd


def add_tb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tax burden variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "fyear",
        "txfo",
        "txfed",
        "txt",
        "txdi",
        "ib",
        "datadate",
        "ffi49",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for tb: {missing_cols}")

    condition_list = [
        output_df["fyear"] <= 1978,
        (1979 <= output_df["fyear"]) & (output_df["fyear"] <= 1986),
        output_df["fyear"] == 1987,
        (1988 <= output_df["fyear"]) & (output_df["fyear"] <= 1992),
        1993 <= output_df["fyear"],
    ]

    choice_list = [
        0.48,
        0.46,
        0.40,
        0.34,
        0.35,
    ]

    output_df["tr"] = np.select(
        condition_list,
        choice_list,
        default=np.nan,
    )

    output_df["tb_1"] = (
        (output_df["txfo"] + output_df["txfed"]) / output_df["tr"]
    ) / output_df["ib"]

    output_df["tb_1"] = np.where(
        output_df["txfo"].isna() | output_df["txfed"].isna(),
        ((output_df["txt"] - output_df["txdi"]) / output_df["tr"])
        / output_df["ib"],
        output_df["tb_1"],
    )

    positive_tax_condition = (
        ((output_df["txfo"] + output_df["txfed"] > 0)
         | (output_df["txt"] > output_df["txdi"]))
        & (output_df["ib"] <= 0)
    )

    output_df["tb_1"] = np.where(
        positive_tax_condition,
        1,
        output_df["tb_1"],
    )

    industry_temp_df = (
        output_df.groupby(["datadate", "ffi49"], as_index=False)["tb_1"]
        .mean()
        .rename(columns={"tb_1": "tb_1_ind"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["datadate", "ffi49"],
    ).reset_index(drop=True)

    output_df["tb"] = output_df["tb_1"] - output_df["tb_1_ind"]

    return output_df