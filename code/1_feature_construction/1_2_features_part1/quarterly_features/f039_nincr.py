from __future__ import annotations

import numpy as np
import pandas as pd


def add_nincr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly number of consecutive earnings increases variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ibq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly nincr: {missing_cols}"
        )

    output_df["ibq_l1"] = output_df.groupby("permno")["ibq"].shift(1)
    output_df["ibq_l2"] = output_df.groupby("permno")["ibq"].shift(2)
    output_df["ibq_l3"] = output_df.groupby("permno")["ibq"].shift(3)
    output_df["ibq_l4"] = output_df.groupby("permno")["ibq"].shift(4)
    output_df["ibq_l5"] = output_df.groupby("permno")["ibq"].shift(5)
    output_df["ibq_l6"] = output_df.groupby("permno")["ibq"].shift(6)
    output_df["ibq_l7"] = output_df.groupby("permno")["ibq"].shift(7)
    output_df["ibq_l8"] = output_df.groupby("permno")["ibq"].shift(8)

    output_df["nincr_temp1"] = np.where(
        output_df["ibq"] > output_df["ibq_l1"],
        1,
        0,
    )

    output_df["nincr_temp2"] = np.where(
        output_df["ibq_l1"] > output_df["ibq_l2"],
        1,
        0,
    )

    output_df["nincr_temp3"] = np.where(
        output_df["ibq_l2"] > output_df["ibq_l3"],
        1,
        0,
    )

    output_df["nincr_temp4"] = np.where(
        output_df["ibq_l3"] > output_df["ibq_l4"],
        1,
        0,
    )

    output_df["nincr_temp5"] = np.where(
        output_df["ibq_l4"] > output_df["ibq_l5"],
        1,
        0,
    )

    output_df["nincr_temp6"] = np.where(
        output_df["ibq_l5"] > output_df["ibq_l6"],
        1,
        0,
    )

    output_df["nincr_temp7"] = np.where(
        output_df["ibq_l6"] > output_df["ibq_l7"],
        1,
        0,
    )

    output_df["nincr_temp8"] = np.where(
        output_df["ibq_l7"] > output_df["ibq_l8"],
        1,
        0,
    )

    output_df["nincr"] = (
        output_df["nincr_temp1"]
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
            * output_df["nincr_temp4"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
            * output_df["nincr_temp4"]
            * output_df["nincr_temp5"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
            * output_df["nincr_temp4"]
            * output_df["nincr_temp5"]
            * output_df["nincr_temp6"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
            * output_df["nincr_temp4"]
            * output_df["nincr_temp5"]
            * output_df["nincr_temp6"]
            * output_df["nincr_temp7"]
        )
        + (
            output_df["nincr_temp1"]
            * output_df["nincr_temp2"]
            * output_df["nincr_temp3"]
            * output_df["nincr_temp4"]
            * output_df["nincr_temp5"]
            * output_df["nincr_temp6"]
            * output_df["nincr_temp7"]
            * output_df["nincr_temp8"]
        )
    )

    output_df = output_df.drop(
        [
            "ibq_l1",
            "ibq_l2",
            "ibq_l3",
            "ibq_l4",
            "ibq_l5",
            "ibq_l6",
            "ibq_l7",
            "ibq_l8",
            "nincr_temp1",
            "nincr_temp2",
            "nincr_temp3",
            "nincr_temp4",
            "nincr_temp5",
            "nincr_temp6",
            "nincr_temp7",
            "nincr_temp8",
        ],
        axis=1,
    )

    return output_df