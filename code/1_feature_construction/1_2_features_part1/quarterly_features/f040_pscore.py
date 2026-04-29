from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_ttm4(column_name: str, df: pd.DataFrame) -> pd.Series:
    """
    Compute trailing-four-quarter sum.
    """

    return (
        df[column_name]
        + df.groupby("permno")[column_name].shift(1)
        + df.groupby("permno")[column_name].shift(2)
        + df.groupby("permno")[column_name].shift(3)
    )


def add_pscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly performance score variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "niq",
        "oancfy",
        "atq",
        "atq_l4",
        "dlttq",
        "actq",
        "lctq",
        "actq_l4",
        "lctq_l4",
        "cogsq4",
        "saleq4",
        "saleq4_l4",
        "scstkcy",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly pscore: {missing_cols}"
        )

    output_df["niq4"] = _compute_ttm4("niq", output_df)
    output_df["niq4_l4"] = output_df.groupby("permno")["niq4"].shift(4)
    output_df["dlttq_l4"] = output_df.groupby("permno")["dlttq"].shift(4)
    output_df["cogsq4_l4"] = output_df.groupby("permno")["cogsq4"].shift(4)

    output_df["p_temp1"] = np.where(
        output_df["niq4"] > 0,
        1,
        0,
    )

    output_df["p_temp2"] = np.where(
        output_df["oancfy"] > 0,
        1,
        0,
    )

    output_df["p_temp3"] = np.where(
        output_df["niq4"] / output_df["atq"]
        > output_df["niq4_l4"] / output_df["atq_l4"],
        1,
        0,
    )

    output_df["p_temp4"] = np.where(
        output_df["oancfy"] > output_df["niq4"],
        1,
        0,
    )

    output_df["p_temp5"] = np.where(
        output_df["dlttq"] / output_df["atq"]
        < output_df["dlttq_l4"] / output_df["atq_l4"],
        1,
        0,
    )

    output_df["p_temp6"] = np.where(
        output_df["actq"] / output_df["lctq"]
        > output_df["actq_l4"] / output_df["lctq_l4"],
        1,
        0,
    )

    gross_margin = (
        output_df["saleq4"] - output_df["cogsq4"]
    ) / output_df["saleq4"]

    gross_margin_l4 = (
        output_df["saleq4_l4"] - output_df["cogsq4_l4"]
    ) / output_df["saleq4_l4"]

    output_df["p_temp7"] = np.where(
        gross_margin > gross_margin_l4,
        1,
        0,
    )

    output_df["p_temp8"] = np.where(
        output_df["saleq4"] / output_df["atq"]
        > output_df["saleq4_l4"] / output_df["atq_l4"],
        1,
        0,
    )

    output_df["p_temp9"] = np.where(
        output_df["scstkcy"] == 0,
        1,
        0,
    )

    output_df["pscore"] = (
        output_df["p_temp1"]
        + output_df["p_temp2"]
        + output_df["p_temp3"]
        + output_df["p_temp4"]
        + output_df["p_temp5"]
        + output_df["p_temp6"]
        + output_df["p_temp7"]
        + output_df["p_temp8"]
        + output_df["p_temp9"]
    )

    output_df = output_df.drop(
        [
            "p_temp1",
            "p_temp2",
            "p_temp3",
            "p_temp4",
            "p_temp5",
            "p_temp6",
            "p_temp7",
            "p_temp8",
            "p_temp9",
        ],
        axis=1,
    )

    return output_df