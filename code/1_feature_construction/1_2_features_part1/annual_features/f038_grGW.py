from __future__ import annotations

import numpy as np
import pandas as pd


def add_grGW(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add goodwill growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "gdwl",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for grGW: {missing_cols}")

    output_df["gdwl_l1"] = output_df.groupby("permno")["gdwl"].shift(1)

    output_df["grGW"] = (
        output_df["gdwl"] - output_df["gdwl_l1"]
    ) / output_df["gdwl"]

    condition_list = [
        (output_df["gdwl"] == 0) | (output_df["gdwl"].isna()),
        (
            output_df["gdwl"].notna()
            & (output_df["gdwl"] != 0)
            & output_df["grGW"].isna()
        ),
    ]

    choice_list = [
        0,
        1,
    ]

    output_df["grGW"] = np.select(
        condition_list,
        choice_list,
        default=output_df["grGW"],
    )

    return output_df