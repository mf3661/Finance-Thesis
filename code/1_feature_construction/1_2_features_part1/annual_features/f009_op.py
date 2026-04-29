from __future__ import annotations

import numpy as np
import pandas as pd


def add_op(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add operating profitability variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "cogs",
        "xint",
        "xsga",
        "revt",
        "be",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for op: {missing_cols}")

    output_df["cogs0"] = np.where(
        output_df["cogs"].isna(),
        0,
        output_df["cogs"],
    )

    output_df["xint0"] = np.where(
        output_df["xint"].isna(),
        0,
        output_df["xint"],
    )

    output_df["xsga0"] = np.where(
        output_df["xsga"].isna(),
        0,
        output_df["xsga"],
    )

    condition_list = [
        output_df["revt"].isna(),
        output_df["be"].isna(),
    ]

    choice_list = [
        np.nan,
        np.nan,
    ]

    output_df["op"] = np.select(
        condition_list,
        choice_list,
        default=(
            output_df["revt"]
            - output_df["cogs0"]
            - output_df["xsga0"]
            - output_df["xint0"]
        ) / output_df["be"],
    )

    return output_df