from __future__ import annotations

import numpy as np
import pandas as pd


def add_hire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add employee growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "emp",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for hire: {missing_cols}")

    output_df["emp_l1"] = output_df.groupby("permno")["emp"].shift(1)

    output_df["hire"] = (
        output_df["emp"] - output_df["emp_l1"]
    ) / output_df["emp_l1"]

    output_df["hire"] = np.where(
        output_df["emp"].isna() | output_df["emp_l1"].isna(),
        0,
        output_df["hire"],
    )

    return output_df