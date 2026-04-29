from __future__ import annotations

import pandas as pd


def add_rdbias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R&D bias variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "xrd0",
        "ib",
        "ceq_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for rdbias: {missing_cols}")

    output_df["xrd_l1"] = output_df.groupby("permno")["xrd0"].shift(1)

    output_df["rdbias"] = (
        output_df["xrd0"] / output_df["xrd_l1"]
    ) - 1 - (
        output_df["ib"] / output_df["ceq_l1"]
    )

    return output_df