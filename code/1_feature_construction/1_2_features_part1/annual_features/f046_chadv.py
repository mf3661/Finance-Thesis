from __future__ import annotations

import numpy as np
import pandas as pd


def add_chadv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add change in advertising expense variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "xad",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for chadv: {missing_cols}")

    output_df["xad_l1"] = output_df.groupby("permno")["xad"].shift(1)

    output_df["chadv"] = (
        np.log(output_df["xad"] + 1)
        - np.log(output_df["xad_l1"] + 1)
    )

    return output_df