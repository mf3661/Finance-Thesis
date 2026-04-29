from __future__ import annotations

import numpy as np
import pandas as pd


def add_realestate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add real estate holdings variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "fatb",
        "fatl",
        "ppegt",
        "ppent",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for realestate: {missing_cols}")

    output_df["realestate"] = (
        output_df["fatb"] + output_df["fatl"]
    ) / output_df["ppegt"]

    output_df["realestate"] = np.where(
        output_df["ppegt"].isna(),
        (output_df["fatb"] + output_df["fatl"]) / output_df["ppent"],
        output_df["realestate"],
    )

    return output_df