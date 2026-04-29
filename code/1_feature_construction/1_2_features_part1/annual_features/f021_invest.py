from __future__ import annotations

import numpy as np
import pandas as pd


def add_invest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add investment variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ppent",
        "invt",
        "ppegt",
        "at_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for invest: {missing_cols}")

    output_df["ppent_l1"] = output_df.groupby("permno")["ppent"].shift(1)
    output_df["invt_l1"] = output_df.groupby("permno")["invt"].shift(1)

    output_df["invest"] = np.where(
        output_df["ppegt"].isna(),
        (
            (output_df["ppent"] - output_df["ppent_l1"])
            + (output_df["invt"] - output_df["invt_l1"])
        ) / output_df["at_l1"],
        (
            (output_df["ppegt"] - output_df["ppent_l1"])
            + (output_df["invt"] - output_df["invt_l1"])
        ) / output_df["at_l1"],
    )

    return output_df