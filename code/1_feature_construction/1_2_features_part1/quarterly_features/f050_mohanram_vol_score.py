from __future__ import annotations

import numpy as np
import pandas as pd


def add_mohanram_vol_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly Mohanram volatility score components.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "fyearq",
        "fqtr",
        "ffi49",
        "roavol",
        "sgrvol",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly Mohanram volatility score: {missing_cols}"
        )

    industry_temp_df = (
        output_df.groupby(["fyearq", "fqtr", "ffi49"], as_index=False)["roavol"]
        .median()
        .rename(columns={"roavol": "md_roavol"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyearq", "fqtr", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyearq", "fqtr", "ffi49"], as_index=False)["sgrvol"]
        .median()
        .rename(columns={"sgrvol": "md_sgrvol"})
    )

    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyearq", "fqtr", "ffi49"],
    ).reset_index(drop=True)

    output_df["m7"] = np.where(
        output_df["roavol"] < output_df["md_roavol"],
        1,
        0,
    )

    output_df["m8"] = np.where(
        output_df["sgrvol"] < output_df["md_sgrvol"],
        1,
        0,
    )

    return output_df