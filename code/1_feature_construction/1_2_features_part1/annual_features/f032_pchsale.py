from __future__ import annotations

import pandas as pd


def add_pchgm_pchsale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gross margin change minus sales growth variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "sale",
        "sale_l1",
        "cogs",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for pchgm_pchsale: {missing_cols}"
        )

    output_df["cogs_l1"] = output_df.groupby("permno")["cogs"].shift(1)

    output_df["pchgm_pchsale"] = (
        (
            (output_df["sale"] - output_df["cogs"])
            - (output_df["sale_l1"] - output_df["cogs_l1"])
        )
        / (output_df["sale_l1"] - output_df["cogs_l1"])
    ) - (
        (output_df["sale"] - output_df["sale_l1"])
        / output_df["sale"]
    )

    return output_df