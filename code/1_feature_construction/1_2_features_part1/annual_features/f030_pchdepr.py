from __future__ import annotations

import pandas as pd


def add_pchdepr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add percent change in depreciation variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "dp",
        "ppent",
        "ppent_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for pchdepr: {missing_cols}")

    output_df["dp_l1"] = output_df.groupby("permno")["dp"].shift(1)

    output_df["pchdepr"] = (
        (output_df["dp"] / output_df["ppent"])
        - (output_df["dp_l1"] / output_df["ppent_l1"])
    ) / (
        output_df["dp_l1"] / output_df["ppent"]
    )

    return output_df