from __future__ import annotations

import pandas as pd


def add_tang(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add asset tangibility variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "che",
        "rect",
        "invt",
        "ppent",
        "at",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for tang: {missing_cols}")

    output_df["tang"] = (
        output_df["che"]
        + output_df["rect"] * 0.715
        + output_df["invt"] * 0.547
        + output_df["ppent"] * 0.535
    ) / output_df["at"]

    return output_df