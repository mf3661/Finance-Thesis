from __future__ import annotations

import pandas as pd


def add_cashpr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly cash productivity variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "me",
        "dlttq",
        "atq",
        "cheq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly cashpr: {missing_cols}"
        )

    output_df["cashpr"] = (
        output_df["me"]
        + output_df["dlttq"]
        - output_df["atq"]
    ) / output_df["cheq"]

    return output_df