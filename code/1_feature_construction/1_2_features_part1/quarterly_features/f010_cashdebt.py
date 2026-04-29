from __future__ import annotations

import pandas as pd


def add_cashdebt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly cash-flow-to-debt variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "ltq",
        "ibq4",
        "dpq4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly cashdebt: {missing_cols}"
        )

    output_df["ltq_l4"] = output_df.groupby("permno")["ltq"].shift(4)

    output_df["cashdebt"] = (
        output_df["ibq4"] + output_df["dpq4"]
    ) / (
        (output_df["ltq"] + output_df["ltq_l4"]) / 2
    )

    return output_df