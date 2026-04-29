from __future__ import annotations

import numpy as np
import pandas as pd


def add_pctacc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly percent accruals variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "ibq",
        "oancfy",
        "actq",
        "actq_l4",
        "cheq",
        "cheq_l4",
        "lctq",
        "lctq_l4",
        "dlcq",
        "dlcq_l4",
        "txpq",
        "txpq_l4",
        "dpq",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly pctacc: {missing_cols}"
        )

    balance_sheet_accrual = (
        (output_df["actq"] - output_df["actq_l4"])
        - (output_df["cheq"] - output_df["cheq_l4"])
        - (output_df["lctq"] - output_df["lctq_l4"])
        + (output_df["dlcq"] - output_df["dlcq_l4"])
        + (output_df["txpq"] - output_df["txpq_l4"]).fillna(0)
        - output_df["dpq"]
    )

    condition_list = [
        output_df["ibq"] == 0,
        output_df["oancfy"].isna(),
        output_df["oancfy"].isna() & (output_df["ibq"] == 0),
    ]

    choice_list = [
        (output_df["ibq"] - output_df["oancfy"]) / 0.01,
        balance_sheet_accrual / output_df["ibq"].abs(),
        balance_sheet_accrual / 0.01,
    ]

    output_df["pctacc"] = np.select(
        condition_list,
        choice_list,
        default=(
            output_df["ibq"] - output_df["oancfy"]
        ) / output_df["ibq"].abs(),
    )

    return output_df