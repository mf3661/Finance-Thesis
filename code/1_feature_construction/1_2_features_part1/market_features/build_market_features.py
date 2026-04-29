from __future__ import annotations

import pandas as pd

from market_data import prepare_market_data

from f001_mom1m import add_mom1m
from f002_mom6m import add_mom6m
from f003_mom12m import add_mom12m
from f004_mom36m import add_mom36m
from f005_mom60m import add_mom60m
from f006_chmom import add_chmom
from f007_seas1a import add_seas1a
from f008_dolvol import add_dolvol
from f009_turn import add_turn
from f010_dy import add_dy


MARKET_OUTPUT_COLUMNS: list[str] = [
    "permno",
    "date",
    "jdate",
    "ret",
    "retx",
    "retadj",
    "prc",
    "shrout",
    "me",
    "mom1m",
    "mom6m",
    "mom12m",
    "mom36m",
    "mom60m",
    "seas1a",
    "chmom",
    "turn",
    "dolvol",
    "dy",
]


def build_market_features(
    base_monthly_df: pd.DataFrame,
    dlret_df: pd.DataFrame,
    keep_all_columns: bool = False,
) -> pd.DataFrame:
    """
    Build monthly market and momentum features.
    """

    df: pd.DataFrame = prepare_market_data(
        crsp_monthly_df=base_monthly_df,
        dlret_df=dlret_df,
        scale_me_to_millions=True,
    )

    df = add_mom1m(df)
    df = add_mom6m(df)
    df = add_mom12m(df)
    df = add_mom36m(df)
    df = add_mom60m(df)
    df = add_chmom(df)
    df = add_seas1a(df)
    df = add_dolvol(df)
    df = add_turn(df)
    df = add_dy(df)

    if keep_all_columns:
        return df

    missing_cols: list[str] = [
        col for col in MARKET_OUTPUT_COLUMNS if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing expected market output columns: {missing_cols}"
        )

    output_df: pd.DataFrame = df[MARKET_OUTPUT_COLUMNS].copy()

    return output_df