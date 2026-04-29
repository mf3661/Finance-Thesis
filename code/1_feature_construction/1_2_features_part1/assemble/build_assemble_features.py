from __future__ import annotations

import pandas as pd

from f001_populate_to_monthly import populate_to_monthly
from f002_recompute_monthly_me_features import (
    recompute_annual_monthly_me_features,
    recompute_quarterly_monthly_me_features,
)
from f003_select_final_columns import select_final_columns


def assemble_features(
    annual_features_df: pd.DataFrame,
    quarterly_features_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble annual and quarterly features onto monthly market panels.
    """

    annual_monthly_df: pd.DataFrame = populate_to_monthly(
        market_df=market_features_df,
        accounting_df=annual_features_df,
        exclude_market_cols=("dy",),
    )

    quarterly_monthly_df: pd.DataFrame = populate_to_monthly(
        market_df=market_features_df,
        accounting_df=quarterly_features_df,
        exclude_market_cols=("dy",),
    )

    annual_monthly_df = recompute_annual_monthly_me_features(
        annual_df=annual_monthly_df,
    )

    quarterly_monthly_df = recompute_quarterly_monthly_me_features(
        quarterly_df=quarterly_monthly_df,
    )

    annual_output_df, quarterly_output_df = select_final_columns(
        annual_df=annual_monthly_df,
        quarterly_df=quarterly_monthly_df,
    )

    return annual_output_df, quarterly_output_df