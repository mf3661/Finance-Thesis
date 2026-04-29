from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather

from f001_abr import build_abr_from_wrds
from f002_sue import build_sue_from_wrds


QUARTERLY_EXTRA_FEATURE_COLUMNS: list[str] = [
    "abr",
    "sue",
]


def _standardize_feature_df(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Keep permno, jdate, and selected quarterly extra feature columns.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "jdate",
    ]

    missing_key_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_key_cols:
        raise KeyError(
            f"Missing required key columns for quarterly extra feature dataframe: {missing_key_cols}"
        )

    missing_feature_cols: list[str] = [
        col for col in feature_cols if col not in output_df.columns
    ]

    if missing_feature_cols:
        raise KeyError(
            f"Missing feature columns for quarterly extra feature dataframe: {missing_feature_cols}"
        )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    output_df = output_df[
        [
            "permno",
            "jdate",
            *feature_cols,
        ]
    ].copy()

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .drop_duplicates(subset=["permno", "jdate"], keep="last")
        .reset_index(drop=True)
    )

    return output_df


def _merge_feature_frames(
    feature_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Outer-merge all quarterly extra feature dataframes by permno and jdate.
    """

    if not feature_frames:
        return pd.DataFrame(
            columns=[
                "permno",
                "jdate",
            ]
        )

    output_df: pd.DataFrame = feature_frames[0].copy()

    for feature_df in feature_frames[1:]:
        output_df = pd.merge(
            output_df,
            feature_df,
            how="outer",
            on=[
                "permno",
                "jdate",
            ],
        )

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .reset_index(drop=True)
    )

    return output_df


def _filter_start_date(
    df: pd.DataFrame,
    output_start_date: str,
) -> pd.DataFrame:
    """
    Keep observations with jdate on or after output_start_date.
    """

    output_df: pd.DataFrame = df.copy()

    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    output_df = output_df[
        output_df["jdate"] >= pd.Timestamp(output_start_date)
    ].reset_index(drop=True)

    return output_df


def build_quarterly_extra_features(
    conn: Any,
    wrds_start_date: str = "01/01/1950",
    output_start_date: str = "1950-01-01",
) -> pd.DataFrame:
    """
    Build quarterly extra features: abr and sue.
    """

    feature_frames: list[pd.DataFrame] = []

    print("Building abr...")
    abr_df: pd.DataFrame = build_abr_from_wrds(
        conn=conn,
        start_date=wrds_start_date,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=abr_df,
            feature_cols=["abr"],
        )
    )

    print("Building sue...")
    sue_df: pd.DataFrame = build_sue_from_wrds(
        conn=conn,
        start_date=wrds_start_date,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=sue_df,
            feature_cols=["sue"],
        )
    )

    quarterly_extra_df: pd.DataFrame = _merge_feature_frames(
        feature_frames=feature_frames,
    )

    quarterly_extra_df = _filter_start_date(
        df=quarterly_extra_df,
        output_start_date=output_start_date,
    )

    return quarterly_extra_df


def save_quarterly_extra_features(
    quarterly_extra_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save quarterly extra features to feather.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    feather.write_feather(
        quarterly_extra_df.reset_index(drop=True),
        output_path,
    )

    print(f"SAVED: {output_path}")