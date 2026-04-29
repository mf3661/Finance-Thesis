from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather

from f001_beta import (
    load_beta_daily_data,
    add_beta,
)
from f002_rvar_capm import add_rvar_capm
from f003_std_dolvol import (
    load_std_dolvol_daily_data,
    add_std_dolvol,
)
from f004_std_turn import (
    load_std_turn_daily_data,
    add_std_turn,
)
from f005_rvar_ff3 import add_rvar_ff3
from f006_baspread import (
    load_baspread_daily_data,
    add_baspread,
)
from f007_zerotrade import (
    load_zerotrade_daily_data,
    add_zerotrade,
)
from f008_maxret import (
    load_maxret_daily_data,
    add_maxret,
)
from f009_rvar_mean import (
    load_rvar_mean_daily_data,
    add_rvar_mean,
)
from f010_ill import (
    load_ill_daily_data,
    add_ill,
)
from f011_re import (
    load_re_source_data,
    add_re,
)
from iclink import build_iclink_from_wrds


MONTHLY_EXTRA_FEATURE_COLUMNS: list[str] = [
    "beta",
    "rvar_capm",
    "std_dolvol",
    "std_turn",
    "rvar_ff3",
    "baspread",
    "zerotrade",
    "maxret",
    "rvar_mean",
    "ill",
    "re",
]


def _standardize_feature_df(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Keep permno, jdate, and selected feature columns.
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
            f"Missing required key columns for monthly extra feature dataframe: {missing_key_cols}"
        )

    missing_feature_cols: list[str] = [
        col for col in feature_cols if col not in output_df.columns
    ]

    if missing_feature_cols:
        raise KeyError(
            f"Missing feature columns for monthly extra feature dataframe: {missing_feature_cols}"
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
    Outer-merge all monthly extra feature dataframes by permno and jdate.
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


def _load_or_build_iclink(
    conn: Any,
    iclink_cache_path: Path | None,
    force_rebuild_iclink: bool,
) -> pd.DataFrame:
    """
    Load cached iclink if available; otherwise build it from WRDS.
    """

    if (
        iclink_cache_path is not None
        and iclink_cache_path.exists()
        and not force_rebuild_iclink
    ):
        print(f"Reading cached ICLINK: {iclink_cache_path}")
        iclink_df: pd.DataFrame = feather.read_feather(iclink_cache_path)
        return iclink_df

    print("Building ICLINK from WRDS...")
    iclink_df = build_iclink_from_wrds(conn=conn)

    if iclink_cache_path is not None:
        iclink_cache_path.parent.mkdir(parents=True, exist_ok=True)
        feather.write_feather(
            iclink_df.reset_index(drop=True),
            iclink_cache_path,
        )
        print(f"SAVED cached ICLINK: {iclink_cache_path}")

    return iclink_df


def build_monthly_extra_features(
    conn: Any,
    n_processes: int = 20,
    min_obs: int = 21,
    daily_start_date: str = "01/01/1959",
    include_re: bool = True,
    skip_re_on_error: bool = True,
    iclink_cache_path: Path | None = None,
    force_rebuild_iclink: bool = False,
) -> pd.DataFrame:
    """
    Build all monthly extra features.
    """

    feature_frames: list[pd.DataFrame] = []

    print("Building beta, rvar_capm, and rvar_ff3 source daily data...")
    daily_ff_df: pd.DataFrame = load_beta_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building beta...")
    beta_df: pd.DataFrame = add_beta(
        daily_df=daily_ff_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=beta_df,
            feature_cols=["beta"],
        )
    )

    print("Building rvar_capm...")
    rvar_capm_df: pd.DataFrame = add_rvar_capm(
        daily_df=daily_ff_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=rvar_capm_df,
            feature_cols=["rvar_capm"],
        )
    )

    print("Building rvar_ff3...")
    rvar_ff3_df: pd.DataFrame = add_rvar_ff3(
        daily_df=daily_ff_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=rvar_ff3_df,
            feature_cols=["rvar_ff3"],
        )
    )

    print("Building std_dolvol source daily data...")
    std_dolvol_daily_df: pd.DataFrame = load_std_dolvol_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building std_dolvol...")
    std_dolvol_df: pd.DataFrame = add_std_dolvol(
        daily_df=std_dolvol_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=std_dolvol_df,
            feature_cols=["std_dolvol"],
        )
    )

    print("Building std_turn / zerotrade source daily data...")
    turnover_daily_df: pd.DataFrame = load_std_turn_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building std_turn...")
    std_turn_df: pd.DataFrame = add_std_turn(
        daily_df=turnover_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=std_turn_df,
            feature_cols=["std_turn"],
        )
    )

    print("Building zerotrade...")
    zerotrade_df: pd.DataFrame = add_zerotrade(
        daily_df=turnover_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=zerotrade_df,
            feature_cols=["zerotrade"],
        )
    )

    print("Building baspread source daily data...")
    baspread_daily_df: pd.DataFrame = load_baspread_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building baspread...")
    baspread_df: pd.DataFrame = add_baspread(
        daily_df=baspread_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=baspread_df,
            feature_cols=["baspread"],
        )
    )

    print("Building maxret source daily data...")
    maxret_daily_df: pd.DataFrame = load_maxret_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building maxret...")
    maxret_df: pd.DataFrame = add_maxret(
        daily_df=maxret_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=maxret_df,
            feature_cols=["maxret"],
        )
    )

    print("Building rvar_mean source daily data...")
    rvar_mean_daily_df: pd.DataFrame = load_rvar_mean_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building rvar_mean...")
    rvar_mean_df: pd.DataFrame = add_rvar_mean(
        daily_df=rvar_mean_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=rvar_mean_df,
            feature_cols=["rvar_mean"],
        )
    )

    print("Building ill source daily data...")
    ill_daily_df: pd.DataFrame = load_ill_daily_data(
        conn=conn,
        start_date=daily_start_date,
    )

    print("Building ill...")
    ill_df: pd.DataFrame = add_ill(
        daily_df=ill_daily_df,
        n_processes=n_processes,
        min_obs=min_obs,
    )

    feature_frames.append(
        _standardize_feature_df(
            df=ill_df,
            feature_cols=["ill"],
        )
    )

    if include_re:
        print("Building re...")

        try:
            iclink_df: pd.DataFrame = _load_or_build_iclink(
                conn=conn,
                iclink_cache_path=iclink_cache_path,
                force_rebuild_iclink=force_rebuild_iclink,
            )

            ibes_df, crsp_msf_df = load_re_source_data(
                conn=conn,
                start_date=None,
            )

            re_df: pd.DataFrame = add_re(
                ibes_df=ibes_df,
                crsp_msf_df=crsp_msf_df,
                iclink_df=iclink_df,
            )

            feature_frames.append(
                _standardize_feature_df(
                    df=re_df,
                    feature_cols=["re"],
                )
            )

        except Exception as exc:
            if not skip_re_on_error:
                raise

            print(
                "WARNING: Skipping re because IBES / ICLINK step failed. "
                f"Original error: {repr(exc)}"
            )

    monthly_extra_df: pd.DataFrame = _merge_feature_frames(
        feature_frames=feature_frames,
    )

    return monthly_extra_df