from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather

from f001_crsp_fill import (
    build_crsp_fill_from_wrds,
    save_crsp_fill,
)

from f002_merge_extra_features import (
    build_raw_features_from_files,
    save_raw_features,
)


CURRENT_DIR: Path = Path(__file__).resolve().parent

DATA_ROOT: Path = (
    CURRENT_DIR
    / "../../../../data/feature_construction"
).resolve()

PART2_DIR: Path = DATA_ROOT / "1_3_features_part2"
INTERMEDIATE_DIR: Path = PART2_DIR / "intermediate"

CRSP_FILL_CACHE_PATH: Path = INTERMEDIATE_DIR / "crsp_fill.feather"


def read_crsp_fill_cache(
    path: Path = CRSP_FILL_CACHE_PATH,
) -> pd.DataFrame:
    """
    Read cached CRSP fill dataframe.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing CRSP fill cache: {path}")

    print(f"Reading cached CRSP fill: {path}")

    crsp_fill_df: pd.DataFrame = feather.read_feather(path)

    crsp_fill_df["permno"] = crsp_fill_df["permno"].astype(int)
    crsp_fill_df["jdate"] = pd.to_datetime(crsp_fill_df["jdate"])

    return crsp_fill_df


def load_or_build_crsp_fill(
    conn: Any,
    start_date: str = "01/01/1950",
    end_date: str | None = None,
    cache_path: Path = CRSP_FILL_CACHE_PATH,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load cached CRSP fill if available; otherwise build it from WRDS.
    """

    if cache_path.exists() and not force_rebuild:
        return read_crsp_fill_cache(
            path=cache_path,
        )

    print("Building CRSP fill from WRDS...")

    crsp_fill_df: pd.DataFrame = build_crsp_fill_from_wrds(
        conn=conn,
        start_date=start_date,
        end_date=end_date,
    )

    save_crsp_fill(
        crsp_fill_df=crsp_fill_df,
        output_path=cache_path,
    )

    return crsp_fill_df


def build_and_save_raw_features(
    conn: Any,
    crsp_start_date: str = "01/01/1950",
    crsp_end_date: str | None = None,
    output_start_date: str = "1950-01-01",
    force_rebuild_crsp_fill: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and save chars_a_raw and chars_q_raw.
    """

    crsp_fill_df: pd.DataFrame = load_or_build_crsp_fill(
        conn=conn,
        start_date=crsp_start_date,
        end_date=crsp_end_date,
        cache_path=CRSP_FILL_CACHE_PATH,
        force_rebuild=force_rebuild_crsp_fill,
    )

    print("Building raw annual and quarterly feature panels...")

    chars_a_raw_df, chars_q_raw_df = build_raw_features_from_files(
        crsp_fill_df=crsp_fill_df,
        output_start_date=output_start_date,
    )

    print("Saving raw annual and quarterly feature panels...")

    save_raw_features(
        chars_a_raw_df=chars_a_raw_df,
        chars_q_raw_df=chars_q_raw_df,
    )

    print(f"Annual raw shape:    {chars_a_raw_df.shape}")
    print(f"Quarterly raw shape: {chars_q_raw_df.shape}")

    return chars_a_raw_df, chars_q_raw_df