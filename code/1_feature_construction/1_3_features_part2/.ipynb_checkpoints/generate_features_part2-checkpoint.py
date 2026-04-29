from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.feather as feather
import wrds


WRDS_USERNAME: str = ""

WRDS_START_DATE: str = "01/01/1950"
OUTPUT_START_DATE: str = "1950-01-01"

N_PROCESSES: int = 20
MIN_OBS: int = 21

INCLUDE_RE: bool = True
SKIP_RE_ON_ERROR: bool = True
FORCE_REBUILD_ICLINK: bool = False
FORCE_REBUILD_CRSP_FILL: bool = False

BUILD_MONTHLY_EXTRA: bool = True
BUILD_QUARTERLY_EXTRA: bool = True
BUILD_RAW_FEATURES: bool = True

CURRENT_DIR: Path = Path(__file__).resolve().parent

MONTHLY_EXTRA_DIR: Path = CURRENT_DIR / "monthly_extra"
QUARTERLY_EXTRA_DIR: Path = CURRENT_DIR / "quarterly_extra"
ASSEMBLE_RAW_DIR: Path = CURRENT_DIR / "assemble_raw"

OUTPUT_DIR: Path = Path("../../../data/feature_construction/1_3_features_part2")
INTERMEDIATE_DIR: Path = OUTPUT_DIR / "intermediate"


def add_module_paths() -> None:
    """Add feature subfolders to Python import path."""

    module_dirs: list[Path] = [
        MONTHLY_EXTRA_DIR,
        QUARTERLY_EXTRA_DIR,
        ASSEMBLE_RAW_DIR,
    ]

    for module_dir in module_dirs:
        if not module_dir.exists():
            raise FileNotFoundError(f"Missing module folder: {module_dir}")

        sys.path.insert(0, str(module_dir))


add_module_paths()

from build_monthly_extra_features import build_monthly_extra_features
from build_quarterly_extra_features import build_quarterly_extra_features
from build_raw_features import build_and_save_raw_features


def get_wrds_connection(username: str = "") -> wrds.Connection:
    """Create WRDS connection."""

    import sqlalchemy.engine

    sys.modules["sqlalchemy"].create_engine = sqlalchemy.engine.create_engine
    wrds.Connection.load_library_list = lambda self: None

    if username:
        conn = wrds.Connection(wrds_username=username)
    else:
        conn = wrds.Connection()

    print("Connected to WRDS successfully!")

    return conn


def save_feather(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save dataframe to feather."""

    path.parent.mkdir(parents=True, exist_ok=True)

    feather.write_feather(
        df.reset_index(drop=True),
        path,
    )

    print(f"SAVED: {path}")


def read_feather(path: Path) -> pd.DataFrame:
    """Read dataframe from feather."""

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    print(f"Reading: {path}")

    return feather.read_feather(path)


def filter_start_date(
    df: pd.DataFrame,
    output_start_date: str,
) -> pd.DataFrame:
    """Filter dataframe by jdate."""

    output_df: pd.DataFrame = df.copy()

    if "jdate" not in output_df.columns:
        raise KeyError("Missing jdate column for start-date filtering.")

    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    output_df = output_df[
        output_df["jdate"] >= pd.Timestamp(output_start_date)
    ].reset_index(drop=True)

    return output_df


def build_or_load_monthly_extra(
    conn: Any,
) -> pd.DataFrame:
    """Build or load monthly extra features."""

    output_path: Path = OUTPUT_DIR / "monthly_extra_features.feather"

    if not BUILD_MONTHLY_EXTRA and output_path.exists():
        return read_feather(output_path)

    print("Building monthly extra features...")

    monthly_extra_df: pd.DataFrame = build_monthly_extra_features(
        conn=conn,
        n_processes=N_PROCESSES,
        min_obs=MIN_OBS,
        daily_start_date=WRDS_START_DATE,
        include_re=INCLUDE_RE,
        skip_re_on_error=SKIP_RE_ON_ERROR,
        iclink_cache_path=INTERMEDIATE_DIR / "iclink.feather",
        force_rebuild_iclink=FORCE_REBUILD_ICLINK,
    )

    monthly_extra_df = filter_start_date(
        df=monthly_extra_df,
        output_start_date=OUTPUT_START_DATE,
    )

    save_feather(
        monthly_extra_df,
        output_path,
    )

    return monthly_extra_df


def build_or_load_quarterly_extra(
    conn: Any,
) -> pd.DataFrame:
    """Build or load quarterly extra features."""

    output_path: Path = OUTPUT_DIR / "quarterly_extra_features.feather"

    if not BUILD_QUARTERLY_EXTRA and output_path.exists():
        return read_feather(output_path)

    print("Building quarterly extra features...")

    quarterly_extra_df: pd.DataFrame = build_quarterly_extra_features(
        conn=conn,
        wrds_start_date=WRDS_START_DATE,
        output_start_date=OUTPUT_START_DATE,
    )

    save_feather(
        quarterly_extra_df,
        output_path,
    )

    return quarterly_extra_df


def main() -> None:
    """Run feature construction part 2."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    conn: Any = get_wrds_connection(WRDS_USERNAME)

    monthly_extra_df: pd.DataFrame | None = None
    quarterly_extra_df: pd.DataFrame | None = None
    chars_a_raw_df: pd.DataFrame | None = None
    chars_q_raw_df: pd.DataFrame | None = None

    try:
        monthly_extra_df = build_or_load_monthly_extra(
            conn=conn,
        )

        quarterly_extra_df = build_or_load_quarterly_extra(
            conn=conn,
        )

        if BUILD_RAW_FEATURES:
            print("Building chars_a_raw and chars_q_raw...")

            chars_a_raw_df, chars_q_raw_df = build_and_save_raw_features(
                conn=conn,
                crsp_start_date=WRDS_START_DATE,
                crsp_end_date=None,
                output_start_date=OUTPUT_START_DATE,
                force_rebuild_crsp_fill=FORCE_REBUILD_CRSP_FILL,
            )

    finally:
        close_method = getattr(conn, "close", None)

        if callable(close_method):
            close_method()

    print("\nFeature construction part 2 finished.")

    if monthly_extra_df is not None:
        print(f"Monthly extra shape:   {monthly_extra_df.shape}")

    if quarterly_extra_df is not None:
        print(f"Quarterly extra shape: {quarterly_extra_df.shape}")

    if chars_a_raw_df is not None:
        print(f"Annual raw shape:      {chars_a_raw_df.shape}")

    if chars_q_raw_df is not None:
        print(f"Quarterly raw shape:   {chars_q_raw_df.shape}")

    print(f"Output directory:      {OUTPUT_DIR}")


if __name__ == "__main__":
    main()