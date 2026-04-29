from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

from f001_crsp_fill import apply_crsp_fill


CURRENT_DIR: Path = Path(__file__).resolve().parent

DATA_ROOT: Path = (
    CURRENT_DIR
    / "../../../../data/feature_construction"
).resolve()

PART1_DIR: Path = DATA_ROOT / "1_2_features_part1"
PART2_DIR: Path = DATA_ROOT / "1_3_features_part2"

ANNUAL_ACCOUNTING_PATH: Path = PART1_DIR / "chars_a_accounting.feather"
QUARTERLY_ACCOUNTING_PATH: Path = PART1_DIR / "chars_q_accounting.feather"

MONTHLY_EXTRA_PATH: Path = PART2_DIR / "monthly_extra_features.feather"
QUARTERLY_EXTRA_PATH: Path = PART2_DIR / "quarterly_extra_features.feather"

ANNUAL_RAW_OUTPUT_PATH: Path = PART2_DIR / "chars_a_raw.feather"
QUARTERLY_RAW_OUTPUT_PATH: Path = PART2_DIR / "chars_q_raw.feather"


def read_feather(path: Path) -> pd.DataFrame:
    """
    Read a feather file.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    print(f"Reading: {path}")

    return feather.read_feather(path)


def save_feather(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save dataframe to feather.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    feather.write_feather(
        df.reset_index(drop=True),
        path,
    )

    print(f"SAVED: {path}")


def _standardize_accounting_panel(
    df: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:
    """
    Standardize accounting panel keys before merging extra features.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "gvkey",
        "jdate",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for {panel_name} accounting panel: {missing_cols}"
        )

    output_df = output_df.dropna(
        subset=[
            "permno",
        ]
    ).copy()

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["gvkey"] = output_df["gvkey"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .drop_duplicates(subset=["permno", "jdate"], keep="last")
        .reset_index(drop=True)
    )

    return output_df


def _standardize_extra_features(
    df: pd.DataFrame,
    extra_name: str,
) -> pd.DataFrame:
    """
    Standardize monthly or quarterly extra feature dataframe.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "jdate",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for {extra_name}: {missing_cols}"
        )

    output_df["permno"] = output_df["permno"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    feature_cols: list[str] = [
        col for col in output_df.columns if col not in ["permno", "jdate"]
    ]

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


def _merge_extra_features(
    accounting_df: pd.DataFrame,
    monthly_extra_df: pd.DataFrame,
    quarterly_extra_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge monthly and quarterly extra features onto one accounting panel.
    """

    output_df: pd.DataFrame = accounting_df.copy()

    output_df = pd.merge(
        output_df,
        monthly_extra_df,
        how="left",
        on=[
            "permno",
            "jdate",
        ],
    )

    output_df = pd.merge(
        output_df,
        quarterly_extra_df,
        how="left",
        on=[
            "permno",
            "jdate",
        ],
    )

    return output_df


def _filter_valid_market_records(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep observations with valid CRSP returns and common-stock exchange/share codes.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "jdate",
        "ret",
        "retx",
        "retadj",
        "exchcd",
        "shrcd",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for raw market filtering: {missing_cols}"
        )

    output_df = output_df.dropna(
        subset=[
            "permno",
            "jdate",
            "ret",
            "retx",
            "retadj",
        ]
    ).copy()

    output_df = output_df[
        output_df["exchcd"].isin([1, 2, 3])
        & output_df["shrcd"].isin([10, 11])
    ].reset_index(drop=True)

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


def build_raw_feature_panel(
    accounting_df: pd.DataFrame,
    monthly_extra_df: pd.DataFrame,
    quarterly_extra_df: pd.DataFrame,
    crsp_fill_df: pd.DataFrame | None,
    panel_name: str,
    output_start_date: str = "1950-01-01",
) -> pd.DataFrame:
    """
    Build one raw feature panel from accounting, extra features, and CRSP fill data.
    """

    output_df: pd.DataFrame = _standardize_accounting_panel(
        df=accounting_df,
        panel_name=panel_name,
    )

    output_df = _merge_extra_features(
        accounting_df=output_df,
        monthly_extra_df=monthly_extra_df,
        quarterly_extra_df=quarterly_extra_df,
    )

    if crsp_fill_df is not None:
        output_df = apply_crsp_fill(
            df=output_df,
            crsp_fill_df=crsp_fill_df,
        )
    else:
        output_df = _filter_valid_market_records(
            df=output_df,
        )

    output_df = _filter_start_date(
        df=output_df,
        output_start_date=output_start_date,
    )

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .reset_index(drop=True)
    )

    return output_df


def build_raw_features(
    annual_accounting_df: pd.DataFrame,
    quarterly_accounting_df: pd.DataFrame,
    monthly_extra_df: pd.DataFrame,
    quarterly_extra_df: pd.DataFrame,
    crsp_fill_df: pd.DataFrame | None,
    output_start_date: str = "1950-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build annual and quarterly raw feature panels.
    """

    monthly_extra_clean_df: pd.DataFrame = _standardize_extra_features(
        df=monthly_extra_df,
        extra_name="monthly_extra_features",
    )

    quarterly_extra_clean_df: pd.DataFrame = _standardize_extra_features(
        df=quarterly_extra_df,
        extra_name="quarterly_extra_features",
    )

    chars_a_raw_df: pd.DataFrame = build_raw_feature_panel(
        accounting_df=annual_accounting_df,
        monthly_extra_df=monthly_extra_clean_df,
        quarterly_extra_df=quarterly_extra_clean_df,
        crsp_fill_df=crsp_fill_df,
        panel_name="annual",
        output_start_date=output_start_date,
    )

    chars_q_raw_df: pd.DataFrame = build_raw_feature_panel(
        accounting_df=quarterly_accounting_df,
        monthly_extra_df=monthly_extra_clean_df,
        quarterly_extra_df=quarterly_extra_clean_df,
        crsp_fill_df=crsp_fill_df,
        panel_name="quarterly",
        output_start_date=output_start_date,
    )

    return chars_a_raw_df, chars_q_raw_df


def build_raw_features_from_files(
    crsp_fill_df: pd.DataFrame | None,
    output_start_date: str = "1950-01-01",
    annual_accounting_path: Path = ANNUAL_ACCOUNTING_PATH,
    quarterly_accounting_path: Path = QUARTERLY_ACCOUNTING_PATH,
    monthly_extra_path: Path = MONTHLY_EXTRA_PATH,
    quarterly_extra_path: Path = QUARTERLY_EXTRA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read part1 and part2 feature files, then build raw annual and quarterly panels.
    """

    annual_accounting_df: pd.DataFrame = read_feather(
        annual_accounting_path,
    )

    quarterly_accounting_df: pd.DataFrame = read_feather(
        quarterly_accounting_path,
    )

    monthly_extra_df: pd.DataFrame = read_feather(
        monthly_extra_path,
    )

    quarterly_extra_df: pd.DataFrame = read_feather(
        quarterly_extra_path,
    )

    chars_a_raw_df, chars_q_raw_df = build_raw_features(
        annual_accounting_df=annual_accounting_df,
        quarterly_accounting_df=quarterly_accounting_df,
        monthly_extra_df=monthly_extra_df,
        quarterly_extra_df=quarterly_extra_df,
        crsp_fill_df=crsp_fill_df,
        output_start_date=output_start_date,
    )

    return chars_a_raw_df, chars_q_raw_df


def save_raw_features(
    chars_a_raw_df: pd.DataFrame,
    chars_q_raw_df: pd.DataFrame,
    annual_output_path: Path = ANNUAL_RAW_OUTPUT_PATH,
    quarterly_output_path: Path = QUARTERLY_RAW_OUTPUT_PATH,
) -> None:
    """
    Save raw annual and quarterly feature panels.
    """

    save_feather(
        chars_a_raw_df,
        annual_output_path,
    )

    save_feather(
        chars_q_raw_df,
        quarterly_output_path,
    )