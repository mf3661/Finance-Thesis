from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
import wrds


from compustat_data import (
    load_and_clean_compustat_annual,
    load_and_clean_compustat_quarterly,
)

from crsp_data import (
    load_and_clean_crsp_monthly,
)

from ccm_link import (
    load_and_clean_ccm_linktable,
    build_annual_base,
    build_quarterly_base,
    build_monthly_base,
)


WRDS_USERNAME: str = "" # Please change to a valid username.
START_DATE: str = "01/01/1950"
END_DATE: str = "03/31/2025"

OUTPUT_DIR: Path = Path("../../../data/feature_construction/1_1_base_data")
SAVE_INTERMEDIATE: bool = True


def get_wrds_connection(username: str) -> wrds.Connection:
    """Create WRDS connection."""
    import sys
    import sqlalchemy.engine

    sys.modules["sqlalchemy"].create_engine = sqlalchemy.engine.create_engine
    wrds.Connection.load_library_list = lambda self: None

    if username:
        conn = wrds.Connection(wrds_username=username)
    else:
        conn = wrds.Connection()

    print("Connected to WRDS successfully!")

    return conn


def save_feather(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to feather."""

    path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(df, path)
    print(f"SAVED: {path}")


def main() -> None:
    """Build and save annual, quarterly, and monthly base data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_wrds_connection(WRDS_USERNAME)

    # ---------------------------------------------------------------------
    # 1. Load and clean source-level datasets.
    # ---------------------------------------------------------------------

    compustat_annual_df = load_and_clean_compustat_annual(
        conn=conn,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    compustat_quarterly_df = load_and_clean_compustat_quarterly(
        conn=conn,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    crsp_monthly_df = load_and_clean_crsp_monthly(
        conn=conn,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    ccm_link_df = load_and_clean_ccm_linktable(
        conn=conn,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # ---------------------------------------------------------------------
    # 2. Optionally save cleaned source-level data.
    # ---------------------------------------------------------------------

    if SAVE_INTERMEDIATE:
        save_feather(
            compustat_annual_df,
            OUTPUT_DIR / "clean_compustat_annual.feather",
        )

        save_feather(
            compustat_quarterly_df,
            OUTPUT_DIR / "clean_compustat_quarterly.feather",
        )

        save_feather(
            crsp_monthly_df,
            OUTPUT_DIR / "clean_crsp_monthly.feather",
        )

        save_feather(
            ccm_link_df,
            OUTPUT_DIR / "clean_ccm_linktable.feather",
        )

    # ---------------------------------------------------------------------
    # 3. Build linked base datasets.
    # ---------------------------------------------------------------------

    annual_base_df = build_annual_base(
        compustat_annual_df=compustat_annual_df,
        crsp_monthly_df=crsp_monthly_df,
        ccm_link_df=ccm_link_df,
    )

    quarterly_base_df = build_quarterly_base(
        compustat_quarterly_df=compustat_quarterly_df,
        crsp_monthly_df=crsp_monthly_df,
        ccm_link_df=ccm_link_df,
    )

    monthly_base_df = build_monthly_base(
        crsp_monthly_df=crsp_monthly_df,
    )

    # ---------------------------------------------------------------------
    # 4. Save final base datasets.
    # ---------------------------------------------------------------------

    save_feather(
        annual_base_df,
        OUTPUT_DIR / "base_annual.feather",
    )

    save_feather(
        quarterly_base_df,
        OUTPUT_DIR / "base_quarterly.feather",
    )

    save_feather(
        monthly_base_df,
        OUTPUT_DIR / "base_monthly.feather",
    )

    # ---------------------------------------------------------------------
    # 5. Print simple diagnostics.
    # ---------------------------------------------------------------------

    print("\nBase data construction finished.")
    print(f"Annual base shape:    {annual_base_df.shape}")
    print(f"Quarterly base shape: {quarterly_base_df.shape}")
    print(f"Monthly base shape:   {monthly_base_df.shape}")


if __name__ == "__main__":
    main()