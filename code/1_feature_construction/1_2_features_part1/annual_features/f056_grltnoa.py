from __future__ import annotations

import pandas as pd


def add_grltnoa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add growth in long-term net operating assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "rect",
        "invt",
        "ppent",
        "aco",
        "intan",
        "ao",
        "ap",
        "lco",
        "lo",
        "dp",
        "at",
        "at_l1",
        "invt_l1",
        "ppent_l1",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for grltnoa: {missing_cols}")

    output_df["aco_l1"] = output_df.groupby("permno")["aco"].shift(1)
    output_df["intan_l1"] = output_df.groupby("permno")["intan"].shift(1)
    output_df["ao_l1"] = output_df.groupby("permno")["ao"].shift(1)
    output_df["ap_l1"] = output_df.groupby("permno")["ap"].shift(1)
    output_df["lco_l1"] = output_df.groupby("permno")["lco"].shift(1)
    output_df["lo_l1"] = output_df.groupby("permno")["lo"].shift(1)
    output_df["rect_l1"] = output_df.groupby("permno")["rect"].shift(1)

    current_noa_components = (
        output_df["rect"]
        + output_df["invt"]
        + output_df["ppent"]
        + output_df["aco"]
        + output_df["intan"]
        + output_df["ao"]
        - output_df["ap"]
        - output_df["lco"]
        - output_df["lo"]
    )

    lagged_noa_components = (
        output_df["rect_l1"]
        + output_df["invt_l1"]
        + output_df["ppent_l1"]
        + output_df["aco_l1"]
        + output_df["intan_l1"]
        + output_df["ao_l1"]
        - output_df["ap_l1"]
        - output_df["lco_l1"]
        - output_df["lo_l1"]
    )

    working_capital_accruals = (
        output_df["rect"]
        - output_df["rect_l1"]
        + output_df["invt"]
        - output_df["invt_l1"]
        + output_df["aco"]
        - output_df["aco_l1"]
        - (
            output_df["ap"]
            - output_df["ap_l1"]
            + output_df["lco"]
            - output_df["lco_l1"]
        )
        - output_df["dp"]
    )

    output_df["grltnoa"] = (
        current_noa_components
        - lagged_noa_components
        - working_capital_accruals
    ) / (
        (output_df["at"] + output_df["at_l1"]) / 2
    )

    return output_df