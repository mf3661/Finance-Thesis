from __future__ import annotations

import numpy as np
import pandas as pd


def add_mohanram_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add annual Mohanram score components.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "fyear",
        "ffi49",
        "roa",
        "oancf",
        "at",
        "at_l1",
        "ib",
        "dp",
        "xrd0",
        "capxint",
        "xadint",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(f"Missing required columns for Mohanram score: {missing_cols}")

    avg_at = (output_df["at"] + output_df["at_l1"]) / 2

    output_df["cfroa"] = output_df["oancf"] / avg_at

    output_df["cfroa"] = np.where(
        output_df["oancf"].isna(),
        (output_df["ib"] + output_df["dp"]) / avg_at,
        output_df["cfroa"],
    )

    output_df["xrdint"] = output_df["xrd0"] / avg_at

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["roa"]
        .median()
        .rename(columns={"roa": "md_roa"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["cfroa"]
        .median()
        .rename(columns={"cfroa": "md_cfroa"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["oancf"]
        .median()
        .rename(columns={"oancf": "md_oancf"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["xrdint"]
        .median()
        .rename(columns={"xrdint": "md_xrdint"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["capxint"]
        .median()
        .rename(columns={"capxint": "md_capxint"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    industry_temp_df = (
        output_df.groupby(["fyear", "ffi49"], as_index=False)["xadint"]
        .median()
        .rename(columns={"xadint": "md_xadint"})
    )
    output_df = pd.merge(
        output_df,
        industry_temp_df,
        how="left",
        on=["fyear", "ffi49"],
    ).reset_index(drop=True)

    output_df["m1"] = np.where(
        output_df["roa"] > output_df["md_roa"],
        1,
        0,
    )

    output_df["m2"] = np.where(
        output_df["cfroa"] > output_df["md_cfroa"],
        1,
        0,
    )

    output_df["m3"] = np.where(
        output_df["oancf"] > output_df["md_oancf"],
        1,
        0,
    )

    output_df["m4"] = np.where(
        output_df["xrdint"] > output_df["md_xrdint"],
        1,
        0,
    )

    output_df["m5"] = np.where(
        output_df["capxint"] > output_df["md_capxint"],
        1,
        0,
    )

    output_df["m6"] = np.where(
        output_df["xadint"] > output_df["md_xadint"],
        1,
        0,
    )

    return output_df