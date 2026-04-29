from __future__ import annotations

import pandas as pd


def add_grltnoa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quarterly growth in long-term net operating assets variable.
    """

    output_df: pd.DataFrame = df.copy()

    required_cols: list[str] = [
        "permno",
        "rectq",
        "invtq",
        "ppentq",
        "acoq",
        "intanq",
        "aoq",
        "apq",
        "lcoq",
        "loq",
        "atq",
        "dpq4",
    ]

    missing_cols: list[str] = [
        col for col in required_cols if col not in output_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for quarterly grltnoa: {missing_cols}"
        )

    output_df["rectq_l4"] = output_df.groupby("permno")["rectq"].shift(4)
    output_df["acoq_l4"] = output_df.groupby("permno")["acoq"].shift(4)
    output_df["apq_l4"] = output_df.groupby("permno")["apq"].shift(4)
    output_df["lcoq_l4"] = output_df.groupby("permno")["lcoq"].shift(4)
    output_df["loq_l4"] = output_df.groupby("permno")["loq"].shift(4)
    output_df["invtq_l4"] = output_df.groupby("permno")["invtq"].shift(4)
    output_df["ppentq_l4"] = output_df.groupby("permno")["ppentq"].shift(4)
    output_df["atq_l4"] = output_df.groupby("permno")["atq"].shift(4)

    current_noa_components = (
        output_df["rectq"]
        + output_df["invtq"]
        + output_df["ppentq"]
        + output_df["acoq"]
        + output_df["intanq"]
        + output_df["aoq"]
        - output_df["apq"]
        - output_df["lcoq"]
        - output_df["loq"]
    )

    lagged_noa_components = (
        output_df["rectq_l4"]
        + output_df["invtq_l4"]
        + output_df["ppentq_l4"]
        + output_df["acoq_l4"]
        - output_df["apq_l4"]
        - output_df["lcoq_l4"]
        - output_df["loq_l4"]
    )

    working_capital_accruals = (
        output_df["rectq"]
        - output_df["rectq_l4"]
        + output_df["invtq"]
        - output_df["invtq_l4"]
        + output_df["acoq"]
        - (
            output_df["apq"]
            - output_df["apq_l4"]
            + output_df["lcoq"]
            - output_df["lcoq_l4"]
        )
        - output_df["dpq4"]
    )

    output_df["grltnoa"] = (
        current_noa_components
        - lagged_noa_components
        - working_capital_accruals
    ) / (
        (output_df["atq"] + output_df["atq_l4"]) / 2
    )

    return output_df