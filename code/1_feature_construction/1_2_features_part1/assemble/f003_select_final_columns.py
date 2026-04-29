from __future__ import annotations

import pandas as pd


ANNUAL_FINAL_COLUMNS: list[str] = [
    "cusip",
    "ncusip",
    "gvkey",
    "permno",
    "exchcd",
    "shrcd",
    "datadate",
    "jdate",
    "ticker",
    "conm",
    "comnam",
    "prc",
    "shrout",
    "sic",
    "ret",
    "retx",
    "retadj",
    "acc",
    "agr",
    "bm",
    "cfp",
    "ep",
    "ni",
    "op",
    "rsup",
    "cash",
    "chcsho",
    "rd",
    "cashdebt",
    "pctacc",
    "gma",
    "lev",
    "rdm",
    "adm",
    "sgr",
    "sp",
    "invest",
    "roe",
    "rd_sale",
    "lgr",
    "roa",
    "depr",
    "egr",
    "chato",
    "chtx",
    "noa",
    "rna",
    "pm",
    "ato",
    "dy",
    "roic",
    "chinv",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchgm_pchsale",
    "pchsale_pchxsga",
    "pchdepr",
    "chadv",
    "pchcapx",
    "grcapx",
    "grGW",
    "currat",
    "pchcurrat",
    "quick",
    "pchquick",
    "salecash",
    "salerec",
    "saleinv",
    "pchsaleinv",
    "realestate",
    "obklg",
    "chobklg",
    "grltnoa",
    "conv",
    "chdrc",
    "rdbias",
    "operprof",
    "capxint",
    "xadint",
    "chpm",
    "ala",
    "alm",
    "mom1m",
    "mom6m",
    "mom12m",
    "mom60m",
    "mom36m",
    "seas1a",
    "me",
    "hire",
    "herf",
    "bm_ia",
    "me_ia",
    "turn",
    "dolvol",
    "absacc",
    "age",
    "cashpr",
    "chatoia",
    "chempia",
    "chmom",
    "chpmia",
    "convind",
    "divi",
    "divo",
    "secured",
    "securedind",
    "sin",
    "cfp_ia",
    "indmom",
    "pchcapx_ia",
    "tang",
    "tb",
    "m1",
    "m2",
    "m3",
    "m4",
    "m5",
    "m6",
]


QUARTERLY_FINAL_COLUMNS: list[str] = [
    "gvkey",
    "permno",
    "datadate",
    "jdate",
    "sic",
    "exchcd",
    "shrcd",
    "ticker",
    "conm",
    "comnam",
    "prc",
    "shrout",
    "ret",
    "retx",
    "retadj",
    "acc",
    "bm",
    "cfp",
    "ep",
    "agr",
    "ni",
    "ope",
    "opa",
    "cop",
    "cash",
    "chcsho",
    "rd",
    "cashdebt",
    "pctacc",
    "gma",
    "lev",
    "rdm",
    "sgr",
    "sp",
    "invest",
    "rd_sale",
    "lgr",
    "roa",
    "depr",
    "egr",
    "roe",
    "chato",
    "chpm",
    "chtx",
    "noa",
    "rna",
    "pm",
    "ato",
    "stdcf",
    "grltnoa",
    "ala",
    "alm",
    "rsup",
    "stdacc",
    "sgrvol",
    "roavol",
    "scf",
    "cinvest",
    "mom1m",
    "mom6m",
    "mom12m",
    "mom60m",
    "mom36m",
    "seas1a",
    "me",
    "pscore",
    "nincr",
    "cfp_ia",
    "bm_ia",
    "me_ia",
    "chatoia",
    "chmom",
    "turn",
    "dolvol",
    "cashpr",
    "indmom",
    "m7",
    "m8",
]


def _select_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Select final columns from a feature dataframe.
    """

    missing_cols: list[str] = [
        col for col in columns if col not in df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing final columns for {dataset_name}: {missing_cols}"
        )

    output_df: pd.DataFrame = df[columns].copy()
    output_df = output_df.reset_index(drop=True)

    return output_df


def select_annual_final_columns(
    annual_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select final annual feature columns.
    """

    return _select_columns(
        df=annual_df,
        columns=ANNUAL_FINAL_COLUMNS,
        dataset_name="annual",
    )


def select_quarterly_final_columns(
    quarterly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select final quarterly feature columns.
    """

    return _select_columns(
        df=quarterly_df,
        columns=QUARTERLY_FINAL_COLUMNS,
        dataset_name="quarterly",
    )


def select_final_columns(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select final annual and quarterly feature columns.
    """

    annual_output_df: pd.DataFrame = select_annual_final_columns(
        annual_df=annual_df,
    )

    quarterly_output_df: pd.DataFrame = select_quarterly_final_columns(
        quarterly_df=quarterly_df,
    )

    return annual_output_df, quarterly_output_df