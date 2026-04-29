from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from tqdm import tqdm


INPUT_DIR: Path = Path("../../../data/feature_construction/1_3_features_part2")
OUTPUT_DIR: Path = Path("../../../data/feature_construction/1_4_rank_feature")

CHARS_A_RAW_PATH: Path = INPUT_DIR / "chars_a_raw.feather"
CHARS_Q_RAW_PATH: Path = INPUT_DIR / "chars_q_raw.feather"

CHARS_RAW_NO_IMPUTE_PATH: Path = OUTPUT_DIR / "chars_raw_no_impute.feather"
CHARS_RAW_IMPUTED_PATH: Path = OUTPUT_DIR / "chars_raw_imputed.feather"
CHARS_RANK_NO_IMPUTE_PATH: Path = OUTPUT_DIR / "chars_rank_no_impute.feather"
CHARS_RANK_IMPUTED_PATH: Path = OUTPUT_DIR / "chars_rank_imputed.feather"

CHARS_SUMMARY_PATH: Path = Path(
    "../../../data/1_data_processing/1_rank_data/input/chars_summary.csv"
)
RANKED_DATA_OUTPUT_PATH: Path = OUTPUT_DIR / "ranked_data.pkl.gz"
RANKED_BY_YEAR_OUTPUT_DIR: Path = OUTPUT_DIR / "by_year"


OBS_VAR_LIST: list[str] = [
    "gvkey",
    "permno",
    "jdate",
    "ticker",
    "conm",
    "comnam",
    "sic",
    "ret",
    "retx",
    "retadj",
    "exchcd",
    "shrcd",
    "prc",
    "shrout",
]


ACCOUNTING_VAR_LIST: list[str] = [
    "datadate",
    "acc",
    "bm",
    "agr",
    "alm",
    "ato",
    "cash",
    "cashdebt",
    "cfp",
    "chcsho",
    "chtx",
    "depr",
    "ep",
    "gma",
    "grltnoa",
    "lev",
    "lgr",
    "ni",
    "noa",
    "op",
    "pctacc",
    "pm",
    "rd_sale",
    "rdm",
    "rna",
    "roa",
    "roe",
    "rsup",
    "sgr",
    "sp",
    "me_ia",
    "bm_ia",
    "cashpr",
    "cfp_ia",
    "chatoia",
    "egr",
    "invest",
    "chmom",
    "rd",
]


A_ONLY_LIST: list[str] = [
    "adm",
    "herf",
    "hire",
    "absacc",
    "age",
    "chempia",
    "chinv",
    "convind",
    "currat",
    "divi",
    "divo",
    "grcapx",
    "pchcapx_ia",
    "pchcurrat",
    "pchdepr",
    "pchgm_pchsale",
    "pchquick",
    "pchsale_pchinvt",
    "pchsale_pchrect",
    "pchsale_pchxsga",
    "pchsaleinv",
    "quick",
    "realestate",
    "roic",
    "salecash",
    "salerec",
    "saleinv",
    "secured",
    "securedind",
    "sin",
    "tang",
    "tb",
    "chpmia",
]


Q_ONLY_LIST: list[str] = [
    "abr",
    "sue",
    "cinvest",
    "nincr",
    "pscore",
    "roavol",
    "stdacc",
    "stdcf",
]


M_VAR_LIST: list[str] = [
    "baspread",
    "beta",
    "ill",
    "maxret",
    "mom12m",
    "mom1m",
    "mom36m",
    "mom60m",
    "mom6m",
    "re",
    "rvar_capm",
    "rvar_ff3",
    "rvar_mean",
    "seas1a",
    "std_dolvol",
    "std_turn",
    "zerotrade",
    "me",
    "dy",
    "turn",
    "dolvol",
    "indmom",
]


FINAL_OBS_COLS: list[str] = [
    "gvkey",
    "permno",
    "date",
    "ticker",
    "conm",
    "comnam",
    "sic",
    "ret",
    "exchcd",
    "shrcd",
    "prc",
    "shrout",
]


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


def ensure_columns(
    df: pd.DataFrame,
    columns: list[str],
    df_name: str,
) -> pd.DataFrame:
    """
    Ensure columns exist. Missing feature columns are created as NaN.
    """

    output_df: pd.DataFrame = df.copy()

    for col in columns:
        if col not in output_df.columns:
            print(f"WARNING: {df_name} missing column {col}; filling with NaN.")
            output_df[col] = np.nan

    return output_df


def prepare_raw_panel(
    df: pd.DataFrame,
    panel_name: str,
) -> pd.DataFrame:
    """
    Clean raw annual or quarterly panel before final merge.
    """

    output_df: pd.DataFrame = df.copy()

    if "permno" not in output_df.columns:
        raise KeyError(f"{panel_name} missing permno.")

    if "jdate" not in output_df.columns:
        raise KeyError(f"{panel_name} missing jdate.")

    output_df = output_df.dropna(subset=["permno"]).copy()
    output_df["permno"] = output_df["permno"].astype(int)
    output_df["jdate"] = pd.to_datetime(output_df["jdate"])

    if "gvkey" in output_df.columns:
        output_df["gvkey"] = pd.to_numeric(
            output_df["gvkey"],
            errors="coerce",
        )

    output_df = (
        output_df.sort_values(by=["permno", "jdate"])
        .drop_duplicates(subset=["permno", "jdate"], keep="last")
        .reset_index(drop=True)
    )

    return output_df


def merge_annual_quarterly_raw(
    chars_a: pd.DataFrame,
    chars_q: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge annual and quarterly raw panels into one all-stock raw panel.
    """

    chars_a = prepare_raw_panel(
        df=chars_a,
        panel_name="annual raw",
    )

    chars_q = prepare_raw_panel(
        df=chars_q,
        panel_name="quarterly raw",
    )

    annual_required_cols = (
        OBS_VAR_LIST
        + ACCOUNTING_VAR_LIST
        + A_ONLY_LIST
        + M_VAR_LIST
    )

    quarterly_required_cols = (
        OBS_VAR_LIST
        + ACCOUNTING_VAR_LIST
        + Q_ONLY_LIST
    )

    chars_a = ensure_columns(
        df=chars_a,
        columns=annual_required_cols,
        df_name="chars_a_raw",
    )

    chars_q = ensure_columns(
        df=chars_q,
        columns=quarterly_required_cols,
        df_name="chars_q_raw",
    )

    a_var_list: list[str] = [
        "a_" + col for col in ACCOUNTING_VAR_LIST
    ]

    q_var_list: list[str] = [
        "q_" + col for col in ACCOUNTING_VAR_LIST
    ]

    df_a: pd.DataFrame = chars_a[
        OBS_VAR_LIST
        + ACCOUNTING_VAR_LIST
        + A_ONLY_LIST
        + M_VAR_LIST
    ].copy()

    df_a.columns = (
        OBS_VAR_LIST
        + a_var_list
        + A_ONLY_LIST
        + M_VAR_LIST
    )

    df_a = df_a.sort_values(by=OBS_VAR_LIST).reset_index(drop=True)

    df_q: pd.DataFrame = chars_q[
        OBS_VAR_LIST
        + ACCOUNTING_VAR_LIST
        + Q_ONLY_LIST
    ].copy()

    df_q.columns = (
        OBS_VAR_LIST
        + q_var_list
        + Q_ONLY_LIST
    )

    drop_q_info_cols: list[str] = [
        "sic",
        "ret",
        "retx",
        "retadj",
        "exchcd",
        "shrcd",
        "ticker",
        "conm",
        "comnam",
        "prc",
        "shrout",
    ]

    df_q = df_q.drop(
        columns=[
            col for col in drop_q_info_cols if col in df_q.columns
        ]
    )

    df: pd.DataFrame = df_a.merge(
        df_q,
        how="left",
        on=[
            "gvkey",
            "jdate",
            "permno",
        ],
    )

    for var_name in tqdm(ACCOUNTING_VAR_LIST[1:]):
        print(f"processing {var_name}")

        annual_col = "a_" + var_name
        quarterly_col = "q_" + var_name

        tmp1 = "tmp1_" + var_name
        tmp2 = "tmp2_" + var_name
        tmp3 = "tmp3_" + var_name
        tmp4 = "tmp4_" + var_name
        tmp5 = "tmp5_" + var_name

        df[tmp1] = np.where(df[annual_col].isna(), False, True)
        df[tmp2] = np.where(df[quarterly_col].isna(), False, True)
        df[tmp3] = df[tmp1] & df[tmp2]

        # If quarterly datadate is older than annual datadate, use annual;
        # otherwise use quarterly.
        df[tmp4] = np.where(
            df["q_datadate"] < df["a_datadate"],
            df[annual_col],
            df[quarterly_col],
        )

        # If only one side is available, use the available side.
        df[tmp5] = np.where(
            df[tmp1],
            df[annual_col],
            df[quarterly_col],
        )

        df[var_name] = np.where(
            df[tmp3],
            df[tmp4],
            df[tmp5],
        )

        df = df.drop(
            columns=[
                annual_col,
                quarterly_col,
                tmp1,
                tmp2,
                tmp3,
                tmp4,
                tmp5,
            ]
        )

    df = df.drop(
        columns=[
            col for col in ["a_datadate", "q_datadate"] if col in df.columns
        ]
    )

    return df


def finalize_raw_no_impute(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Finalize raw no-impute panel: shift return to next month and clean SIC.
    """

    output_df: pd.DataFrame = df.copy()

    output_df = output_df.drop(
        columns=[
            col for col in ["ret", "retx"] if col in output_df.columns
        ]
    )

    output_df = output_df.rename(
        columns={
            "retadj": "ret",
        }
    )

    output_df = output_df.sort_values(
        by=[
            "permno",
            "jdate",
        ]
    ).reset_index(drop=True)

    # Predictor at jdate predicts next-month adjusted return.
    output_df["ret"] = output_df.groupby("permno")["ret"].shift(-1)
    output_df["date"] = output_df.groupby("permno")["jdate"].shift(-1)

    output_df = output_df.drop(columns=["jdate"])

    output_df = output_df.dropna(
        subset=[
            "ret",
        ]
    ).reset_index(drop=True)

    output_df = output_df.replace(
        [
            -np.inf,
            np.inf,
        ],
        np.nan,
    )

    output_df["sic"] = output_df.groupby("permno")["sic"].ffill()
    output_df["sic"] = output_df["sic"].fillna(0)
    output_df["sic"] = output_df["sic"].astype(int)

    output_df["date"] = pd.to_datetime(output_df["date"])

    return output_df


def assign_ffi49_from_sic(
    df: pd.DataFrame,
    sic_col: str = "sic",
) -> pd.Series:
    """
    Assign Fama-French 49 industry code from SIC.
    """

    sic = pd.to_numeric(
        df[sic_col],
        errors="coerce",
    ).fillna(0).astype(int)

    ffi = pd.Series(
        np.nan,
        index=df.index,
        dtype="float64",
    )

    def set_ffi(code: int, ranges: list[tuple[int, int]]) -> None:
        mask = pd.Series(False, index=df.index)
        for low, high in ranges:
            mask = mask | ((sic >= low) & (sic <= high))
        ffi.loc[mask] = code

    set_ffi(1, [(100, 199), (200, 299), (700, 799), (910, 919), (2048, 2048)])
    set_ffi(2, [(2000, 2046), (2050, 2063), (2070, 2079), (2090, 2092), (2095, 2095), (2098, 2099)])
    set_ffi(3, [(2064, 2068), (2086, 2087), (2096, 2097)])
    set_ffi(4, [(2080, 2085)])
    set_ffi(5, [(2100, 2199)])
    set_ffi(6, [(920, 999), (3650, 3652), (3732, 3732), (3930, 3949)])
    set_ffi(7, [(7800, 7833), (7840, 7841), (7900, 7911), (7920, 7933), (7940, 7949), (7980, 7980), (7990, 7999)])
    set_ffi(8, [(2700, 2749), (2770, 2799)])
    set_ffi(9, [(2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2844), (3160, 3199), (3229, 3231), (3260, 3260), (3262, 3263), (3269, 3269), (3630, 3639), (3750, 3751), (3800, 3800), (3860, 3873), (3910, 3911), (3914, 3915), (3960, 3962), (3991, 3991), (3995, 3995)])
    set_ffi(10, [(2300, 2390), (3020, 3021), (3100, 3111), (3130, 3159), (3963, 3965)])
    set_ffi(11, [(8000, 8099)])
    set_ffi(12, [(3693, 3693), (3840, 3851)])
    set_ffi(13, [(2830, 2836)])
    set_ffi(14, [(2800, 2829), (2850, 2899)])
    set_ffi(15, [(3031, 3031), (3041, 3041), (3050, 3099)])
    set_ffi(16, [(2200, 2284), (2290, 2295), (2297, 2299), (2393, 2395), (2397, 2399)])
    set_ffi(17, [(800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661), (2950, 2952), (3200, 3211), (3240, 3259), (3261, 3261), (3264, 3264), (3270, 3299), (3420, 3442), (3446, 3452), (3490, 3499), (3996, 3996)])
    set_ffi(18, [(1500, 1549), (1600, 1699), (1700, 1799)])
    set_ffi(19, [(3300, 3369), (3390, 3399)])
    set_ffi(20, [(3400, 3412), (3443, 3444), (3460, 3479)])
    set_ffi(21, [(3510, 3536), (3540, 3569), (3580, 3599)])
    set_ffi(22, [(3600, 3621), (3623, 3629), (3640, 3646), (3648, 3649), (3660, 3660), (3690, 3692), (3699, 3699)])
    set_ffi(23, [(2296, 2296), (2396, 2396), (3010, 3011), (3537, 3537), (3647, 3647), (3694, 3694), (3700, 3716), (3790, 3792), (3799, 3799)])
    set_ffi(24, [(3720, 3729)])
    set_ffi(25, [(3730, 3731), (3740, 3743)])
    set_ffi(26, [(3760, 3769), (3795, 3795), (3480, 3489)])
    set_ffi(27, [(1040, 1049)])
    set_ffi(28, [(1000, 1039), (1060, 1099), (1400, 1499)])
    set_ffi(29, [(1200, 1299)])
    set_ffi(30, [(1310, 1389), (2900, 2912), (2990, 2999)])
    set_ffi(31, [(4900, 4942)])
    set_ffi(32, [(4800, 4899)])
    set_ffi(33, [(7020, 7021), (7030, 7039), (7200, 7212), (7215, 7299), (7395, 7395), (7500, 7500), (7520, 7549), (7600, 7699), (8100, 8199), (8200, 8299), (8300, 8399), (8400, 8499), (8600, 8699), (8800, 8899), (7510, 7515)])
    set_ffi(34, [(2750, 2769), (3993, 3993), (7213, 7213), (7300, 7372), (7374, 7394), (7397, 7397), (7399, 7399), (7519, 7519), (8700, 8748), (8900, 8999)])
    set_ffi(35, [(3570, 3579), (3680, 3689), (3695, 3695), (7373, 7373)])
    set_ffi(36, [(3622, 3622), (3661, 3679), (3810, 3810), (3812, 3812)])
    set_ffi(37, [(3811, 3811), (3820, 3830)])
    set_ffi(38, [(2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955)])
    set_ffi(39, [(2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412)])
    set_ffi(40, [(4000, 4099), (4100, 4199), (4200, 4299), (4400, 4499), (4500, 4599), (4600, 4699), (4700, 4799)])
    set_ffi(41, [(5000, 5099), (5100, 5199)])
    set_ffi(42, [(5200, 5299), (5300, 5399), (5400, 5499), (5500, 5599), (5600, 5699), (5700, 5736), (5900, 5999)])
    set_ffi(43, [(5800, 5813), (5890, 5890), (7000, 7019), (7040, 7049), (7214, 7214)])
    set_ffi(44, [(6000, 6099)])
    set_ffi(45, [(6100, 6199)])
    set_ffi(46, [(6300, 6411)])
    set_ffi(47, [(6500, 6553)])
    set_ffi(48, [(6200, 6299), (6700, 6799)])
    set_ffi(49, [(4950, 4961), (4970, 4971), (4990, 4991)])

    return ffi


def get_feature_columns_for_imputation(
    df: pd.DataFrame,
    not_fill_col: list[str],
) -> list[str]:
    """
    Get numeric columns eligible for imputation.
    """

    excluded = set(
        col for col in not_fill_col if col in df.columns
    )

    excluded.update(
        [
            "date",
            "ffi49",
            "ticker",
            "conm",
            "comnam",
        ]
    )

    feature_cols: list[str] = []

    for col in df.columns:
        if col in excluded:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def fillna_industry(
    df: pd.DataFrame,
    method: str = "median",
    ffi: int = 49,
    not_fill_col: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fill missing values by date-industry median or mean.
    """

    output_df: pd.DataFrame = df.copy()

    if not_fill_col is None:
        not_fill_col = []

    industry_col = f"ffi{ffi}"

    if industry_col not in output_df.columns:
        raise KeyError(f"Missing industry column: {industry_col}")

    feature_cols = get_feature_columns_for_imputation(
        df=output_df,
        not_fill_col=not_fill_col,
    )

    if method == "median":
        fill_values = output_df.groupby(["date", industry_col])[feature_cols].transform("median")
    elif method == "mean":
        fill_values = output_df.groupby(["date", industry_col])[feature_cols].transform("mean")
    else:
        raise ValueError("method must be 'median' or 'mean'.")

    output_df[feature_cols] = output_df[feature_cols].fillna(fill_values)

    return output_df


def fillna_all(
    df: pd.DataFrame,
    method: str = "median",
    not_fill_col: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fill remaining missing values by date-level median or mean.
    """

    output_df: pd.DataFrame = df.copy()

    if not_fill_col is None:
        not_fill_col = []

    feature_cols = get_feature_columns_for_imputation(
        df=output_df,
        not_fill_col=not_fill_col,
    )

    if method == "median":
        fill_values = output_df.groupby("date")[feature_cols].transform("median")
    elif method == "mean":
        fill_values = output_df.groupby("date")[feature_cols].transform("mean")
    else:
        raise ValueError("method must be 'median' or 'mean'.")

    output_df[feature_cols] = output_df[feature_cols].fillna(fill_values)

    return output_df


def build_raw_imputed(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build imputed raw feature panel.
    """

    output_df: pd.DataFrame = df.copy()

    output_df["date"] = pd.to_datetime(output_df["date"])

    output_df["ffi49"] = assign_ffi49_from_sic(output_df)
    output_df["ffi49"] = output_df["ffi49"].fillna(49)
    output_df["ffi49"] = output_df["ffi49"].astype(int)

    output_df = output_df.replace(
        [
            -np.inf,
            np.inf,
        ],
        np.nan,
    )

    output_df = fillna_industry(
        df=output_df,
        method="median",
        ffi=49,
        not_fill_col=FINAL_OBS_COLS,
    )

    output_df = fillna_all(
        df=output_df,
        method="median",
        not_fill_col=FINAL_OBS_COLS,
    )

    if "re" in output_df.columns:
        output_df["re"] = output_df["re"].fillna(0)

    return output_df


def standardize_rank_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cross-sectional rank variables by date.

    Rank scale follows the old code's standardize convention:
        percentile rank * 2 - 1
    """

    output_df: pd.DataFrame = df.copy()
    output_df["date"] = pd.to_datetime(output_df["date"])

    excluded_cols = set(FINAL_OBS_COLS)
    excluded_cols.update(
        [
            "lag_me",
            "ffi49",
            "log_me",
        ]
    )

    candidate_cols: list[str] = []

    for col in output_df.columns:
        if col in excluded_cols:
            continue

        if col.startswith("rank_"):
            continue

        if pd.api.types.is_numeric_dtype(output_df[col]):
            candidate_cols.append(col)

    for col in tqdm(candidate_cols):
        rank_col = "rank_" + col

        output_df[rank_col] = (
            output_df.groupby("date")[col]
            .rank(method="average", pct=True)
            * 2
            - 1
        )

    return output_df


def build_rank_no_impute(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ranked features from no-impute raw data.
    """

    output_df: pd.DataFrame = df.copy()

    output_df["lag_me"] = output_df["me"]

    if "bm" in output_df.columns:
        output_df["bm"] = np.where(
            output_df["bm"] < 0,
            np.nan,
            output_df["bm"],
        )

    output_df = standardize_rank_features(output_df)

    output_df["log_me"] = np.log(output_df["lag_me"])

    output_df = output_df.replace(
        [
            -np.inf,
            np.inf,
        ],
        0,
    )

    return output_df


def build_rank_imputed_from_rank_no_impute(
    df_rank_no_impute: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build rank-imputed version following the original code:
    fill missing rank_* variables with 0.
    """

    output_df: pd.DataFrame = df_rank_no_impute.copy()

    rank_cols: list[str] = [
        col for col in output_df.columns if col.startswith("rank_")
    ]

    output_df[rank_cols] = output_df[rank_cols].fillna(0)

    return output_df


def cross_sectional_rank_0_1(x: pd.Series) -> pd.Series:
    """
    Cross-sectional rank one column to [0, 1] within one month.
    """

    notna = x.notna()
    n = notna.sum()

    out = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    if n == 0:
        return out

    if n == 1:
        out.loc[notna] = 0.5
        return out

    ranks = x.loc[notna].rank(method="average")
    out.loc[notna] = (ranks - 1) / (n - 1)

    return out


def build_ranked_data_for_downstream_pipeline(
    chars_raw_imputed: pd.DataFrame,
    chars_summary_path: Path = CHARS_SUMMARY_PATH,
    ranked_data_output_path: Path = RANKED_DATA_OUTPUT_PATH,
    by_year_output_dir: Path = RANKED_BY_YEAR_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Build downstream ranked_data.pkl.gz and yearly parquet files.

    Output columns:
        permno, date, ret, size, {feature}_rank columns.
    """

    df: pd.DataFrame = chars_raw_imputed.copy()

    required_base_cols: list[str] = [
        "permno",
        "date",
        "ret",
        "prc",
        "shrout",
    ]

    missing_base_cols: list[str] = [
        col for col in required_base_cols if col not in df.columns
    ]

    if missing_base_cols:
        raise KeyError(
            f"Missing required base columns for downstream ranked data: {missing_base_cols}"
        )

    if not chars_summary_path.exists():
        raise FileNotFoundError(
            f"Missing chars summary file: {chars_summary_path}"
        )

    df["size"] = (df["prc"] * df["shrout"]).abs()

    chars_summary_df: pd.DataFrame = pd.read_csv(chars_summary_path)

    if "Acronym" not in chars_summary_df.columns:
        raise KeyError(
            f"Missing 'Acronym' column in chars summary: {chars_summary_path}"
        )

    acronyms: list[str] = (
        chars_summary_df["Acronym"]
        .astype(str)
        .str.rstrip()
        .tolist()
    )

    rank_features: list[str] = [
        "size",
        *[col for col in acronyms if col != "size"],
    ]

    missing_cols: list[str] = [
        col for col in rank_features if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in raw imputed data for downstream ranking: {missing_cols}"
        )

    # Match the downstream ranking code: zero values are treated as missing.
    df = df.replace(0, np.nan)

    keep_raw_cols: list[str] = [
        "permno",
        "date",
        "ret",
        "size",
        *[col for col in rank_features if col != "size"],
    ]

    df = df[keep_raw_cols].copy()

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values(["date", "permno"])
        .reset_index(drop=True)
    )

    for col in rank_features:
        rank_col = f"{col}_rank"

        df[rank_col] = (
            df.groupby("date", group_keys=False)[col]
            .apply(cross_sectional_rank_0_1)
        )

    rank_cols: list[str] = [
        f"{col}_rank" for col in rank_features
    ]

    final_cols: list[str] = [
        "permno",
        "date",
        "ret",
        "size",
        *rank_cols,
    ]

    df = df[final_cols].copy()

    ranked_data_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_pickle(
        ranked_data_output_path,
    )

    print(f"SAVED: {ranked_data_output_path}")

    by_year_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    years: list[int] = sorted(
        df["date"].dt.year.dropna().astype(int).unique().tolist()
    )

    for year in years:
        df_year = df[df["date"].dt.year == year].copy()

        year_path = by_year_output_dir / f"{year}.parquet"

        df_year.to_parquet(
            year_path,
            index=False,
        )

        print(f"Saved {year}: {df_year.shape}")

    print("Saved downstream ranked data successfully.")
    print(f"Final downstream ranked shape: {df.shape}")
    print(f"Number of rank features: {len(rank_features)}")

    return df


def generate_rank_features() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Generate final raw and ranked feature datasets.
    """

    chars_q: pd.DataFrame = read_feather(CHARS_Q_RAW_PATH)
    chars_a: pd.DataFrame = read_feather(CHARS_A_RAW_PATH)

    merged_df: pd.DataFrame = merge_annual_quarterly_raw(
        chars_a=chars_a,
        chars_q=chars_q,
    )

    chars_raw_no_impute: pd.DataFrame = finalize_raw_no_impute(
        df=merged_df,
    )

    save_feather(
        chars_raw_no_impute,
        CHARS_RAW_NO_IMPUTE_PATH,
    )

    chars_raw_imputed: pd.DataFrame = build_raw_imputed(
        df=chars_raw_no_impute,
    )

    save_feather(
        chars_raw_imputed,
        CHARS_RAW_IMPUTED_PATH,
    )

    chars_rank_no_impute: pd.DataFrame = build_rank_no_impute(
        df=chars_raw_no_impute,
    )

    save_feather(
        chars_rank_no_impute,
        CHARS_RANK_NO_IMPUTE_PATH,
    )

    chars_rank_imputed: pd.DataFrame = build_rank_imputed_from_rank_no_impute(
        df_rank_no_impute=chars_rank_no_impute,
    )

    save_feather(
        chars_rank_imputed,
        CHARS_RANK_IMPUTED_PATH,
    )

    downstream_ranked_df: pd.DataFrame = build_ranked_data_for_downstream_pipeline(
        chars_raw_imputed=chars_raw_imputed,
        chars_summary_path=CHARS_SUMMARY_PATH,
        ranked_data_output_path=RANKED_DATA_OUTPUT_PATH,
        by_year_output_dir=RANKED_BY_YEAR_OUTPUT_DIR,
    )

    return (
        chars_raw_no_impute,
        chars_raw_imputed,
        chars_rank_no_impute,
        chars_rank_imputed,
        downstream_ranked_df,
    )


def main() -> None:
    """
    Main entry point.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        chars_raw_no_impute,
        chars_raw_imputed,
        chars_rank_no_impute,
        chars_rank_imputed,
        downstream_ranked_df,
    ) = generate_rank_features()

    print("\nFeature ranking / imputation finished.")
    print(f"chars_raw_no_impute shape: {chars_raw_no_impute.shape}")
    print(f"chars_raw_imputed shape:   {chars_raw_imputed.shape}")
    print(f"chars_rank_no_impute shape:{chars_rank_no_impute.shape}")
    print(f"chars_rank_imputed shape:  {chars_rank_imputed.shape}")
    print(f"downstream ranked shape:   {downstream_ranked_df.shape}")
    print(f"Output directory:          {OUTPUT_DIR}")


if __name__ == "__main__":
    main()