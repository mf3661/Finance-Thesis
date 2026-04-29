from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# =========================================================
# paths
# =========================================================
BY_YEAR_DIR = "../../../data/feature_construction/1_4_rank_feature/by_year/impute"
CHARS_PATH = "../../../data/common/chars_summary.csv"

OUTPUT_DIR = "../../../data/3_ridge/3_1_run_ridge/3_1_1_ridge/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# fixed columns
# =========================================================
DATE_COL = "date"
ID_COL = "permno"
RET_COL = "ret"
WEIGHT_COL = "size"


# =========================================================
# rolling setup
# =========================================================
TRAIN_YEARS = 20
VAL_YEARS = 10

OOS_START = 1981
OOS_END = 2024

# long top 20%, short bottom 20%
TOP_Q = 0.8
BOT_Q = 0.2


# =========================================================
# ridge alpha grid
# =========================================================
ALPHA_GRID = np.array([
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
    1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0
])


# =========================================================
# read characteristic names
# =========================================================
def load_feature_cols(chars_path: str) -> list[str]:
    """
    Use ranked characteristic columns from 1_4_rank_feature/by_year/impute.

    Example:
        size -> size_rank
        bm   -> bm_rank
    """
    chars = pd.read_csv(chars_path)
    acronyms = chars["Acronym"].astype(str).str.rstrip().tolist()

    feature_cols = [f"{c}_rank" for c in acronyms]
    feature_cols = list(dict.fromkeys(feature_cols))

    return feature_cols


FEATURE_COLS_ALL = load_feature_cols(CHARS_PATH)


# =========================================================
# helper functions
# =========================================================
def load_years(
    year_list: list[int],
    by_year_dir: str,
    needed_cols: list[str],
) -> pd.DataFrame:
    dfs = []

    for year in year_list:
        path = os.path.join(by_year_dir, f"{year}.parquet")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing yearly file: {path}")

        df_year = pd.read_parquet(path, columns=needed_cols)
        dfs.append(df_year)

    out = pd.concat(dfs, axis=0, ignore_index=True)
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    return out


def clean_features_fill_zero(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Only clean X features:
    - numeric conversion
    - inf -> NaN
    - NaN -> 0

    Important:
        Do NOT use this for y / ret.
    """
    tmp = df.copy()

    for c in feature_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp[feature_cols] = tmp[feature_cols].replace([np.inf, -np.inf], np.nan)
    tmp[feature_cols] = tmp[feature_cols].fillna(0.0)

    return tmp


def clean_ret_drop_missing(
    df: pd.DataFrame,
    ret_col: str = RET_COL,
) -> pd.DataFrame:
    """
    Clean y / ret.

    y cannot be filled with 0.
    Missing or infinite ret must be dropped.
    """
    tmp = df.copy()

    tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")
    tmp[ret_col] = tmp[ret_col].replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[ret_col]).copy()

    return tmp


def get_nonzero_variance_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    var_series = df[feature_cols].var(axis=0, ddof=0)
    keep_features = var_series[var_series > 0].index.tolist()
    return keep_features


def prepare_train_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    ret_col: str = RET_COL,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Train set:
    - y / ret: numeric conversion, inf -> NaN, drop missing
    - X features: numeric conversion, inf/NaN -> 0
    - drop zero-variance feature columns

    Important:
        y / ret is NOT filled with zero.
    """
    needed = [ID_COL, DATE_COL, ret_col] + feature_cols
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

    tmp = clean_ret_drop_missing(tmp, ret_col=ret_col)
    tmp = clean_features_fill_zero(tmp, feature_cols=feature_cols)

    used_feature_cols = get_nonzero_variance_features(tmp, feature_cols)

    if len(used_feature_cols) == 0:
        raise ValueError("All feature columns have zero variance after cleaning.")

    X = tmp[used_feature_cols].to_numpy(dtype=float)
    y = tmp[ret_col].to_numpy(dtype=float)

    print(f"[DEBUG] train rows after dropping missing y: {len(tmp)}")
    print(f"[DEBUG] used feature count: {len(used_feature_cols)}")

    return X, y, used_feature_cols


def prepare_arrays_fixed_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    ret_col: str = RET_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validation / train+val:
    - y / ret: numeric conversion, inf -> NaN, drop missing
    - X features: use fixed feature_cols, inf/NaN -> 0
    - do not re-drop feature columns

    Important:
        y / ret is NOT filled with zero.
    """
    needed = [ID_COL, DATE_COL, ret_col] + feature_cols
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

    tmp = clean_ret_drop_missing(tmp, ret_col=ret_col)
    tmp = clean_features_fill_zero(tmp, feature_cols=feature_cols)

    X = tmp[feature_cols].to_numpy(dtype=float)
    y = tmp[ret_col].to_numpy(dtype=float)

    return X, y


def prepare_X_fixed_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    OOS prediction:
    - only prepare X
    - do not require ret
    - feature NaN/inf -> 0
    """
    needed = [ID_COL, DATE_COL] + feature_cols
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])
    tmp = clean_features_fill_zero(tmp, feature_cols=feature_cols)

    X = tmp[feature_cols].to_numpy(dtype=float)
    return X


def fit_ridge_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[Ridge, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        random_state=0,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_scores(
    model: Ridge,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return pred


def make_vw_ls_returns_and_stock_weights_from_scores(
    df_raw: pd.DataFrame,
    scores: np.ndarray,
    top_q: float = TOP_Q,
    bot_q: float = BOT_Q,
    weight_col: str = WEIGHT_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct value-weighted long-short returns and stock-level weights.

    Long:
        top 20% by score
        stock weight = + size_i / sum(size_i in long leg)

    Short:
        bottom 20% by score
        stock weight = - size_i / sum(size_i in short leg)

    Portfolio return:
        ls_ret = long_ret - short_ret

    Important:
        ret is not filled with 0. Stocks with missing ret are dropped
        when constructing realized OOS returns.
    """
    required_cols = [ID_COL, DATE_COL, RET_COL, weight_col]
    missing = [c for c in required_cols if c not in df_raw.columns]

    if missing:
        raise ValueError(f"Missing columns in df_raw: {missing}")

    if len(df_raw) != len(scores):
        raise ValueError(
            f"df_raw length and scores length do not match: "
            f"{len(df_raw)} vs {len(scores)}"
        )

    tmp = df_raw[required_cols].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])
    tmp[RET_COL] = pd.to_numeric(tmp[RET_COL], errors="coerce")
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")
    tmp["score"] = np.asarray(scores, dtype=float)

    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    # ret cannot be filled with zero.
    tmp = tmp.dropna(subset=[RET_COL, weight_col, "score"])
    tmp = tmp[tmp[weight_col] > 0].copy()

    ret_records = []
    weight_records = []

    for dt, g in tmp.groupby(DATE_COL):
        g = g.copy()

        if g.empty:
            continue

        top_cut = g["score"].quantile(top_q)
        bot_cut = g["score"].quantile(bot_q)

        long_g = g[g["score"] >= top_cut].copy()
        short_g = g[g["score"] <= bot_cut].copy()

        if len(long_g) == 0 or len(short_g) == 0:
            continue

        long_size_sum = float(long_g[weight_col].sum())
        short_size_sum = float(short_g[weight_col].sum())

        if long_size_sum <= 0 or short_size_sum <= 0:
            continue

        long_g["leg_weight"] = long_g[weight_col] / long_size_sum
        short_g["leg_weight"] = short_g[weight_col] / short_size_sum

        long_g["strategy_weight"] = long_g["leg_weight"]
        short_g["strategy_weight"] = -short_g["leg_weight"]

        long_ret = float(
            np.sum(
                long_g["leg_weight"].to_numpy(dtype=float)
                * long_g[RET_COL].to_numpy(dtype=float)
            )
        )
        short_ret = float(
            np.sum(
                short_g["leg_weight"].to_numpy(dtype=float)
                * short_g[RET_COL].to_numpy(dtype=float)
            )
        )
        ls_ret = long_ret - short_ret

        ret_records.append({
            DATE_COL: dt,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret,
            "n_long": int(len(long_g)),
            "n_short": int(len(short_g)),
            "long_size_sum": long_size_sum,
            "short_size_sum": short_size_sum,
            "long_weight_sum": float(long_g["leg_weight"].sum()),
            "short_weight_sum": float(short_g["leg_weight"].sum()),
            "gross_exposure": float(
                np.abs(long_g["strategy_weight"]).sum()
                + np.abs(short_g["strategy_weight"]).sum()
            ),
            "net_exposure": float(
                long_g["strategy_weight"].sum()
                + short_g["strategy_weight"].sum()
            ),
        })

        long_out = long_g[
            [ID_COL, DATE_COL, RET_COL, weight_col, "score", "leg_weight", "strategy_weight"]
        ].copy()
        long_out["side"] = "long"

        short_out = short_g[
            [ID_COL, DATE_COL, RET_COL, weight_col, "score", "leg_weight", "strategy_weight"]
        ].copy()
        short_out["side"] = "short"

        weight_records.append(long_out)
        weight_records.append(short_out)

    if len(ret_records) == 0:
        ret_df = pd.DataFrame(columns=[
            DATE_COL,
            "long_ret",
            "short_ret",
            "ls_ret",
            "n_long",
            "n_short",
            "long_size_sum",
            "short_size_sum",
            "long_weight_sum",
            "short_weight_sum",
            "gross_exposure",
            "net_exposure",
        ])
    else:
        ret_df = pd.DataFrame(ret_records).sort_values(DATE_COL).reset_index(drop=True)

    if len(weight_records) == 0:
        weight_df = pd.DataFrame(columns=[
            ID_COL,
            DATE_COL,
            RET_COL,
            weight_col,
            "score",
            "leg_weight",
            "strategy_weight",
            "side",
        ])
    else:
        weight_df = pd.concat(weight_records, axis=0, ignore_index=True)
        weight_df = weight_df.sort_values([DATE_COL, "side", ID_COL]).reset_index(drop=True)

    return ret_df, weight_df


def pick_best_alpha_by_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    alpha_grid: np.ndarray,
) -> tuple[float, float, list[str]]:
    """
    Validation objective:
        MSE, lower is better.

    Important:
        validation y / ret is not filled with zero.
    """
    best_alpha = None
    best_metric = np.inf
    best_features = None

    X_train, y_train, used_feature_cols = prepare_train_arrays(train_df, feature_cols)

    val_needed = [ID_COL, DATE_COL, RET_COL] + used_feature_cols
    val_sub = val_df[val_needed].copy()
    X_val, y_val = prepare_arrays_fixed_features(val_sub, used_feature_cols)

    for alpha in alpha_grid:
        try:
            model, scaler = fit_ridge_model(X_train, y_train, alpha=alpha)
            val_pred = predict_scores(model, scaler, X_val)

            metric = mean_squared_error(y_val, val_pred)

            print(f"[INFO] alpha={alpha:.6g}, val MSE={metric:.10f}")

            if metric < best_metric:
                best_metric = metric
                best_alpha = float(alpha)
                best_features = used_feature_cols

        except Exception as e:
            print(f"[WARNING] alpha={alpha} failed in validation: {e}")

    if best_alpha is None or best_features is None:
        raise ValueError("No valid alpha found in validation.")

    return best_alpha, float(best_metric), best_features


# =========================================================
# main
# =========================================================
def main(args: argparse.Namespace) -> None:
    oos_year = args.year

    if oos_year < OOS_START or oos_year > OOS_END:
        raise ValueError(f"year must be between {OOS_START} and {OOS_END}, got {oos_year}")

    train_start = oos_year - VAL_YEARS - TRAIN_YEARS
    train_end = oos_year - VAL_YEARS - 1

    val_start = oos_year - VAL_YEARS
    val_end = oos_year - 1

    oos_start = oos_year
    oos_end = oos_year

    train_years = list(range(train_start, train_end + 1))
    val_years = list(range(val_start, val_end + 1))
    oos_years = list(range(oos_start, oos_end + 1))

    # Need size for value-weighted long-short portfolio.
    needed_cols = [ID_COL, DATE_COL, RET_COL, WEIGHT_COL] + FEATURE_COLS_ALL

    print("=" * 80)
    print(f"[INFO] OOS year: {oos_year}")
    print(f"[INFO] Train years: {train_years[0]}-{train_years[-1]}")
    print(f"[INFO] Val years:   {val_years[0]}-{val_years[-1]}")
    print(f"[INFO] OOS years:   {oos_years[0]}-{oos_years[-1]}")
    print(f"[INFO] Input dir:    {BY_YEAR_DIR}")
    print(f"[INFO] Output dir:   {OUTPUT_DIR}")
    print(f"[INFO] Feature count from chars_summary: {len(FEATURE_COLS_ALL)}")
    print(f"[INFO] Long leg: top {int((1 - TOP_Q) * 100)}% by score")
    print(f"[INFO] Short leg: bottom {int(BOT_Q * 100)}% by score")
    print("[INFO] Portfolio weighting: value-weighted by size")
    print("[INFO] y / ret handling: drop missing, never fill with 0")

    # -------------------------
    # load data
    # -------------------------
    train_df = load_years(train_years, BY_YEAR_DIR, needed_cols)
    val_df = load_years(val_years, BY_YEAR_DIR, needed_cols)
    oos_df = load_years(oos_years, BY_YEAR_DIR, needed_cols)

    print(f"[INFO] train shape: {train_df.shape}")
    print(f"[INFO] val shape:   {val_df.shape}")
    print(f"[INFO] oos shape:   {oos_df.shape}")

    # -------------------------
    # choose alpha on validation MSE
    # -------------------------
    best_alpha, best_metric, used_feature_cols = pick_best_alpha_by_val(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLS_ALL,
        alpha_grid=ALPHA_GRID,
    )

    print(f"[INFO] Best alpha on validation: {best_alpha:.6g}, val MSE={best_metric:.10f}")
    print(f"[INFO] Final used feature count: {len(used_feature_cols)}")

    # -------------------------
    # refit on train + val
    # -------------------------
    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    tv_needed = [ID_COL, DATE_COL, RET_COL] + used_feature_cols
    train_val_sub = train_val_df[tv_needed].copy()
    X_train_val, y_train_val = prepare_arrays_fixed_features(train_val_sub, used_feature_cols)

    final_model, final_scaler = fit_ridge_model(
        X_train=X_train_val,
        y_train=y_train_val,
        alpha=best_alpha,
    )

    # -------------------------
    # predict on OOS
    # -------------------------
    # For prediction, do not require ret. ret is only needed later for realized return.
    oos_needed = [ID_COL, DATE_COL] + used_feature_cols
    oos_sub = oos_df[oos_needed].copy()
    X_oos = prepare_X_fixed_features(oos_sub, used_feature_cols)

    oos_scores = predict_scores(final_model, final_scaler, X_oos)

    oos_ls, oos_stock_weights = make_vw_ls_returns_and_stock_weights_from_scores(
        df_raw=oos_df[[ID_COL, DATE_COL, RET_COL, WEIGHT_COL]].copy(),
        scores=oos_scores,
        top_q=TOP_Q,
        bot_q=BOT_Q,
        weight_col=WEIGHT_COL,
    )

    oos_ls["oos_year"] = oos_year
    oos_ls["best_alpha"] = best_alpha
    oos_ls["val_mse"] = best_metric
    oos_ls["n_features"] = len(used_feature_cols)
    oos_ls["top_q"] = TOP_Q
    oos_ls["bot_q"] = BOT_Q
    oos_ls["weighting"] = "value_weighted_size"

    oos_stock_weights["oos_year"] = oos_year
    oos_stock_weights["best_alpha"] = best_alpha
    oos_stock_weights["val_mse"] = best_metric
    oos_stock_weights["n_features"] = len(used_feature_cols)
    oos_stock_weights["top_q"] = TOP_Q
    oos_stock_weights["bot_q"] = BOT_Q
    oos_stock_weights["weighting"] = "value_weighted_size"

    # -------------------------
    # save
    # -------------------------
    save_ret_path = os.path.join(OUTPUT_DIR, f"oos_ls_{oos_year}.csv")
    oos_ls.to_csv(save_ret_path, index=False)

    save_weight_path = os.path.join(OUTPUT_DIR, f"oos_stock_weights_{oos_year}.csv")
    oos_stock_weights.to_csv(save_weight_path, index=False)

    selection_summary = pd.DataFrame([{
        "oos_year": oos_year,
        "train_start": train_years[0],
        "train_end": train_years[-1],
        "val_start": val_years[0],
        "val_end": val_years[-1],
        "oos_start": oos_years[0],
        "oos_end": oos_years[-1],
        "best_alpha": best_alpha,
        "val_mse": best_metric,
        "n_features": len(used_feature_cols),
        "top_q": TOP_Q,
        "bot_q": BOT_Q,
        "weighting": "value_weighted_size",
        "y_missing_policy": "drop_missing_ret",
    }])

    selection_path = os.path.join(OUTPUT_DIR, f"selection_summary_{oos_year}.csv")
    selection_summary.to_csv(selection_path, index=False)

    used_features_path = os.path.join(OUTPUT_DIR, f"used_features_{oos_year}.txt")
    with open(used_features_path, "w") as f:
        for c in used_feature_cols:
            f.write(f"{c}\n")

    print(f"[INFO] Saved OOS monthly returns: {save_ret_path}")
    print(f"[INFO] Saved OOS stock weights:    {save_weight_path}")
    print(f"[INFO] Saved selection summary:    {selection_path}")
    print(f"[INFO] Saved used features:         {used_features_path}")
    print(f"[INFO] OOS months kept: {len(oos_ls)}")
    print(f"[INFO] OOS stock weight rows: {len(oos_stock_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="OOS year, e.g. 1981")
    args = parser.parse_args()
    main(args)