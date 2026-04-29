from __future__ import annotations

import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from ipca import InstrumentedPCA


# =========================================================
# paths
# =========================================================
BY_YEAR_DIR = "../../../data/feature_construction/1_4_rank_feature/by_year/impute"
CHARS_PATH = "../../../data/common/chars_summary.csv"
OUTPUT_DIR = "../../../data/3_IPCA/3_1_run_IPCA/3_1_1_ipca/output"

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
# IPCA setup
# =========================================================
FACTOR_GRID = [1, 2, 3, 4, 5, 6]

IPCA_MAX_ITER = 200
IPCA_ITER_TOL = 1e-6


# =========================================================
# read characteristic names
# =========================================================
chars = pd.read_csv(CHARS_PATH)
acronyms = chars["Acronym"].astype(str).str.rstrip().tolist()

# New project input is ranked/imputed feature data.
# Features should therefore use *_rank columns.
feature_cols_all = [f"{c}_rank" for c in acronyms]
feature_cols_all = list(dict.fromkeys(feature_cols_all))


# =========================================================
# helper functions
# =========================================================
def unique_keep_order(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def load_years(year_list, by_year_dir, needed_cols):
    needed_cols = unique_keep_order(needed_cols)

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


def drop_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Calculate cross-correlation and drop redundant features to reduce
    singular / non-positive-definite matrix issues.
    """
    print(f"[INFO] Checking collinearity (r > {threshold})...")

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    keep_features = [c for c in feature_cols if c not in to_drop]

    print(f"[INFO] Dropped {len(to_drop)} collinear features. Kept {len(keep_features)}.")
    return keep_features


def add_jitter(X: np.ndarray, noise_level: float = 1e-5) -> np.ndarray:
    """
    Inject tiny Gaussian noise to reduce singular matrix errors.
    """
    rng = np.random.default_rng(42)
    jitter = rng.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + jitter


def clean_and_standardize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Clean X features using cross-sectional logic:
    - numeric conversion
    - inf -> NaN
    - fill missing with cross-sectional mean
    - cross-sectional z-score standardization

    Even though the new input is already rank/impute data, this keeps the
    successful IPCA numerical stabilization logic from the working version.
    """
    tmp = df.copy()

    for c in feature_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp[feature_cols] = tmp[feature_cols].replace([np.inf, -np.inf], np.nan)

    means = tmp.groupby(DATE_COL)[feature_cols].transform("mean")
    stds = tmp.groupby(DATE_COL)[feature_cols].transform("std", ddof=0)

    tmp[feature_cols] = tmp[feature_cols].fillna(means)

    valid_std = stds > 1e-8
    tmp[feature_cols] = np.where(
        valid_std,
        (tmp[feature_cols] - means) / stds,
        0.0,
    )

    tmp[feature_cols] = tmp[feature_cols].fillna(0.0)

    return tmp


def clean_ret_drop_missing(df: pd.DataFrame, ret_col: str = RET_COL) -> pd.DataFrame:
    """
    Clean y / ret.

    ret cannot be filled with 0.
    Missing or infinite ret must be dropped.
    """
    tmp = df.copy()

    tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")
    tmp[ret_col] = tmp[ret_col].replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[ret_col]).copy()

    return tmp


def get_nonzero_variance_features(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    var_series = df[feature_cols].var(axis=0, ddof=0)
    keep_features = var_series[var_series > 0].index.tolist()
    return keep_features


def prepare_train_panel(
    df: pd.DataFrame,
    feature_cols: list[str],
    ret_col: str = RET_COL,
):
    """
    Prepare train panel for IPCA:
    - y / ret: drop missing, never fill with 0
    - X features: cross-sectional standardization
    - remove zero-variance and highly collinear features
    - add jitter
    - build indices = [permno, date_code]
    """
    needed = unique_keep_order([ID_COL, DATE_COL, ret_col] + feature_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

    tmp = clean_ret_drop_missing(tmp, ret_col=ret_col)
    tmp = clean_and_standardize_features(tmp, feature_cols=feature_cols)

    used_feature_cols = get_nonzero_variance_features(tmp, feature_cols)
    if len(used_feature_cols) == 0:
        raise ValueError("All feature columns have zero variance after cleaning.")

    used_feature_cols = drop_highly_correlated_features(
        tmp,
        used_feature_cols,
        threshold=0.95,
    )

    tmp = tmp.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    X = tmp[used_feature_cols].to_numpy(dtype=float)
    y = tmp[ret_col].to_numpy(dtype=float)

    X = add_jitter(X, noise_level=1e-5)

    date_codes = pd.factorize(tmp[DATE_COL])[0]
    indices = np.column_stack([
        tmp[ID_COL].to_numpy(),
        date_codes,
    ])

    print(f"[DEBUG] train rows after dropping missing y: {len(tmp)}")
    print(f"[DEBUG] used feature count: {len(used_feature_cols)}")
    print(f"[DEBUG] train months: {tmp[DATE_COL].nunique()}")

    return X, y, indices, used_feature_cols


def prepare_panel_fixed_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    ret_col: str = RET_COL,
):
    """
    Prepare validation or train+val panel:
    - y / ret: drop missing, never fill with 0
    - X features: cross-sectional standardization
    - fixed feature columns from training
    """
    needed = unique_keep_order([ID_COL, DATE_COL, ret_col] + feature_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

    tmp = clean_ret_drop_missing(tmp, ret_col=ret_col)
    tmp = clean_and_standardize_features(tmp, feature_cols=feature_cols)

    tmp = tmp.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    X = tmp[feature_cols].to_numpy(dtype=float)
    y = tmp[ret_col].to_numpy(dtype=float)

    date_codes = pd.factorize(tmp[DATE_COL])[0]
    indices = np.column_stack([
        tmp[ID_COL].to_numpy(),
        date_codes,
    ])

    return X, y, indices, tmp


def prepare_oos_X_fixed_features(
    df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Prepare OOS X only:
    - do not require ret
    - X features: cross-sectional standardization
    - fixed feature columns
    """
    needed = unique_keep_order([ID_COL, DATE_COL] + feature_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tmp = df[needed].copy()
    tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL])

    tmp = clean_and_standardize_features(tmp, feature_cols=feature_cols)

    tmp = tmp.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)

    X = tmp[feature_cols].to_numpy(dtype=float)

    date_codes = pd.factorize(tmp[DATE_COL])[0]
    indices = np.column_stack([
        tmp[ID_COL].to_numpy(),
        date_codes,
    ])

    return X, indices, tmp


def fit_ipca_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    indices_train: np.ndarray,
    n_factors: int,
):
    """
    Fit IPCA.
    """
    model = InstrumentedPCA(
        n_factors=n_factors,
        intercept=False,
        max_iter=IPCA_MAX_ITER,
        iter_tol=IPCA_ITER_TOL,
        n_jobs=-1,
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        model.fit(
            X=X_train,
            y=y_train,
            indices=indices_train,
        )

        if len(caught_warnings) > 0:
            for w in caught_warnings:
                print(f"[WARNING] IPCA warning for K={n_factors}: {w.message}")

    return model


def predict_ipca_mean_factor(
    model,
    X: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    IPCA predicted return using mean factor.
    """
    pred = model.predict(
        X=X,
        indices=indices,
        mean_factor=True,
    )

    pred = np.asarray(pred, dtype=float).reshape(-1)
    return pred


def make_vw_ls_returns_and_stock_weights_from_scores(
    df_raw: pd.DataFrame,
    scores: np.ndarray,
    top_q: float = TOP_Q,
    bot_q: float = BOT_Q,
    weight_col: str = WEIGHT_COL,
):
    """
    Construct value-weighted long-short returns and stock-level weights.

    Long:
        top 20% by predicted return
        strategy_weight = + size_i / sum(size_i in long leg)

    Short:
        bottom 20% by predicted return
        strategy_weight = - size_i / sum(size_i in short leg)
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


def pick_best_n_factors_by_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    factor_grid: list[int],
):
    best_k = None
    best_metric = np.inf
    best_features = None
    val_records = []

    X_train, y_train, indices_train, used_feature_cols = prepare_train_panel(
        train_df,
        feature_cols,
    )

    X_val, y_val, indices_val, _ = prepare_panel_fixed_features(
        val_df,
        used_feature_cols,
    )

    for k in factor_grid:
        try:
            print("-" * 80)
            print(f"[INFO] Fitting IPCA with n_factors={k}, max_iter={IPCA_MAX_ITER}")

            model = fit_ipca_model(
                X_train=X_train,
                y_train=y_train,
                indices_train=indices_train,
                n_factors=k,
            )

            val_pred = predict_ipca_mean_factor(
                model=model,
                X=X_val,
                indices=indices_val,
            )

            metric = mean_squared_error(y_val, val_pred)

            print(f"[INFO] n_factors={k}, val MSE={metric:.10f}")

            val_records.append({
                "n_factors": k,
                "val_mse": float(metric),
                "status": "success",
            })

            if metric < best_metric:
                best_metric = float(metric)
                best_k = int(k)
                best_features = used_feature_cols

        except Exception as e:
            print(f"[WARNING] n_factors={k} failed in validation: {e}")
            val_records.append({
                "n_factors": k,
                "val_mse": np.nan,
                "status": f"failed: {e}",
            })

    if best_k is None or best_features is None:
        val_result_df = pd.DataFrame(val_records)
        print("=" * 80)
        print("[ERROR] Validation results:")
        print(val_result_df.to_string(index=False))
        raise ValueError("No valid n_factors found in validation.")

    val_result_df = pd.DataFrame(val_records)

    return best_k, best_metric, best_features, val_result_df


# =========================================================
# main
# =========================================================
def main(args):
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

    needed_cols = unique_keep_order(
        [ID_COL, DATE_COL, RET_COL, WEIGHT_COL] + feature_cols_all
    )

    print("=" * 80)
    print(f"[INFO] IPCA OOS year: {oos_year}")
    print(f"[INFO] Train years: {train_years[0]}-{train_years[-1]}")
    print(f"[INFO] Val years:   {val_years[0]}-{val_years[-1]}")
    print(f"[INFO] OOS years:   {oos_years[0]}-{oos_years[-1]}")
    print(f"[INFO] Input dir:    {BY_YEAR_DIR}")
    print(f"[INFO] Output dir:   {OUTPUT_DIR}")
    print(f"[INFO] Feature count: {len(feature_cols_all)}")
    print(f"[INFO] Factor grid:   {FACTOR_GRID}")
    print(f"[INFO] IPCA max_iter: {IPCA_MAX_ITER}")
    print(f"[INFO] Long leg: top {int((1 - TOP_Q) * 100)}% by predicted return")
    print(f"[INFO] Short leg: bottom {int(BOT_Q * 100)}% by predicted return")
    print("[INFO] Portfolio weighting: value-weighted by size")
    print("[INFO] IPCA prediction: mean_factor=True")
    print("[INFO] IPCA intercept: False")
    print("[INFO] X input: ranked/imputed features, then cross_sectional_zscore")
    print("[INFO] y / ret handling: drop missing, never fill with 0")

    train_df = load_years(train_years, BY_YEAR_DIR, needed_cols)
    val_df = load_years(val_years, BY_YEAR_DIR, needed_cols)
    oos_df = load_years(oos_years, BY_YEAR_DIR, needed_cols)

    print(f"[INFO] train shape: {train_df.shape}")
    print(f"[INFO] val shape:   {val_df.shape}")
    print(f"[INFO] oos shape:   {oos_df.shape}")

    best_k, best_metric, used_feature_cols, val_result_df = pick_best_n_factors_by_val(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols_all,
        factor_grid=FACTOR_GRID,
    )

    print("=" * 80)
    print(f"[INFO] Best n_factors on validation: {best_k}, val MSE={best_metric:.10f}")
    print(f"[INFO] Final used feature count: {len(used_feature_cols)}")

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    X_train_val, y_train_val, indices_train_val, _ = prepare_panel_fixed_features(
        train_val_df,
        used_feature_cols,
    )

    final_model = fit_ipca_model(
        X_train=X_train_val,
        y_train=y_train_val,
        indices_train=indices_train_val,
        n_factors=best_k,
    )

    oos_needed = unique_keep_order([ID_COL, DATE_COL] + used_feature_cols)
    oos_sub = oos_df[oos_needed].copy()

    X_oos, indices_oos, oos_pred_ref = prepare_oos_X_fixed_features(
        oos_sub,
        used_feature_cols,
    )

    oos_scores = predict_ipca_mean_factor(
        model=final_model,
        X=X_oos,
        indices=indices_oos,
    )

    oos_raw_for_port = oos_pred_ref[[ID_COL, DATE_COL]].copy()
    oos_raw_for_port = oos_raw_for_port.merge(
        oos_df[[ID_COL, DATE_COL, RET_COL, WEIGHT_COL]],
        on=[ID_COL, DATE_COL],
        how="left",
        validate="one_to_one",
    )

    oos_ls, oos_stock_weights = make_vw_ls_returns_and_stock_weights_from_scores(
        df_raw=oos_raw_for_port[[ID_COL, DATE_COL, RET_COL, WEIGHT_COL]].copy(),
        scores=oos_scores,
        top_q=TOP_Q,
        bot_q=BOT_Q,
        weight_col=WEIGHT_COL,
    )

    oos_ls["oos_year"] = oos_year
    oos_ls["best_n_factors"] = best_k
    oos_ls["val_mse"] = best_metric
    oos_ls["n_features"] = len(used_feature_cols)
    oos_ls["top_q"] = TOP_Q
    oos_ls["bot_q"] = BOT_Q
    oos_ls["weighting"] = "value_weighted_size"
    oos_ls["prediction"] = "ipca_mean_factor"
    oos_ls["ipca_intercept"] = False
    oos_ls["ipca_max_iter"] = IPCA_MAX_ITER
    oos_ls["x_scaling"] = "rank_impute_then_cross_sectional_zscore"
    oos_ls["y_missing_policy"] = "drop_missing_ret"

    oos_stock_weights["oos_year"] = oos_year
    oos_stock_weights["best_n_factors"] = best_k
    oos_stock_weights["val_mse"] = best_metric
    oos_stock_weights["n_features"] = len(used_feature_cols)
    oos_stock_weights["top_q"] = TOP_Q
    oos_stock_weights["bot_q"] = BOT_Q
    oos_stock_weights["weighting"] = "value_weighted_size"
    oos_stock_weights["prediction"] = "ipca_mean_factor"
    oos_stock_weights["ipca_intercept"] = False
    oos_stock_weights["ipca_max_iter"] = IPCA_MAX_ITER
    oos_stock_weights["x_scaling"] = "rank_impute_then_cross_sectional_zscore"
    oos_stock_weights["y_missing_policy"] = "drop_missing_ret"

    save_ret_path = os.path.join(OUTPUT_DIR, f"oos_ls_ipca_{oos_year}.csv")
    oos_ls.to_csv(save_ret_path, index=False)

    save_weight_path = os.path.join(OUTPUT_DIR, f"oos_stock_weights_ipca_{oos_year}.csv")
    oos_stock_weights.to_csv(save_weight_path, index=False)

    selection_summary = pd.DataFrame([{
        "oos_year": oos_year,
        "train_start": train_years[0],
        "train_end": train_years[-1],
        "val_start": val_years[0],
        "val_end": val_years[-1],
        "oos_start": oos_years[0],
        "oos_end": oos_years[-1],
        "best_n_factors": best_k,
        "val_mse": best_metric,
        "n_features": len(used_feature_cols),
        "top_q": TOP_Q,
        "bot_q": BOT_Q,
        "factor_grid": ",".join(str(x) for x in FACTOR_GRID),
        "weighting": "value_weighted_size",
        "prediction": "ipca_mean_factor",
        "ipca_intercept": False,
        "ipca_max_iter": IPCA_MAX_ITER,
        "ipca_iter_tol": IPCA_ITER_TOL,
        "x_scaling": "rank_impute_then_cross_sectional_zscore",
        "y_missing_policy": "drop_missing_ret",
    }])

    selection_path = os.path.join(OUTPUT_DIR, f"ipca_selection_summary_{oos_year}.csv")
    selection_summary.to_csv(selection_path, index=False)

    val_result_path = os.path.join(OUTPUT_DIR, f"ipca_validation_mse_{oos_year}.csv")
    val_result_df.to_csv(val_result_path, index=False)

    used_features_path = os.path.join(OUTPUT_DIR, f"ipca_used_features_{oos_year}.txt")
    with open(used_features_path, "w") as f:
        for c in used_feature_cols:
            f.write(f"{c}\n")

    print(f"[INFO] Saved IPCA OOS monthly returns: {save_ret_path}")
    print(f"[INFO] Saved IPCA OOS stock weights:    {save_weight_path}")
    print(f"[INFO] Saved IPCA selection summary:    {selection_path}")
    print(f"[INFO] Saved IPCA validation MSE:        {val_result_path}")
    print(f"[INFO] Saved IPCA used features:         {used_features_path}")
    print(f"[INFO] OOS months kept: {len(oos_ls)}")
    print(f"[INFO] OOS stock weight rows: {len(oos_stock_weights)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True, help="OOS year, e.g. 1981")
    args = parser.parse_args()
    main(args)