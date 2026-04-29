import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ipca import InstrumentedPCA

warnings.filterwarnings("ignore")

# =========================
# SETTINGS
# =========================
TRAIN_YEARS = 10
TEST_YEARS = 1
MONTHS_PER_YEAR = 12
TRAIN_MONTHS = TRAIN_YEARS * MONTHS_PER_YEAR
TEST_MONTHS = TEST_YEARS * MONTHS_PER_YEAR
N_JOBS = int(os.environ.get("N_JOBS", "4"))

CHARS = [
    "abr", "absacc", "acc", "adm", "age",
    "agr", "alm", "ato", "baspread", "beta", "size"
]

# =========================
# HELPER: CROSS-SECTIONAL STANDARDIZATION
# =========================
def cs_zscore(series):
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def cross_sectional_standardize(df, cols):
    df = df.copy()
    df[cols] = df.groupby(level="date")[cols].transform(cs_zscore)
    return df

# =========================
# HELPER: WINDOW PREPROCESSING
# =========================
def prepare_window_data(data_train, data_test, chars):
    train_df = data_train[["ret"] + chars].dropna(subset=["ret"]).copy()
    test_df = data_test[["ret"] + chars].copy()

    if train_df.empty or test_df.empty:
        return None

    # Drop rows with any missing predictors in training
    train_df = train_df.dropna(subset=chars).copy()

    if train_df.empty:
        return None

    # Drop zero / near-zero variance columns within the training window
    stds = train_df[chars].std()
    valid_chars = stds[stds > 1e-8].index.tolist()

    if len(valid_chars) == 0:
        return None

    # Require complete predictors for prediction on the surviving columns
    test_pred_df = test_df.dropna(subset=valid_chars).copy()

    if test_pred_df.empty:
        return None

    return train_df, test_df, test_pred_df, valid_chars

# =========================
# HELPER: RUN ONE WINDOW
# =========================
def run_one_window(start_idx, data, all_dates, chars, train_months, test_months):
    train_start = all_dates[start_idx]
    train_end = all_dates[start_idx + train_months - 1]
    test_start = all_dates[start_idx + train_months]
    test_end = all_dates[start_idx + train_months + test_months - 1]

    print(
        f"Window {start_idx}: "
        f"train {train_start.date()} to {train_end.date()} | "
        f"test {test_start.date()} to {test_end.date()}"
    )

    date_index = data.index.get_level_values("date")

    train_mask = (date_index >= train_start) & (date_index <= train_end)
    test_mask = (date_index >= test_start) & (date_index <= test_end)

    data_train = data.loc[train_mask].copy()
    data_test = data.loc[test_mask].copy()

    prepared = prepare_window_data(data_train, data_test, chars)
    if prepared is None:
        return None

    train_df, test_df, test_pred_df, valid_chars = prepared

    y_train = train_df["ret"]
    X_train = train_df[valid_chars]
    X_pred = test_pred_df[valid_chars]

    try:
        regr = InstrumentedPCA(n_factors=1, intercept=False)
        regr = regr.fit(X=X_train, y=y_train)

        pre_y = regr.predict(
            X=X_pred,
            mean_factor=True,
            data_type="panel",
            label_ind=False
        )

    except np.linalg.LinAlgError as e:
        print(
            f"Skipping window {train_start.date()} to {test_end.date()} "
            f"due to singular matrix: {e}"
        )
        return None
    except Exception as e:
        print(
            f"Skipping window {train_start.date()} to {test_end.date()} "
            f"due to error: {e}"
        )
        return None

    fitted = pd.Series(index=test_df.index, dtype=float, name="fitted")
    fitted.loc[X_pred.index] = np.asarray(pre_y).ravel()

    window_result = pd.concat(
        [
            fitted,
            test_df["ret"].rename("ret"),
            data_test["size_raw"].rename("size_raw")
        ],
        axis=1
    )

    window_result["train_start"] = train_start
    window_result["train_end"] = train_end
    window_result["test_start"] = test_start
    window_result["test_end"] = test_end
    window_result["n_chars_used"] = len(valid_chars)

    return window_result

# =========================
# LOAD DATA
# =========================
print("Loading data...")
data = pd.read_feather("chars_raw_imputed.feather")

data["date"] = pd.to_datetime(data["date"])

# Raw market cap for portfolio weights
data["size_raw"] = (data["prc"] * data["shrout"]).abs()

# Use size as one of the characteristics too
data["size"] = data["size_raw"]

data = data.set_index(["permno", "date"]).sort_index()

# Keep only needed columns
data = data[["ret", "size_raw"] + CHARS].copy()

# =========================
# CROSS-SECTIONAL STANDARDIZATION BY DATE
# =========================
print("Cross-sectionally standardizing characteristics by date...")
data = cross_sectional_standardize(data, CHARS)

# =========================
# ROLLING WINDOWS
# =========================
all_dates = pd.Index(sorted(data.index.get_level_values("date").unique()))
window_start_indices = list(
    range(0, len(all_dates) - TRAIN_MONTHS - TEST_MONTHS + 1, TEST_MONTHS)
)

print(f"Using n_jobs = {N_JOBS}")
print(f"Total rolling windows: {len(window_start_indices)}")

print("Running rolling-window IPCA in parallel...")
oos_prediction_list = Parallel(n_jobs=N_JOBS, backend="loky", verbose=10)(
    delayed(run_one_window)(
        start_idx, data, all_dates, CHARS, TRAIN_MONTHS, TEST_MONTHS
    )
    for start_idx in window_start_indices
)

oos_prediction_list = [x for x in oos_prediction_list if x is not None]

if not oos_prediction_list:
    raise ValueError("No rolling-window OOS predictions were generated.")

print(f"Successful windows: {len(oos_prediction_list)} / {len(window_start_indices)}")

# =========================
# COMBINE OOS PREDICTIONS
# =========================
print("Combining all OOS predictions...")
result = pd.concat(oos_prediction_list, axis=0)
result = result[~result.index.duplicated(keep="first")].sort_index()

# Keep rows with valid signal, return, and raw size for value-weighting
df = result.dropna(subset=["fitted", "ret", "size_raw"]).copy()

# =========================
# CONSTRUCT VALUE-WEIGHTED LONG-SHORT PORTFOLIO
# =========================
print("Constructing value-weighted long-short decile portfolio...")

df["rank"] = df.groupby(level="date")["fitted"].rank(pct=True, method="first")

long_df = df[df["rank"] >= 0.9].copy()
short_df = df[df["rank"] <= 0.1].copy()

long_df["weight"] = long_df.groupby(level="date")["size_raw"].transform(lambda x: x / x.sum())
short_df["weight"] = short_df.groupby(level="date")["size_raw"].transform(lambda x: x / x.sum())

long_ret = long_df.groupby(level="date").apply(
    lambda x: (x["weight"] * x["ret"]).sum(),
    include_groups=False
)
long_ret.name = "long_ret"

short_ret = short_df.groupby(level="date").apply(
    lambda x: (x["weight"] * x["ret"]).sum(),
    include_groups=False
)
short_ret.name = "short_ret"

portfolio = pd.concat([long_ret, short_ret], axis=1)
portfolio["ls_ret"] = portfolio["long_ret"] - portfolio["short_ret"]

# =========================
# SHARPE RATIO
# =========================
print("Computing rolling-window OOS Sharpe ratio...")

mean_ls = portfolio["ls_ret"].mean()
std_ls = portfolio["ls_ret"].std()

if pd.notna(std_ls) and std_ls > 0:
    sharpe_oos = mean_ls / std_ls * np.sqrt(12)
else:
    sharpe_oos = np.nan

print(f"Rolling-window OOS long-short Sharpe ratio: {sharpe_oos:.6f}")

# =========================
# SAVE TIME SERIES
# =========================
print("Saving time-series...")
timeseries = portfolio.reset_index().copy().sort_values("date")
timeseries = timeseries[["date", "long_ret", "short_ret", "ls_ret"]]

timeseries.to_feather(f"Results/ipca_rolling_oos_ls_timeseries_{TRAIN_YEARS}.feather")
timeseries.to_csv(f"Results/ipca_rolling_oos_ls_timeseries_{TRAIN_YEARS}.csv", index=False)

# =========================
# SAVE SUMMARY
# =========================
print("Saving summary...")
summary = pd.DataFrame({
    "metric": [
        "rolling_oos_sharpe",
        "mean_ls_return",
        "std_ls_return",
        "n_periods",
        "train_years",
        "test_years",
        "n_jobs",
        "n_total_windows",
        "n_successful_windows"
    ],
    "value": [
        sharpe_oos,
        mean_ls,
        std_ls,
        timeseries["ls_ret"].notna().sum(),
        TRAIN_YEARS,
        TEST_YEARS,
        N_JOBS,
        len(window_start_indices),
        len(oos_prediction_list)
    ]
})

summary.to_csv(f"Results/ipca_rolling_oos_summary_{TRAIN_YEARS}.csv", index=False)

# =========================
# SAVE STOCK-LEVEL PREDICTIONS
# =========================
print("Saving stock-level OOS predictions...")
result.reset_index().to_feather(f"Results/ipca_rolling_oos_predictions_{TRAIN_YEARS}.feather")

print("Done.")