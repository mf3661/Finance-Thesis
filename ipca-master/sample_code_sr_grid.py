import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ipca import InstrumentedPCA

warnings.filterwarnings("ignore")

MONTHS_PER_YEAR = 12
TRAIN_YEARS = 20
VAL_YEARS = 10
TEST_YEARS = 1
REFIT_MONTHS = 12
TRAIN_MONTHS = TRAIN_YEARS * MONTHS_PER_YEAR
VAL_MONTHS = VAL_YEARS * MONTHS_PER_YEAR
TEST_MONTHS = TEST_YEARS * MONTHS_PER_YEAR
TOTAL_MONTHS = TRAIN_MONTHS + VAL_MONTHS + TEST_MONTHS
EXPECTED_TASKS = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one annual rolling IPCA prediction window on the grid.")
    parser.add_argument("--task_id", type=int, required=True, help="1-based task index, typically 1..44")
    parser.add_argument("--data_path", type=Path, default=Path("chars_raw_imputed.feather"))
    parser.add_argument("--chars_csv_path", type=Path, default=Path("chars_summary.csv"))
    parser.add_argument("--results_dir", type=Path, default=Path("Results/grid_runs"))
    return parser.parse_args()



def load_characteristics(chars_csv_path: Path) -> list[str]:
    chars = pd.read_csv(chars_csv_path, encoding="utf-8-sig")
    if "Acronym" not in chars.columns:
        raise ValueError(f"'Acronym' column not found in {chars_csv_path}")

    acronyms = (
        chars["Acronym"]
        .astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna()
        .tolist()
    )
    acronyms = list(dict.fromkeys(acronyms))
    if not acronyms:
        raise ValueError(f"No characteristics found in {chars_csv_path}")
    return acronyms



def cs_zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std



def cross_sectional_standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[cols] = df.groupby(level="date")[cols].transform(cs_zscore)
    return df



def prepare_window_data(
    data_train: pd.DataFrame,
    data_val: pd.DataFrame,
    data_test: pd.DataFrame,
    chars: list[str],
):
    train_df = data_train[["ret"] + chars].dropna(subset=["ret"]).copy()
    val_df = data_val[["ret"] + chars].dropna(subset=["ret"]).copy()
    test_df = data_test[["ret"] + chars].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        return None

    train_df = train_df.dropna(subset=chars).copy()
    val_df = val_df.dropna(subset=chars).copy()
    if train_df.empty or val_df.empty:
        return None

    stds = train_df[chars].std()
    valid_chars = stds[stds > 1e-8].index.tolist()
    if not valid_chars:
        return None

    train_df = train_df.dropna(subset=valid_chars).copy()
    val_df = val_df.dropna(subset=valid_chars).copy()
    test_pred_df = test_df.dropna(subset=valid_chars).copy()
    if train_df.empty or val_df.empty or test_pred_df.empty:
        return None

    return train_df, val_df, test_df, test_pred_df, valid_chars



def build_window_table(all_dates: pd.Index) -> list[dict]:
    window_starts = list(range(0, len(all_dates) - TOTAL_MONTHS + 1, REFIT_MONTHS))
    windows: list[dict] = []

    for idx, start in enumerate(window_starts, start=1):
        train_start = all_dates[start]
        train_end = all_dates[start + TRAIN_MONTHS - 1]
        val_start = all_dates[start + TRAIN_MONTHS]
        val_end = all_dates[start + TRAIN_MONTHS + VAL_MONTHS - 1]
        test_start = all_dates[start + TRAIN_MONTHS + VAL_MONTHS]
        test_end = all_dates[start + TOTAL_MONTHS - 1]
        windows.append(
            {
                "task_id": idx,
                "start_idx": start,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return windows



def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    chars = load_characteristics(args.chars_csv_path)

    print("Loading data...")
    data = pd.read_feather(args.data_path)
    data["date"] = pd.to_datetime(data["date"])

    required_for_size = {"prc", "shrout"}
    if not required_for_size.issubset(data.columns):
        missing = sorted(required_for_size - set(data.columns))
        raise ValueError(f"Missing required columns for size construction: {missing}")

    data["size_raw"] = (data["prc"] * data["shrout"]).abs()
    if "size" not in data.columns:
        data["size"] = data["size_raw"]

    missing_chars = [c for c in chars if c not in data.columns]
    if missing_chars:
        raise ValueError(
            "The following characteristics from chars_summary.csv are missing in the raw data: "
            f"{missing_chars}"
        )

    keep_cols = ["permno", "date", "ret", "size_raw"] + chars
    data = data[keep_cols].copy()
    data = data.set_index(["permno", "date"]).sort_index()

    print(f"Cross-sectionally standardizing {len(chars)} characteristics by date...")
    data = cross_sectional_standardize(data, chars)

    all_dates = pd.Index(sorted(data.index.get_level_values("date").unique()))
    windows = build_window_table(all_dates)
    n_windows = len(windows)

    print(f"Detected {n_windows} annual rolling windows.")
    if n_windows != EXPECTED_TASKS:
        print(f"[Warning] Expected {EXPECTED_TASKS} tasks, but detected {n_windows} windows from the data.")

    if args.task_id < 1 or args.task_id > n_windows:
        raise ValueError(f"task_id must be between 1 and {n_windows}, got {args.task_id}")

    window = windows[args.task_id - 1]
    print(
        f"Running task {args.task_id}/{n_windows}: "
        f"train {window['train_start'].date()} to {window['train_end'].date()} | "
        f"val {window['val_start'].date()} to {window['val_end'].date()} | "
        f"test {window['test_start'].date()} to {window['test_end'].date()}"
    )

    date_index = data.index.get_level_values("date")
    train_mask = (date_index >= window["train_start"]) & (date_index <= window["train_end"])
    val_mask = (date_index >= window["val_start"]) & (date_index <= window["val_end"])
    test_mask = (date_index >= window["test_start"]) & (date_index <= window["test_end"])

    data_train = data.loc[train_mask].copy()
    data_val = data.loc[val_mask].copy()
    data_test = data.loc[test_mask].copy()

    prepared = prepare_window_data(data_train, data_val, data_test, chars)
    if prepared is None:
        raise ValueError(f"Task {args.task_id}: empty or invalid data after preprocessing.")

    train_df, val_df, test_df, test_pred_df, valid_chars = prepared
    y_train = train_df["ret"]
    X_train = train_df[valid_chars]
    X_pred = test_pred_df[valid_chars]

    print(f"Task {args.task_id}: fitting IPCA with {len(valid_chars)} characteristics...")
    regr = InstrumentedPCA(n_factors=1, intercept=False)
    regr = regr.fit(X=X_train, y=y_train)

    pre_y = regr.predict(
        X=X_pred,
        mean_factor=True,
        data_type="panel",
        label_ind=False,
    )

    fitted = pd.Series(index=test_df.index, dtype=float, name="fitted")
    fitted.loc[X_pred.index] = np.asarray(pre_y).ravel()

    result = pd.concat(
        [
            fitted,
            test_df["ret"].rename("ret"),
            data_test["size_raw"].rename("size_raw"),
        ],
        axis=1,
    )

    result["train_start"] = window["train_start"]
    result["train_end"] = window["train_end"]
    result["val_start"] = window["val_start"]
    result["val_end"] = window["val_end"]
    result["test_start"] = window["test_start"]
    result["test_end"] = window["test_end"]
    result["task_id"] = args.task_id
    result["n_chars_used"] = len(valid_chars)

    df = result.dropna(subset=["fitted", "ret", "size_raw"]).copy()
    if df.empty:
        raise ValueError(f"Task {args.task_id}: no valid stock-level predictions after dropping missing values.")

    df["rank"] = df.groupby(level="date")["fitted"].rank(pct=True, method="first")

    long_df = df[df["rank"] >= 0.9].copy()
    short_df = df[df["rank"] <= 0.1].copy()
    if long_df.empty or short_df.empty:
        raise ValueError(f"Task {args.task_id}: long or short decile is empty.")

    long_df["weight"] = long_df.groupby(level="date")["size_raw"].transform(lambda x: x / x.sum())
    short_df["weight"] = short_df.groupby(level="date")["size_raw"].transform(lambda x: x / x.sum())

    long_ret = long_df.groupby(level="date").apply(
        lambda x: (x["weight"] * x["ret"]).sum(),
        include_groups=False,
    )
    short_ret = short_df.groupby(level="date").apply(
        lambda x: (x["weight"] * x["ret"]).sum(),
        include_groups=False,
    )

    portfolio = pd.concat(
        [long_ret.rename("long_ret"), short_ret.rename("short_ret")],
        axis=1,
    )
    portfolio["ls_ret"] = portfolio["long_ret"] - portfolio["short_ret"]

    mean_ls = portfolio["ls_ret"].mean()
    std_ls = portfolio["ls_ret"].std()
    sharpe_oos = mean_ls / std_ls * np.sqrt(12) if pd.notna(std_ls) and std_ls > 0 else np.nan

    task_tag = f"task{args.task_id:02d}"
    pred_path = args.results_dir / f"ipca_grid_predictions_{task_tag}.feather"
    ts_path = args.results_dir / f"ipca_grid_timeseries_{task_tag}.csv"
    summary_path = args.results_dir / f"ipca_grid_summary_{task_tag}.csv"
    manifest_path = args.results_dir / f"ipca_grid_window_manifest.csv"

    result.reset_index().to_feather(pred_path)
    portfolio.reset_index().to_csv(ts_path, index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "task_id",
                "train_years",
                "validation_years",
                "test_years",
                "refit_frequency_months",
                "n_total_windows",
                "n_chars_total",
                "n_chars_used",
                "rolling_oos_sharpe",
                "mean_ls_return",
                "std_ls_return",
                "n_test_months_realized",
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
            ],
            "value": [
                args.task_id,
                TRAIN_YEARS,
                VAL_YEARS,
                TEST_YEARS,
                REFIT_MONTHS,
                n_windows,
                len(chars),
                len(valid_chars),
                sharpe_oos,
                mean_ls,
                std_ls,
                portfolio["ls_ret"].notna().sum(),
                window["train_start"],
                window["train_end"],
                window["val_start"],
                window["val_end"],
                window["test_start"],
                window["test_end"],
            ],
        }
    )
    summary.to_csv(summary_path, index=False)

    manifest = pd.DataFrame(windows)
    manifest.to_csv(manifest_path, index=False)

    print(f"Task {args.task_id} finished.")
    print(f"Predictions: {pred_path}")
    print(f"Time series: {ts_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
