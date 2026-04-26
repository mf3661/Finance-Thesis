from pathlib import Path
import pandas as pd
import numpy as np
import re

# Change this to your actual grid output folder if needed
RESULTS_DIR = Path("Results/grid_runs")
OUTPUT_XLSX = RESULTS_DIR / "combined_monthly_portfolio_returns.xlsx"

TASK_PATTERN = re.compile(r"task(\d+)")


def extract_task_id(path: Path) -> int:
    m = TASK_PATTERN.search(path.stem)
    if not m:
        raise ValueError(f"Could not extract task id from filename: {path.name}")
    return int(m.group(1))


def combine_from_timeseries(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(results_dir.glob("ipca_grid_timeseries_task*.csv"), key=extract_task_id)
    if not files:
        raise FileNotFoundError("No ipca_grid_timeseries_task*.csv files found.")

    long_frames = []
    wide_frames = []

    for f in files:
        task_id = extract_task_id(f)
        df = pd.read_csv(f)
        if "date" not in df.columns:
            raise ValueError(f"Missing 'date' column in {f.name}")

        df["date"] = pd.to_datetime(df["date"])
        expected_cols = [c for c in ["long_ret", "short_ret", "ls_ret"] if c in df.columns]
        if not expected_cols:
            raise ValueError(f"No portfolio return columns found in {f.name}")

        df.insert(0, "task_id", task_id)
        long_frames.append(df[["task_id", "date"] + expected_cols].copy())

        tmp = df[["date"] + expected_cols].copy().set_index("date")
        tmp.columns = [f"{col}_task{task_id:02d}" for col in tmp.columns]
        wide_frames.append(tmp)

    long_df = pd.concat(long_frames, ignore_index=True).sort_values(["date", "task_id"])
    wide_df = pd.concat(wide_frames, axis=1).sort_index()
    wide_df.index.name = "date"
    wide_df = wide_df.reset_index()
    return long_df, wide_df


def combine_from_feathers(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(results_dir.glob("ipca_grid_predictions_task*.feather"), key=extract_task_id)
    if not files:
        raise FileNotFoundError("No ipca_grid_predictions_task*.feather files found.")

    long_frames = []
    wide_frames = []

    for f in files:
        task_id = extract_task_id(f)
        df = pd.read_feather(f)
        needed = {"date", "fitted", "ret", "size_raw"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {f.name}: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["fitted", "ret", "size_raw"]).copy()
        if df.empty:
            continue

        df["rank"] = df.groupby("date")["fitted"].rank(pct=True, method="first")
        long_df = df[df["rank"] >= 0.9].copy()
        short_df = df[df["rank"] <= 0.1].copy()
        if long_df.empty or short_df.empty:
            continue

        long_df["weight"] = long_df.groupby("date")["size_raw"].transform(lambda x: x / x.sum())
        short_df["weight"] = short_df.groupby("date")["size_raw"].transform(lambda x: x / x.sum())

        long_ret = long_df.groupby("date").apply(lambda x: (x["weight"] * x["ret"]).sum())
        short_ret = short_df.groupby("date").apply(lambda x: (x["weight"] * x["ret"]).sum())

        ts = pd.concat(
            [long_ret.rename("long_ret"), short_ret.rename("short_ret")],
            axis=1,
        )
        ts["ls_ret"] = ts["long_ret"] - ts["short_ret"]
        ts = ts.reset_index()
        ts.insert(0, "task_id", task_id)
        long_frames.append(ts.copy())

        tmp = ts[["date", "long_ret", "short_ret", "ls_ret"]].copy().set_index("date")
        tmp.columns = [f"{col}_task{task_id:02d}" for col in tmp.columns]
        wide_frames.append(tmp)

    if not long_frames:
        raise ValueError("No usable feather prediction files were found.")

    long_out = pd.concat(long_frames, ignore_index=True).sort_values(["date", "task_id"])
    wide_out = pd.concat(wide_frames, axis=1).sort_index().reset_index()
    return long_out, wide_out


def build_overall_ls(long_df: pd.DataFrame) -> pd.DataFrame:
    overall = (
        long_df.groupby("date")[["long_ret", "short_ret", "ls_ret"]]
        .first()
        .reset_index()
        .sort_values("date")
    )
    return overall


def build_task_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    summary = long_df.groupby("task_id").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        n_months=("date", "nunique"),
        mean_long=("long_ret", "mean"),
        mean_short=("short_ret", "mean"),
        mean_ls=("ls_ret", "mean"),
        std_ls=("ls_ret", "std"),
    ).reset_index()
    summary["annualized_sharpe_ls"] = np.where(
        summary["std_ls"].gt(0),
        summary["mean_ls"] / summary["std_ls"] * np.sqrt(12),
        np.nan,
    )
    return summary


def main() -> None:
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir.resolve()}")

    try:
        long_df, wide_df = combine_from_timeseries(results_dir)
        source_used = "timeseries_csv"
    except FileNotFoundError:
        long_df, wide_df = combine_from_feathers(results_dir)
        source_used = "prediction_feather"

    overall_df = build_overall_ls(long_df)
    summary_df = build_task_summary(long_df)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="monthly_returns_long", index=False)
        wide_df.to_excel(writer, sheet_name="monthly_returns_wide", index=False)
        overall_df.to_excel(writer, sheet_name="overall_oos_series", index=False)
        summary_df.to_excel(writer, sheet_name="task_summary", index=False)
        pd.DataFrame({"source_used": [source_used], "results_dir": [str(results_dir.resolve())]}).to_excel(
            writer, sheet_name="meta", index=False
        )

    print(f"Saved Excel file to: {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()
