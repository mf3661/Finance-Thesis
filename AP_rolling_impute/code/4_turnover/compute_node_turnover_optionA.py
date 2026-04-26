#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

NODE_CANDIDATES = [
    "node", "node_id", "leaf", "leaf_id", "port_id", "portfolio_id", "bucket", "group"
]

def infer_column(df, requested, candidates, kind, file_path):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(
                f"{kind} column '{requested}' not found in {file_path}. "
                f"Available columns: {list(df.columns)}"
            )
        return requested
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer {kind} column in {file_path}. "
        f"Available columns: {list(df.columns)}"
    )

def build_stock_weights_from_leaf(leaf_path, date_col, stock_col, node_col=None, weight_col=None):
    df = pd.read_parquet(leaf_path).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    node_col = infer_column(df, node_col, NODE_CANDIDATES, "node", leaf_path)

    out = df[[date_col, stock_col, node_col]].copy()
    out.columns = ["date", "permno", "node_id"]

    if weight_col is None:
        out["w"] = 1.0
    else:
        if weight_col not in df.columns:
            raise ValueError(
                f"weight_col '{weight_col}' not found in {leaf_path}. "
                f"Available columns: {list(df.columns)}"
            )
        out["w"] = df[weight_col].astype(float).values

    out["w"] = out["w"] / out.groupby(["date", "node_id"])["w"].transform("sum")
    return out

def average_weights_across_repeats(weights_list):
    if not weights_list:
        return pd.DataFrame(columns=["date", "permno", "node_id", "w"])
    stacked = pd.concat(weights_list, ignore_index=True)
    avg_w = stacked.groupby(["date", "permno", "node_id"], as_index=False)["w"].mean()
    avg_w["w"] = avg_w["w"] / avg_w.groupby(["date", "node_id"])["w"].transform("sum")
    return avg_w

def compute_monthly_turnover(weights_df, stock_ret_df, stock_ret_date_col, stock_ret_id_col, stock_ret_col):
    w = weights_df.copy()
    r = stock_ret_df.copy()

    w["date"] = pd.to_datetime(w["date"])
    r[stock_ret_date_col] = pd.to_datetime(r[stock_ret_date_col])

    unique_dates = sorted(w["date"].dropna().unique())
    if len(unique_dates) < 2:
        return pd.DataFrame(columns=["date", "node_id", "turnover"]), pd.DataFrame()

    date_map = pd.DataFrame({"t_date": unique_dates})
    date_map["t1_date"] = date_map["t_date"].shift(-1)

    wt = w.merge(date_map, left_on="date", right_on="t_date", how="left").rename(columns={"w": "w_t"})
    wt = wt.drop(columns=["date"])

    wt1 = w.rename(columns={"date": "t1_date", "w": "w_t1"}).copy()

    rr = r[[stock_ret_date_col, stock_ret_id_col, stock_ret_col]].rename(
        columns={
            stock_ret_date_col: "t_date",
            stock_ret_id_col: "permno",
            stock_ret_col: "ret_t1",
        }
    )

    panel = wt.merge(rr, on=["t_date", "permno"], how="left")
    panel["ret_t1"] = panel["ret_t1"].fillna(0.0)

    port_ret = (
        panel.groupby(["node_id", "t_date"], as_index=False)
        .apply(lambda x: np.sum(x["w_t"] * x["ret_t1"]), include_groups=False)
        .rename(columns={None: "port_ret_t1"})
    )

    panel = panel.merge(port_ret, on=["node_id", "t_date"], how="left")
    panel["w_drift"] = panel["w_t"] * (1.0 + panel["ret_t1"]) / (1.0 + panel["port_ret_t1"])

    panel = panel.merge(
        wt1[["t1_date", "permno", "node_id", "w_t1"]],
        on=["t1_date", "permno", "node_id"],
        how="outer",
    )

    panel["w_t1"] = panel["w_t1"].fillna(0.0)
    panel["w_drift"] = panel["w_drift"].fillna(0.0)
    panel = panel[panel["t1_date"].notna()].copy()

    if panel.empty:
        return pd.DataFrame(columns=["date", "node_id", "turnover"]), panel

    panel["abs_diff"] = np.abs(panel["w_t1"] - panel["w_drift"])

    monthly_turnover = (
        panel.groupby(["node_id", "t1_date"], as_index=False)["abs_diff"]
        .sum()
        .rename(columns={"t1_date": "date"})
    )
    monthly_turnover["turnover"] = 0.5 * monthly_turnover["abs_diff"]
    monthly_turnover = monthly_turnover.drop(columns="abs_diff")
    return monthly_turnover, panel

def process_all_years(root_parent, stock_ret_df, out_dir, leaf_date_col, leaf_stock_col,
                      leaf_node_col=None, leaf_weight_col=None, ret_date_col="date",
                      ret_stock_col="permno", ret_col="ret"):
    root_parent = Path(root_parent)
    repeat_dirs = sorted([p for p in root_parent.glob("repeat_*") if p.is_dir()])
    if not repeat_dirs:
        raise ValueError(f"No repeat_* folders found under {root_parent}")

    years = sorted({
        p.name
        for rep in repeat_dirs
        for p in rep.iterdir()
        if p.is_dir() and p.name.isdigit()
    })
    if not years:
        raise ValueError(f"No year folders found under repeat_* in {root_parent}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_monthly_turnover = []
    all_values = []

    for year in years:
        combo_set = set()
        for rep in repeat_dirs:
            year_dir = rep / year
            if year_dir.exists():
                combo_set.update(
                    p.name.replace("_leaf.parquet", "")
                    for p in year_dir.glob("combo_*_leaf.parquet")
                )

        combo_names = sorted(combo_set)
        if not combo_names:
            print(f"[WARN] No combo leaf files found for year {year}")
            continue

        year_out_dir = out_dir / year
        year_out_dir.mkdir(parents=True, exist_ok=True)

        year_turnovers = []
        year_values = []

        for combo_name in combo_names:
            weights_list = []
            for rep in repeat_dirs:
                leaf_path = rep / year / f"{combo_name}_leaf.parquet"
                if not leaf_path.exists():
                    continue
                try:
                    w = build_stock_weights_from_leaf(
                        leaf_path=leaf_path,
                        date_col=leaf_date_col,
                        stock_col=leaf_stock_col,
                        node_col=leaf_node_col,
                        weight_col=leaf_weight_col,
                    )
                    weights_list.append(w)
                except Exception as e:
                    print(f"[WARN] skip {leaf_path}: {e}")

            if not weights_list:
                print(f"[WARN] no usable repeats for year={year} combo={combo_name}")
                continue

            avg_weights = average_weights_across_repeats(weights_list)
            turnover, _ = compute_monthly_turnover(
                weights_df=avg_weights[["date", "permno", "node_id", "w"]],
                stock_ret_df=stock_ret_df,
                stock_ret_date_col=ret_date_col,
                stock_ret_id_col=ret_stock_col,
                stock_ret_col=ret_col,
            )
            if turnover.empty:
                print(f"[WARN] empty turnover for year={year} combo={combo_name}")
                continue

            turnover["combo"] = combo_name
            turnover["year"] = int(year)
            year_turnovers.append(turnover)

            value = float(turnover["turnover"].mean())
            year_values.append(pd.DataFrame({
                "year": [int(year)],
                "combo": [combo_name],
                "portfolio_avg_turnover": [value],
            }))

            print(f"[OK] year={year} combo={combo_name} repeats_used={len(weights_list)} value={value:.6f}")

        if year_turnovers:
            year_turnover_df = pd.concat(year_turnovers, ignore_index=True)
            year_turnover_df.to_parquet(year_out_dir / "monthly_turnover_avg_repeats.parquet", index=False)
            all_monthly_turnover.append(year_turnover_df)

        if year_values:
            year_value_df = pd.concat(year_values, ignore_index=True)
            year_value_df.to_parquet(year_out_dir / "portfolio_turnover_value.parquet", index=False)
            all_values.append(year_value_df)

        print(f"[DONE] Wrote outputs to {year_out_dir}")

    if not all_monthly_turnover:
        raise ValueError("No monthly turnover was produced.")

    all_monthly_turnover_df = pd.concat(all_monthly_turnover, ignore_index=True)
    all_monthly_turnover_df.to_parquet(out_dir / "all_monthly_turnover_avg_repeats.parquet", index=False)

    if all_values:
        all_values_df = pd.concat(all_values, ignore_index=True)
        all_values_df.to_parquet(out_dir / "all_portfolio_turnover_values.parquet", index=False)

    final_turnover = float(all_monthly_turnover_df["turnover"].mean())
    pd.DataFrame({"final_turnover": [final_turnover]}).to_parquet(out_dir / "final_turnover.parquet", index=False)
    with open(out_dir / "final_turnover.txt", "w") as f:
        f.write(f"{final_turnover}\n")

    print(f"[FINAL] Option A full-sample average monthly turnover = {final_turnover:.10f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-parent", required=True)
    parser.add_argument("--stock-ret-path", required=True)
    parser.add_argument("--stock-ret-format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--leaf-date-col", required=True)
    parser.add_argument("--leaf-stock-col", required=True)
    parser.add_argument("--leaf-node-col", default=None)
    parser.add_argument("--leaf-weight-col", default=None)

    parser.add_argument("--ret-date-col", required=True)
    parser.add_argument("--ret-stock-col", required=True)
    parser.add_argument("--ret-col", required=True)

    args = parser.parse_args()

    if args.stock_ret_format == "parquet":
        stock_ret_df = pd.read_parquet(args.stock_ret_path)
    else:
        stock_ret_df = pd.read_csv(args.stock_ret_path)

    stock_ret_df[args.ret_date_col] = pd.to_datetime(stock_ret_df[args.ret_date_col])

    process_all_years(
        root_parent=args.root_parent,
        stock_ret_df=stock_ret_df,
        out_dir=args.out_dir,
        leaf_date_col=args.leaf_date_col,
        leaf_stock_col=args.leaf_stock_col,
        leaf_node_col=args.leaf_node_col,
        leaf_weight_col=args.leaf_weight_col,
        ret_date_col=args.ret_date_col,
        ret_stock_col=args.ret_stock_col,
        ret_col=args.ret_col,
    )

if __name__ == "__main__":
    main()
