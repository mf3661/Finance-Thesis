import os
import argparse
import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
PRUNE_RESULT_BASE_DIR = "../../../data/3_ap_prune/1_prune_cv/output"
FILTER_RESULT_BASE_DIR = "../../../data/2_tree_construction/3_filter/output"
SELECTION_OUTPUT_BASE_DIR = "../../../data/3_ap_prune/2_selection/output"

DATE_COL = "date"

# keep the same lambda grids as before
LAMBDA0_GRID = np.arange(0.0, 0.9001, 0.05)
LAMBDA2_GRID = 0.1 ** np.arange(5.0, 8.0001, 0.25)


# =========================================================
# helpers
# =========================================================
def make_repeat_tag(repeat_idx: int, seed: int, B: int, depth: int):
    return f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"


def make_window_tag(window_idx: int):
    return f"window_{window_idx:04d}"


def read_result_file(window_dir: str, cv_name: str, l0_idx: int, l2_idx: int) -> pd.DataFrame:
    path = os.path.join(window_dir, f"results_{cv_name}_l0_{l0_idx}_l2_{l2_idx}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")
    return pd.read_csv(path)


def read_ports_file(filter_dir: str):
    path = os.path.join(filter_dir, "level_all_excess_ret_combined_filtered.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing filtered ports file: {path}")
    return pd.read_csv(path)


def read_window_summary(prune_repeat_dir: str):
    path = os.path.join(prune_repeat_dir, "rolling_window_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing rolling window summary: {path}")
    return pd.read_csv(path)


def get_best_row_for_k(df: pd.DataFrame, portN: int, score_col: str):
    """
    For a fixed K, there may be multiple rows on the path.
    Keep the one with the best score_col.
    """
    sub = df[df["portsN"] == portN].copy()
    if sub.empty:
        return None

    sub = sub[np.isfinite(sub[score_col])].copy()
    if sub.empty:
        return None

    best_idx = sub[score_col].idxmax()
    return sub.loc[best_idx]


def average_valid_sr_for_one_k(window_dir: str, l0_idx: int, l2_idx: int, portN: int):
    """
    For a fixed (lambda0, lambda2, K):
    - in each cv file, if K appears multiple times, keep the row with best valid_SR
    - then average across cv_1, cv_2, cv_3
    """
    vals = []
    for cv_name in ["cv_1", "cv_2", "cv_3"]:
        df = read_result_file(window_dir, cv_name, l0_idx, l2_idx)
        row = get_best_row_for_k(df, portN, score_col="valid_SR")
        if row is None:
            return np.nan
        vals.append(row["valid_SR"])
    return float(np.mean(vals))


def get_best_full_row_for_k(window_dir: str, l0_idx: int, l2_idx: int, portN: int):
    """
    In the full file, for fixed K there may also be multiple rows.
    We keep the one with the best test_SR.
    """
    df = read_result_file(window_dir, "full", l0_idx, l2_idx)
    row = get_best_row_for_k(df, portN, score_col="test_SR")
    return row, df


# =========================================================
# version 1: conditional on K
# =========================================================
def pick_best_lambda_for_k(
    window_dir: str,
    filter_dir: str,
    portN: int,
    lambda0_grid=LAMBDA0_GRID,
    lambda2_grid=LAMBDA2_GRID,
    write_table=True,
    output_dir=None,
    repeat_idx=None,
    window_idx=None,
):
    """
    For a fixed K=portN, search over all (lambda0, lambda2) and pick
    the one with highest average validation SR.
    If the same K appears multiple times within one file, keep the best one.
    """
    n_l0 = len(lambda0_grid)
    n_l2 = len(lambda2_grid)

    train_SR = np.full((n_l0, n_l2), np.nan)
    valid_SR = np.full((n_l0, n_l2), np.nan)
    test_SR = np.full((n_l0, n_l2), np.nan)

    best_full_rows = {}

    for i in range(1, n_l0 + 1):
        for j in range(1, n_l2 + 1):
            full_row, full_df = get_best_full_row_for_k(window_dir, i, j, portN)
            if full_row is None:
                continue

            train_SR[i - 1, j - 1] = full_row["train_SR"]
            test_SR[i - 1, j - 1] = full_row["test_SR"]

            valid_SR[i - 1, j - 1] = average_valid_sr_for_one_k(
                window_dir=window_dir,
                l0_idx=i,
                l2_idx=j,
                portN=portN
            )

            best_full_rows[(i, j)] = (full_row, full_df)

    if np.all(~np.isfinite(valid_SR)):
        raise ValueError(f"No valid model found for portN={portN}")

    best_idx = np.nanargmax(valid_SR)
    best_i0, best_j0 = np.unravel_index(best_idx, valid_SR.shape)
    best_i = best_i0 + 1
    best_j = best_j0 + 1

    best_lambda0 = lambda0_grid[best_i0]
    best_lambda2 = lambda2_grid[best_j0]

    best_row, best_full_df = best_full_rows[(best_i, best_j)]

    meta_cols = {
        "lambda0_idx", "lambda0",
        "lambda2_idx", "lambda2",
        "train_SR", "valid_SR", "test_SR",
        "portsN"
    }
    beta_cols = [c for c in best_full_df.columns if c not in meta_cols]
    weights = best_row[beta_cols].astype(float)

    # only keep non-zero selected ports
    nonzero_mask = weights != 0
    weights_out = weights[nonzero_mask].copy()

    ports = read_ports_file(filter_dir)
    date_col = DATE_COL if DATE_COL in ports.columns else None

    selected_beta_cols = list(weights_out.index)

    # beta_i column index -> actual portfolio column name
    selected_port_names = []
    for beta_col in selected_beta_cols:
        beta_idx = int(beta_col.replace("beta_", ""))
        selected_port_names.append(ports.columns[1 + beta_idx] if date_col is not None else ports.columns[beta_idx])

    keep_cols = selected_port_names.copy()
    if date_col is not None:
        keep_cols = [date_col] + keep_cols

    selected_ports = ports[keep_cols].copy()

    weights_table = pd.DataFrame({
        "beta_col": selected_beta_cols,
        "column_name": selected_port_names,
        "weight": weights_out.values
    })

    summary = {
        "repeat_idx": repeat_idx,
        "window_idx": window_idx,
        "portN": portN,
        "best_lambda0_idx": best_i,
        "best_lambda2_idx": best_j,
        "best_lambda0": best_lambda0,
        "best_lambda2": best_lambda2,
        "best_train_SR": train_SR[best_i0, best_j0],
        "best_valid_SR": valid_SR[best_i0, best_j0],
        "best_test_SR": test_SR[best_i0, best_j0],
        "n_selected_ports": int((weights != 0).sum()),
    }

    if write_table and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        pd.DataFrame(train_SR).to_csv(
            os.path.join(output_dir, f"train_SR_{portN}.csv"),
            index=False
        )
        pd.DataFrame(valid_SR).to_csv(
            os.path.join(output_dir, f"valid_SR_{portN}.csv"),
            index=False
        )
        pd.DataFrame(test_SR).to_csv(
            os.path.join(output_dir, f"test_SR_{portN}.csv"),
            index=False
        )

        selected_ports.to_csv(
            os.path.join(output_dir, f"Selected_Ports_{portN}.csv"),
            index=False
        )

        weights_table.to_csv(
            os.path.join(output_dir, f"Selected_Ports_Weights_{portN}.csv"),
            index=False
        )

        pd.DataFrame([summary]).to_csv(
            os.path.join(output_dir, f"Best_Lambda_Summary_{portN}.csv"),
            index=False
        )

    return summary


def pick_sr_n(
    window_dir: str,
    filter_dir: str,
    output_dir: str,
    mink: int,
    maxk: int,
    lambda0_grid=LAMBDA0_GRID,
    lambda2_grid=LAMBDA2_GRID,
    repeat_idx=None,
    window_idx=None,
):
    """
    Original version:
    For each K, pick the best lambda pair.
    If K appears multiple times on the path, keep the best-valid one.
    """
    rows = []

    for k in range(mink, maxk + 1):
        print(f"[INFO] Picking best lambda for portN={k}")
        try:
            res = pick_best_lambda_for_k(
                window_dir=window_dir,
                filter_dir=filter_dir,
                portN=k,
                lambda0_grid=lambda0_grid,
                lambda2_grid=lambda2_grid,
                write_table=True,
                output_dir=output_dir,
                repeat_idx=repeat_idx,
                window_idx=window_idx,
            )
            rows.append(res)
        except Exception as e:
            print(f"[WARNING] portN={k} failed: {e}")

    if len(rows) == 0:
        raise ValueError("No valid K found in the whole range.")

    sr_n_df = pd.DataFrame(rows)
    sr_n_df.to_csv(os.path.join(output_dir, "SR_N.csv"), index=False)

    print(f"[INFO] Saved SR_N.csv to: {output_dir}")
    return sr_n_df


# =========================================================
# version 2: regardless of K
# =========================================================
def pick_overall_best_model(
    window_dir: str,
    filter_dir: str,
    output_dir: str,
    mink: int,
    maxk: int,
    lambda0_grid=LAMBDA0_GRID,
    lambda2_grid=LAMBDA2_GRID,
    repeat_idx=None,
    window_idx=None,
):
    """
    For each (lambda0, lambda2, K), first compress duplicate K rows by keeping
    the best-valid one in each cv file and the best-test one in full.
    Then search over all (lambda0, lambda2, K) and pick the single model with
    highest validation SR, regardless of K.
    """
    best_obj = None

    for i in range(1, len(lambda0_grid) + 1):
        for j in range(1, len(lambda2_grid) + 1):
            try:
                full_df = read_result_file(window_dir, "full", i, j)
                cv1_df = read_result_file(window_dir, "cv_1", i, j)
                cv2_df = read_result_file(window_dir, "cv_2", i, j)
                cv3_df = read_result_file(window_dir, "cv_3", i, j)
            except Exception as e:
                print(f"[WARNING] Missing lambda results for l0={i}, l2={j}: {e}")
                continue

            # candidate K values from full file
            candidate_k = sorted(set(full_df["portsN"].astype(int).tolist()))

            for portN in candidate_k:
                if portN < mink or portN > maxk:
                    continue

                full_row = get_best_row_for_k(full_df, portN, score_col="test_SR")
                cv1_row = get_best_row_for_k(cv1_df, portN, score_col="valid_SR")
                cv2_row = get_best_row_for_k(cv2_df, portN, score_col="valid_SR")
                cv3_row = get_best_row_for_k(cv3_df, portN, score_col="valid_SR")

                if full_row is None or cv1_row is None or cv2_row is None or cv3_row is None:
                    continue

                valid_sr = np.mean([
                    cv1_row["valid_SR"],
                    cv2_row["valid_SR"],
                    cv3_row["valid_SR"],
                ])

                candidate = {
                    "repeat_idx": repeat_idx,
                    "window_idx": window_idx,
                    "portN": portN,
                    "best_lambda0_idx": i,
                    "best_lambda2_idx": j,
                    "best_lambda0": lambda0_grid[i - 1],
                    "best_lambda2": lambda2_grid[j - 1],
                    "best_train_SR": full_row["train_SR"],
                    "best_valid_SR": valid_sr,
                    "best_test_SR": full_row["test_SR"],
                    "full_row": full_row,
                    "full_df": full_df,
                }

                if (best_obj is None) or (candidate["best_valid_SR"] > best_obj["best_valid_SR"]):
                    best_obj = candidate

    if best_obj is None:
        raise ValueError("No valid overall-best model found.")

    full_row = best_obj["full_row"]
    full_df = best_obj["full_df"]

    meta_cols = {
        "lambda0_idx", "lambda0",
        "lambda2_idx", "lambda2",
        "train_SR", "valid_SR", "test_SR",
        "portsN"
    }
    beta_cols = [c for c in full_df.columns if c not in meta_cols]
    weights = full_row[beta_cols].astype(float)

    nonzero_mask = weights != 0
    weights_out = weights[nonzero_mask].copy()

    ports = read_ports_file(filter_dir)
    date_col = DATE_COL if DATE_COL in ports.columns else None

    selected_beta_cols = list(weights_out.index)
    selected_port_names = []
    for beta_col in selected_beta_cols:
        beta_idx = int(beta_col.replace("beta_", ""))
        selected_port_names.append(ports.columns[1 + beta_idx] if date_col is not None else ports.columns[beta_idx])

    keep_cols = selected_port_names.copy()
    if date_col is not None:
        keep_cols = [date_col] + keep_cols

    selected_ports = ports[keep_cols].copy()

    weights_table = pd.DataFrame({
        "beta_col": selected_beta_cols,
        "column_name": selected_port_names,
        "weight": weights_out.values
    })

    summary = {
        "repeat_idx": repeat_idx,
        "window_idx": window_idx,
        "best_portN": best_obj["portN"],
        "best_lambda0_idx": best_obj["best_lambda0_idx"],
        "best_lambda2_idx": best_obj["best_lambda2_idx"],
        "best_lambda0": best_obj["best_lambda0"],
        "best_lambda2": best_obj["best_lambda2"],
        "best_train_SR": best_obj["best_train_SR"],
        "best_valid_SR": best_obj["best_valid_SR"],
        "best_test_SR": best_obj["best_test_SR"],
        "n_selected_ports": int((weights != 0).sum()),
    }

    os.makedirs(output_dir, exist_ok=True)

    selected_ports.to_csv(
        os.path.join(output_dir, "Selected_Ports_OverallBest.csv"),
        index=False
    )
    weights_table.to_csv(
        os.path.join(output_dir, "Selected_Ports_Weights_OverallBest.csv"),
        index=False
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "Overall_Best_Summary.csv"),
        index=False
    )

    return summary


# =========================================================
# main
# =========================================================
def main(args):
    B = args.B
    depth = args.depth
    repeat_idx = args.repeat_idx
    window_idx = args.window_idx
    base_seed = args.base_seed

    seed = base_seed + repeat_idx - 1
    repeat_tag = make_repeat_tag(repeat_idx, seed, B, depth)
    window_tag = make_window_tag(window_idx)

    prune_repeat_dir = os.path.join(PRUNE_RESULT_BASE_DIR, repeat_tag)
    prune_window_dir = os.path.join(prune_repeat_dir, window_tag)

    filter_dir = os.path.join(FILTER_RESULT_BASE_DIR, repeat_tag)

    output_dir = os.path.join(SELECTION_OUTPUT_BASE_DIR, repeat_tag, window_tag)
    os.makedirs(output_dir, exist_ok=True)

    # save window summary copy for convenience
    window_summary = read_window_summary(prune_repeat_dir)
    window_summary.to_csv(os.path.join(os.path.dirname(output_dir), "rolling_window_summary.csv"), index=False)

    print("=" * 100)
    print(f"[INFO] Repeat tag : {repeat_tag}")
    print(f"[INFO] Window tag : {window_tag}")
    print(f"[INFO] Prune dir   : {prune_window_dir}")
    print(f"[INFO] Filter dir  : {filter_dir}")
    print(f"[INFO] Output dir  : {output_dir}")

    # version 1: conditional on K
    sr_n_df = pick_sr_n(
        window_dir=prune_window_dir,
        filter_dir=filter_dir,
        output_dir=output_dir,
        mink=args.kmin,
        maxk=args.kmax,
        lambda0_grid=LAMBDA0_GRID,
        lambda2_grid=LAMBDA2_GRID,
        repeat_idx=repeat_idx,
        window_idx=window_idx,
    )

    # version 2: regardless of K
    overall_summary = pick_overall_best_model(
        window_dir=prune_window_dir,
        filter_dir=filter_dir,
        output_dir=output_dir,
        mink=args.kmin,
        maxk=args.kmax,
        lambda0_grid=LAMBDA0_GRID,
        lambda2_grid=LAMBDA2_GRID,
        repeat_idx=repeat_idx,
        window_idx=window_idx,
    )

    print("=" * 100)
    print("[INFO] Selection finished successfully")
    print("[INFO] Conditional-on-K summary head:")
    print(sr_n_df.head())
    print("[INFO] Overall-best summary:")
    print(pd.DataFrame([overall_summary]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)

    # parallelization dimensions
    parser.add_argument("--repeat_idx", type=int, required=True)
    parser.add_argument("--window_idx", type=int, required=True)

    parser.add_argument("--base_seed", type=int, default=0)

    parser.add_argument("--kmin", type=int, default=5)
    parser.add_argument("--kmax", type=int, default=50)

    args = parser.parse_args()
    main(args)