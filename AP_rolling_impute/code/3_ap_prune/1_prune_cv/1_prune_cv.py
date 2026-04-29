import os
import argparse
import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
FILTER_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/3_filter/output"
PRUNE_OUTPUT_BASE_DIR = "../../../data/3_ap_prune/1_prune_cv/output"

DATE_COL = "date"

# keep the same lambda grids as before
# LAMBDA0_GRID = np.arange(0.0, 0.9001, 0.05)
# LAMBDA2_GRID = 0.1 ** np.arange(5.0, 8.0001, 0.25)

LAMBDA0_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
LAMBDA2_GRID = np.array([1e-5, 1e-6, 1e-7, 1e-8])


# =========================================================
# helpers
# =========================================================
def make_repeat_tag(repeat_idx: int, seed: int, B: int, depth: int):
    return f"repeat_{repeat_idx:02d}_seed_{seed}_B_{B}_depth_{depth}"


def build_depth_map(metadata: pd.DataFrame):
    depth_map = {}
    for _, row in metadata.iterrows():
        col = row["column_name"]
        node_id = str(row["node_id"])
        depth = len(node_id) - 1
        depth_map[col] = depth
    return depth_map


def compute_adj_w(columns, depth_map):
    adj_w = np.ones(len(columns), dtype=float)
    for i, col in enumerate(columns):
        depth = depth_map.get(col, 0)
        adj_w[i] = 1.0 / np.sqrt(2 ** depth)
    return adj_w


def load_ports_and_weights(port_path: str, meta_path: str, date_col: str = DATE_COL):
    """
    这里只负责读取，不再要求全样本完全无 NaN。
    """
    ports = pd.read_pickle(port_path).copy()
    metadata = pd.read_csv(meta_path).copy()

    if date_col not in ports.columns:
        raise ValueError(f"Missing date column: {date_col}")

    ports[date_col] = pd.to_datetime(ports[date_col])
    ports = ports.sort_values(date_col).reset_index(drop=True)

    dates = ports[date_col].copy()
    ports = ports.drop(columns=[date_col])

    depth_map = build_depth_map(metadata)
    adj_w = compute_adj_w(list(ports.columns), depth_map)

    return dates, ports, metadata, adj_w


def safe_sr(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return np.nan
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    mu = np.mean(x)
    return mu / sd


def rolling_window_ranges(n_total: int, n_train_valid: int, test_months: int, step_months: int):
    windows = []

    max_start = n_total - n_train_valid - test_months
    if max_start < 0:
        return windows

    for start in range(0, max_start + 1, step_months):
        train_valid_start = start
        train_valid_end = start + n_train_valid - 1

        test_start = start + n_train_valid
        test_end = test_start + test_months - 1

        windows.append({
            "train_valid_start": train_valid_start,
            "train_valid_end": train_valid_end,
            "test_start": test_start,
            "test_end": test_end,
        })

    return windows


def filter_window_complete_columns(ports_window_df: pd.DataFrame, metadata: pd.DataFrame, adj_w: np.ndarray):
    """
    对当前 window，只保留完全无 NaN 的列。
    同时同步过滤 metadata 和 adj_w。
    """
    keep_cols = [c for c in ports_window_df.columns if ports_window_df[c].notna().all()]

    if len(keep_cols) == 0:
        return None, None, None

    ports_window_df = ports_window_df[keep_cols].copy()

    keep_col_set = set(keep_cols)
    metadata_window = metadata[metadata["column_name"].isin(keep_col_set)].copy()

    metadata_window["column_name"] = pd.Categorical(
        metadata_window["column_name"],
        categories=keep_cols,
        ordered=True
    )
    metadata_window = metadata_window.sort_values("column_name").reset_index(drop=True)
    metadata_window["column_name"] = metadata_window["column_name"].astype(str)

    original_cols = list(metadata["column_name"])
    col_to_idx = {c: i for i, c in enumerate(original_cols)}
    adj_idx = [col_to_idx[c] for c in keep_cols]
    adj_w_window = adj_w[adj_idx]

    # 对齐检查
    if keep_cols != metadata_window["column_name"].tolist():
        raise ValueError("Window-level columns and metadata are not aligned.")

    return ports_window_df, metadata_window, adj_w_window


# =========================================================
# core prune logic
# =========================================================
def run_one_split(
    ports_train,
    ports_valid,
    ports_test,
    adj_w,
    lambda0,
    lambda2,
    kmin,
    kmax,
    lambda0_idx,
    lambda2_idx,
):
    """
    Return a DataFrame with columns:
        lambda0_idx, lambda0, lambda2_idx, lambda2,
        train_SR, valid_SR, test_SR, portsN, beta_*
    """
    from sklearn.linear_model import lars_path

    ports_train = np.asarray(ports_train, dtype=float)
    ports_test = np.asarray(ports_test, dtype=float)

    p = ports_train.shape[1]

    mu = ports_train.mean(axis=0)
    sigma = np.cov(ports_train, rowvar=False)

    # if p=1, np.cov returns scalar
    if np.ndim(sigma) == 0:
        sigma = np.array([[float(sigma)]])

    mu_bar = mu.mean()
    sigma_tilde = sigma + lambda2 * np.eye(p)
    mu_tilde = mu + lambda0 * mu_bar

    eigvals, eigvecs = np.linalg.eigh(sigma_tilde)
    keep = eigvals > 1e-10
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]

    if eigvals.size == 0:
        return pd.DataFrame()

    sigma_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    sigma_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    y = sigma_inv_sqrt @ mu_tilde
    X = sigma_sqrt

    _, _, coefs = lars_path(X, y, method="lasso", verbose=False)

    K = (np.abs(coefs) > 1e-12).sum(axis=0)
    keep_k = (K >= kmin) & (K <= kmax)

    coefs = coefs[:, keep_k].T
    K = K[keep_k]

    if coefs.shape[0] == 0:
        return pd.DataFrame()

    rows = []

    if ports_valid is not None:
        ports_valid = np.asarray(ports_valid, dtype=float)

    for i, b in enumerate(coefs):
        b = b * adj_w
        denom = np.abs(b).sum()
        if denom <= 0:
            continue
        b = b / denom

        sdf_train = ports_train @ (b / adj_w)
        train_sr = safe_sr(sdf_train)

        sdf_test = ports_test @ (b / adj_w)
        test_sr = safe_sr(sdf_test)

        if ports_valid is not None:
            sdf_valid = ports_valid @ (b / adj_w)
            valid_sr = safe_sr(sdf_valid)
        else:
            valid_sr = np.nan

        row = {
            "lambda0_idx": lambda0_idx,
            "lambda0": lambda0,
            "lambda2_idx": lambda2_idx,
            "lambda2": lambda2,
            "train_SR": train_sr,
            "valid_SR": valid_sr,
            "test_SR": test_sr,
            "portsN": int(K[i]),
        }

        for j in range(len(b)):
            row[f"beta_{j}"] = b[j]

        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def prune_one_window_all_lambda(
    ports_window_df,
    dates_window,
    metadata_window,
    adj_w_window,
    window_dir,
    n_train_valid=360,
    cvN=3,
    kmin=5,
    kmax=50,
    lambda0_grid=LAMBDA0_GRID,
    lambda2_grid=LAMBDA2_GRID,
):
    """
    run_full_cv is always TRUE by design.
    """
    os.makedirs(window_dir, exist_ok=True)

    # 保存 window-level metadata，这个后面 selection 必须用它来对 beta_i 做映射
    metadata_window.to_csv(os.path.join(window_dir, "portfolio_metadata_window.csv"), index=False)

    ports_window = ports_window_df.to_numpy(dtype=float)
    n_total, p = ports_window.shape

    if n_total <= n_train_valid:
        raise ValueError("Window too short: test period is empty.")

    ports_test = ports_window[n_train_valid:, :]
    n_valid = n_train_valid // cvN

    info_df = pd.DataFrame([{
        "window_n_total": n_total,
        "n_train_valid": n_train_valid,
        "n_test": n_total - n_train_valid,
        "n_portfolios_window": p,
        "window_start_date": pd.to_datetime(dates_window.iloc[0]),
        "train_valid_start_date": pd.to_datetime(dates_window.iloc[0]),
        "train_valid_end_date": pd.to_datetime(dates_window.iloc[n_train_valid - 1]),
        "test_start_date": pd.to_datetime(dates_window.iloc[n_train_valid]),
        "test_end_date": pd.to_datetime(dates_window.iloc[-1]),
        "cvN": cvN,
        "kmin": kmin,
        "kmax": kmax,
        "n_lambda0": len(lambda0_grid),
        "n_lambda2": len(lambda2_grid),
        "run_full_cv": False, # True
    }])
    info_df.to_csv(os.path.join(window_dir, "window_info.csv"), index=False)

    for i0, lambda0 in enumerate(lambda0_grid, start=1):
        for i2, lambda2 in enumerate(lambda2_grid, start=1):
            print(f"[INFO]   lambda0_idx={i0}, lambda2_idx={i2}, lambda0={lambda0:.4f}, lambda2={lambda2:.6g}")

            # full CV always on
            for i in range(cvN):
                valid_idx = np.arange(i * n_valid, (i + 1) * n_valid)
                train_idx = np.setdiff1d(np.arange(n_train_valid), valid_idx)

                ports_train = ports_window[train_idx, :]
                ports_valid = ports_window[valid_idx, :]

                results = run_one_split(
                    ports_train=ports_train,
                    ports_valid=ports_valid,
                    ports_test=ports_test,
                    adj_w=adj_w_window,
                    lambda0=lambda0,
                    lambda2=lambda2,
                    kmin=kmin,
                    kmax=kmax,
                    lambda0_idx=i0,
                    lambda2_idx=i2,
                )
                results.to_csv(
                    os.path.join(window_dir, f"results_cv_{i + 1}_l0_{i0}_l2_{i2}.csv"),
                    index=False
                )

            # full train+valid
            ports_train_full = ports_window[:n_train_valid, :]
            results_full = run_one_split(
                ports_train=ports_train_full,
                ports_valid=None,
                ports_test=ports_test,
                adj_w=adj_w_window,
                lambda0=lambda0,
                lambda2=lambda2,
                kmin=kmin,
                kmax=kmax,
                lambda0_idx=i0,
                lambda2_idx=i2,
            )
            results_full.to_csv(
                os.path.join(window_dir, f"results_full_l0_{i0}_l2_{i2}.csv"),
                index=False
            )


def main(args):
    B = args.B
    depth = args.depth
    repeat_idx = args.repeat_idx
    window_idx = args.window_idx
    base_seed = args.base_seed

    seed = base_seed + repeat_idx - 1
    repeat_tag = make_repeat_tag(repeat_idx, seed, B, depth)

    port_path = os.path.join(
        FILTER_OUTPUT_BASE_DIR,
        repeat_tag,
        "level_all_excess_ret_combined_filtered.pkl.gz"
    )
    meta_path = os.path.join(
        FILTER_OUTPUT_BASE_DIR,
        repeat_tag,
        "portfolio_metadata_filtered.csv"
    )
    output_dir = os.path.join(PRUNE_OUTPUT_BASE_DIR, repeat_tag)

    dates, ports_df, metadata, adj_w = load_ports_and_weights(
        port_path=port_path,
        meta_path=meta_path,
        date_col=DATE_COL,
    )

    n_total = len(ports_df)
    windows = rolling_window_ranges(
        n_total=n_total,
        n_train_valid=args.n_train_valid,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    if len(windows) == 0:
        raise ValueError(
            f"No valid rolling windows for repeat {repeat_tag}. "
            f"Need at least n_train_valid + test_months rows."
        )

    if window_idx < 1 or window_idx > len(windows):
        raise ValueError(
            f"window_idx={window_idx} is out of range. "
            f"Valid range: 1 to {len(windows)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # repeat-level metadata 和 window summary 保留
    metadata.to_csv(os.path.join(output_dir, "portfolio_metadata_filtered.csv"), index=False)

    summary_rows = []
    for w_idx, w in enumerate(windows, start=1):
        tv_start = w["train_valid_start"]
        tv_end = w["train_valid_end"]
        test_start = w["test_start"]
        test_end = w["test_end"]

        summary_rows.append({
            "window_idx": w_idx,
            "window_id": f"window_{w_idx:04d}",
            "train_valid_start_idx": tv_start,
            "train_valid_end_idx": tv_end,
            "test_start_idx": test_start,
            "test_end_idx": test_end,
            "train_valid_start_date": dates.iloc[tv_start],
            "train_valid_end_date": dates.iloc[tv_end],
            "test_start_date": dates.iloc[test_start],
            "test_end_date": dates.iloc[test_end],
        })

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "rolling_window_summary.csv"),
        index=False
    )

    # 当前 window
    w = windows[window_idx - 1]
    tv_start = w["train_valid_start"]
    tv_end = w["train_valid_end"]
    test_start = w["test_start"]
    test_end = w["test_end"]

    ports_window_df = ports_df.iloc[tv_start:test_end + 1, :].copy()
    dates_window = dates.iloc[tv_start:test_end + 1].reset_index(drop=True)

    # 这里做 window-level 对齐
    ports_window_df, metadata_window, adj_w_window = filter_window_complete_columns(
        ports_window_df=ports_window_df,
        metadata=metadata,
        adj_w=adj_w
    )

    if ports_window_df is None or ports_window_df.shape[1] == 0:
        raise ValueError(
            f"No complete portfolio columns remain in repeat={repeat_idx}, window={window_idx}."
        )

    window_tag = f"window_{window_idx:04d}"
    window_dir = os.path.join(output_dir, window_tag)
    os.makedirs(window_dir, exist_ok=True)

    print("=" * 100)
    print(f"[INFO] Repeat tag : {repeat_tag}")
    print(f"[INFO] Window tag : {window_tag}")
    print(f"[INFO] Train+valid: {dates.iloc[tv_start]} -> {dates.iloc[tv_end]}")
    print(f"[INFO] Test       : {dates.iloc[test_start]} -> {dates.iloc[test_end]}")
    print(f"[INFO] Input ports shape (full repeat): {ports_df.shape}")
    print(f"[INFO] Window shape before column filter: {(test_end - tv_start + 1, ports_df.shape[1])}")
    print(f"[INFO] Window shape after column filter : {ports_window_df.shape}")
    print(f"[INFO] n_train_valid    : {args.n_train_valid}")
    print(f"[INFO] test_months      : {args.test_months}")
    print(f"[INFO] step_months      : {args.step_months}")
    print(f"[INFO] cvN              : {args.cvN}")
    print(f"[INFO] kmin-kmax        : {args.kmin}-{args.kmax}")
    print(f"[INFO] n_lambda0        : {len(LAMBDA0_GRID)}")
    print(f"[INFO] n_lambda2        : {len(LAMBDA2_GRID)}")
    print(f"[INFO] run_full_cv      : False")

    prune_one_window_all_lambda(
        ports_window_df=ports_window_df,
        dates_window=dates_window,
        metadata_window=metadata_window,
        adj_w_window=adj_w_window,
        window_dir=window_dir,
        n_train_valid=args.n_train_valid,
        cvN=args.cvN,
        kmin=args.kmin,
        kmax=args.kmax,
        lambda0_grid=LAMBDA0_GRID,
        lambda2_grid=LAMBDA2_GRID,
    )

    print("=" * 100)
    print("[INFO] Prune CV finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)

    # parallelization dimensions
    parser.add_argument("--repeat_idx", type=int, required=True)
    parser.add_argument("--window_idx", type=int, required=True)

    parser.add_argument("--base_seed", type=int, default=0)

    # yearly rolling by default:
    # 30 years train+valid, 1 year test, update once per year
    parser.add_argument("--n_train_valid", type=int, default=360)
    parser.add_argument("--test_months", type=int, default=12)
    parser.add_argument("--step_months", type=int, default=12)

    parser.add_argument("--cvN", type=int, default=3)
    parser.add_argument("--kmin", type=int, default=5)
    parser.add_argument("--kmax", type=int, default=50)

    args = parser.parse_args()
    main(args)