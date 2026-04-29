import os
import argparse
import numpy as np
import pandas as pd


# =========================================================
# paths
# =========================================================
FILTER_OUTPUT_BASE_DIR = "../../../data/2_tree_construction/2_3_filter/2_3_5_filter_ptree/output"
PRUNE_OUTPUT_BASE_DIR = "../../../data/3_ap_prune/3_1_prune_cv/3_1_5_prune_cv_ptree/output"

DATE_COL = "date"

# reduced lambda grids
LAMBDA0_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7])
LAMBDA2_GRID = np.array([1e-5, 1e-6, 1e-7, 1e-8])


# =========================================================
# helpers
# =========================================================
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
    Only read inputs. Do not require the full sample to be NaN-free.
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


def load_sample_map(sample_map_path: str):
    if not os.path.exists(sample_map_path):
        raise FileNotFoundError(f"Cannot find sample_map: {sample_map_path}")

    if sample_map_path.endswith(".parquet"):
        sample_map = pd.read_parquet(sample_map_path).copy()
    else:
        sample_map = pd.read_csv(sample_map_path).copy()

    if DATE_COL not in sample_map.columns or "sample" not in sample_map.columns:
        raise ValueError("sample_map must contain date and sample columns.")

    sample_map[DATE_COL] = pd.to_datetime(sample_map[DATE_COL])
    sample_map = sample_map.sort_values(DATE_COL).reset_index(drop=True)
    return sample_map


def safe_sr(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return np.nan
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    mu = np.mean(x)
    return mu / sd


def filter_window_complete_columns(ports_window_df: pd.DataFrame, metadata: pd.DataFrame, adj_w: np.ndarray):
    """
    For the current window, only keep columns with no NaN in the whole window.
    Also synchronize metadata and adj_w.
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

    if keep_cols != metadata_window["column_name"].tolist():
        raise ValueError("Window-level columns and metadata are not aligned.")

    return ports_window_df, metadata_window, adj_w_window


# =========================================================
# core prune logic (kept as close as possible to original)
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

    Depth scaling is applied before estimating moments:
        R_scaled = R_raw * adj_w.

    LARS coefficients are estimated in the scaled-return space. They are
    then mapped back to raw portfolio weights by
        beta_raw = beta_scaled * adj_w,
    and normalized by gross exposure sum(abs(beta_raw)).

    The saved beta_* columns are final raw portfolio weights that can be
    applied directly to the original, unscaled portfolio return matrix.
    """
    from sklearn.linear_model import lars_path

    ports_train = np.asarray(ports_train, dtype=float)
    ports_test = np.asarray(ports_test, dtype=float)
    adj_w = np.asarray(adj_w, dtype=float)

    if ports_valid is not None:
        ports_valid = np.asarray(ports_valid, dtype=float)

    p = ports_train.shape[1]

    if adj_w.shape[0] != p:
        raise ValueError(f"adj_w length {adj_w.shape[0]} does not match p={p}")

    # -----------------------------------------------------
    # 1. Apply depth scaling before computing mu and Sigma
    # -----------------------------------------------------
    ports_train_scaled = ports_train * adj_w
    ports_test_scaled = ports_test * adj_w

    if ports_valid is not None:
        ports_valid_scaled = ports_valid * adj_w
    else:
        ports_valid_scaled = None

    # -----------------------------------------------------
    # 2. Compute moments using scaled portfolio returns
    # -----------------------------------------------------
    mu = ports_train_scaled.mean(axis=0)
    sigma = np.cov(ports_train_scaled, rowvar=False)

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

    # Number of selected portfolios is invariant to multiplying by nonzero adj_w
    K = (np.abs(coefs) > 1e-12).sum(axis=0)
    keep_k = (K >= kmin) & (K <= kmax)

    coefs = coefs[:, keep_k].T
    K = K[keep_k]

    if coefs.shape[0] == 0:
        return pd.DataFrame()

    rows = []

    for i, beta_scaled in enumerate(coefs):
        # -----------------------------------------------------
        # 3. Convert scaled-space coefficients back to raw weights
        # -----------------------------------------------------
        beta_raw = beta_scaled * adj_w

        # Gross-exposure normalization: sum(abs(weights)), not abs(sum(weights))
        denom = np.abs(beta_raw).sum()
        if denom <= 0 or not np.isfinite(denom):
            continue

        beta_raw = beta_raw / denom

        # -----------------------------------------------------
        # 4. Compute SDF returns using raw portfolio returns
        # -----------------------------------------------------
        sdf_train = ports_train @ beta_raw
        train_sr = safe_sr(sdf_train)

        sdf_test = ports_test @ beta_raw
        test_sr = safe_sr(sdf_test)

        if ports_valid is not None:
            sdf_valid = ports_valid @ beta_raw
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

        # Save final raw portfolio weights
        for j in range(len(beta_raw)):
            row[f"beta_{j}"] = beta_raw[j]

        rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows)

def prune_one_window_all_lambda(
    ports_window_df,
    dates_window,
    metadata_window,
    adj_w_window,
    sample_map_window,
    window_dir,
    cvN=3,
    kmin=40,
    kmax=55,
    lambda0_grid=LAMBDA0_GRID,
    lambda2_grid=LAMBDA2_GRID,
):
    """
    Single fixed window:
      train = 240 months
      valid = 120 months
      test  = 12 months

    Keep output filenames/format as close as possible to original:
      - results_cv_3_l0_*_l2_*.csv
      - results_full_l0_*_l2_*.csv
      - window_info.csv
      - portfolio_metadata_window.csv
    """
    os.makedirs(window_dir, exist_ok=True)

    metadata_window.to_csv(os.path.join(window_dir, "portfolio_metadata_window.csv"), index=False)

    sample_map_window = sample_map_window.reset_index(drop=True).copy()
    dates_window = pd.to_datetime(dates_window).reset_index(drop=True)

    # alignment check
    if len(sample_map_window) != len(dates_window):
        raise ValueError("sample_map_window and dates_window length mismatch.")
    if not (pd.to_datetime(sample_map_window[DATE_COL]).reset_index(drop=True) == dates_window).all():
        raise ValueError("sample_map_window dates and dates_window are not aligned.")

    train_mask = sample_map_window["sample"] == "train"
    valid_mask = sample_map_window["sample"] == "valid"
    test_mask = sample_map_window["sample"] == "test"

    n_train = int(train_mask.sum())
    n_valid = int(valid_mask.sum())
    n_test = int(test_mask.sum())

    ports_window = ports_window_df.to_numpy(dtype=float)
    p = ports_window.shape[1]

    ports_train = ports_window[train_mask.values, :]
    ports_valid = ports_window[valid_mask.values, :]
    ports_test = ports_window[test_mask.values, :]
    ports_train_full = ports_window[train_mask.values | valid_mask.values, :]

    info_df = pd.DataFrame([{
        "window_n_total": len(sample_map_window),
        "n_train_valid": n_train + n_valid,
        "n_test": n_test,
        "n_portfolios_window": p,
        "window_start_date": pd.to_datetime(dates_window.iloc[0]),
        "train_valid_start_date": pd.to_datetime(dates_window.iloc[0]),
        "train_valid_end_date": pd.to_datetime(dates_window.iloc[n_train + n_valid - 1]),
        "test_start_date": pd.to_datetime(dates_window.iloc[n_train + n_valid]),
        "test_end_date": pd.to_datetime(dates_window.iloc[-1]),
        "cvN": cvN,
        "kmin": kmin,
        "kmax": kmax,
        "n_lambda0": len(lambda0_grid),
        "n_lambda2": len(lambda2_grid),
        "run_last_split_only": True,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test_exact": n_test,
    }])
    info_df.to_csv(os.path.join(window_dir, "window_info.csv"), index=False)

    # keep original style: only last split (cv_N) + full
    cv_name = f"cv_{cvN}"

    for i0, lambda0 in enumerate(lambda0_grid, start=1):
        for i2, lambda2 in enumerate(lambda2_grid, start=1):
            print(f"[INFO]   lambda0_idx={i0}, lambda2_idx={i2}, lambda0={lambda0:.4f}, lambda2={lambda2:.6g}")

            # only cv_N
            results_cv = run_one_split(
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
            results_cv.to_csv(
                os.path.join(window_dir, f"results_{cv_name}_l0_{i0}_l2_{i2}.csv"),
                index=False
            )

            # full train+valid
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


def prune_one_repeat(year: int, repeat_idx: int, filter_repeat_dir: str, output_repeat_dir: str, cvN: int, kmin: int, kmax: int):
    port_path = os.path.join(filter_repeat_dir, "level_all_excess_ret_combined_filtered.pkl.gz")
    meta_path = os.path.join(filter_repeat_dir, "portfolio_metadata_filtered.csv")

    sample_map_path_parquet = os.path.join(filter_repeat_dir, "sample_map.parquet")
    sample_map_path_csv = os.path.join(filter_repeat_dir, "sample_map.csv")
    if os.path.exists(sample_map_path_parquet):
        sample_map_path = sample_map_path_parquet
    elif os.path.exists(sample_map_path_csv):
        sample_map_path = sample_map_path_csv
    else:
        raise FileNotFoundError(f"Cannot find sample_map under {filter_repeat_dir}")

    dates, ports_df, metadata, adj_w = load_ports_and_weights(
        port_path=port_path,
        meta_path=meta_path,
        date_col=DATE_COL,
    )
    sample_map = load_sample_map(sample_map_path)

    # align by date
    ports_with_date = pd.concat([dates.rename(DATE_COL), ports_df], axis=1)
    merged = sample_map.merge(ports_with_date, on=DATE_COL, how="left", validate="one_to_one")
    merged = merged.sort_values(DATE_COL).reset_index(drop=True)

    if merged.drop(columns=[DATE_COL, "sample"]).isna().all(axis=None):
        raise ValueError(f"All merged portfolio values are NaN for year={year}, repeat={repeat_idx}")

    dates_window = merged[DATE_COL].copy()
    sample_map_window = merged[[DATE_COL, "sample"]].copy()
    ports_window_df = merged.drop(columns=[DATE_COL, "sample"]).copy()

    ports_window_df, metadata_window, adj_w_window = filter_window_complete_columns(
        ports_window_df=ports_window_df,
        metadata=metadata,
        adj_w=adj_w
    )

    if ports_window_df is None or ports_window_df.shape[1] == 0:
        raise ValueError(
            f"No complete portfolio columns remain in year={year}, repeat={repeat_idx}."
        )

    # keep original output feel: one fixed window_0001
    os.makedirs(output_repeat_dir, exist_ok=True)
    metadata.to_csv(os.path.join(output_repeat_dir, "portfolio_metadata_filtered.csv"), index=False)

    summary_rows = [{
        "window_idx": 1,
        "window_id": "window_0001",
        "train_valid_start_idx": 0,
        "train_valid_end_idx": int((sample_map_window["sample"].isin(["train", "valid"])).sum()) - 1,
        "test_start_idx": int((sample_map_window["sample"].isin(["train", "valid"])).sum()),
        "test_end_idx": len(sample_map_window) - 1,
        "train_valid_start_date": dates_window.iloc[0],
        "train_valid_end_date": dates_window.iloc[int((sample_map_window["sample"].isin(["train", "valid"])).sum()) - 1],
        "test_start_date": dates_window.iloc[int((sample_map_window["sample"].isin(["train", "valid"])).sum())],
        "test_end_date": dates_window.iloc[-1],
    }]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_repeat_dir, "rolling_window_summary.csv"),
        index=False
    )

    window_dir = os.path.join(output_repeat_dir, "window_0001")
    os.makedirs(window_dir, exist_ok=True)

    print("=" * 100)
    print(f"[INFO] OOS year      : {year}")
    print(f"[INFO] Repeat idx    : {repeat_idx}")
    print(f"[INFO] Window tag    : window_0001")
    print(f"[INFO] Train count   : {(sample_map_window['sample'] == 'train').sum()}")
    print(f"[INFO] Valid count   : {(sample_map_window['sample'] == 'valid').sum()}")
    print(f"[INFO] Test count    : {(sample_map_window['sample'] == 'test').sum()}")
    print(f"[INFO] Input ports shape (full repeat): {ports_df.shape}")
    print(f"[INFO] Window shape before column filter: {merged.drop(columns=[DATE_COL, 'sample']).shape}")
    print(f"[INFO] Window shape after column filter : {ports_window_df.shape}")
    print(f"[INFO] cvN              : {cvN}")
    print(f"[INFO] kmin-kmax        : {kmin}-{kmax}")
    print(f"[INFO] n_lambda0        : {len(LAMBDA0_GRID)}")
    print(f"[INFO] n_lambda2        : {len(LAMBDA2_GRID)}")
    print(f"[INFO] run_last_split_only: True")

    prune_one_window_all_lambda(
        ports_window_df=ports_window_df,
        dates_window=dates_window,
        metadata_window=metadata_window,
        adj_w_window=adj_w_window,
        sample_map_window=sample_map_window,
        window_dir=window_dir,
        cvN=cvN,
        kmin=kmin,
        kmax=kmax,
        lambda0_grid=LAMBDA0_GRID,
        lambda2_grid=LAMBDA2_GRID,
    )

    print("=" * 100)
    print(f"[INFO] Repeat {repeat_idx} prune finished successfully")


def main(args):
    year = args.year
    N = args.N

    filter_root = os.path.join(FILTER_OUTPUT_BASE_DIR, f"oos_{year}")
    output_root = os.path.join(PRUNE_OUTPUT_BASE_DIR, f"oos_{year}")
    os.makedirs(output_root, exist_ok=True)

    print("=" * 100)
    print(f"[INFO] OOS year       : {year}")
    print(f"[INFO] Filter root    : {filter_root}")
    print(f"[INFO] Prune output   : {output_root}")
    print(f"[INFO] N repeats      : {N}")
    print(f"[INFO] cvN            : {args.cvN}")
    print(f"[INFO] kmin-kmax      : {args.kmin}-{args.kmax}")
    print(f"[INFO] n_lambda0      : {len(LAMBDA0_GRID)}")
    print(f"[INFO] n_lambda2      : {len(LAMBDA2_GRID)}")

    for repeat_idx in range(1, N + 1):
        filter_repeat_dir = os.path.join(filter_root, f"repeat_{repeat_idx:02d}")
        output_repeat_dir = os.path.join(output_root, f"repeat_{repeat_idx:02d}")

        prune_one_repeat(
            year=year,
            repeat_idx=repeat_idx,
            filter_repeat_dir=filter_repeat_dir,
            output_repeat_dir=output_repeat_dir,
            cvN=args.cvN,
            kmin=args.kmin,
            kmax=args.kmax,
        )

    print("=" * 100)
    print("[INFO] All repeats prune finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)

    parser.add_argument("--cvN", type=int, default=3)
    parser.add_argument("--kmin", type=int, default=30)
    parser.add_argument("--kmax", type=int, default=55)

    args = parser.parse_args()
    main(args)
