import numpy as np
import pandas as pd
import os

df = pd.read_feather("../../../data/common/chars_raw_imputed.feather")

# size
df["size"] = abs(df["prc"] * df["shrout"])

# read char names
chars = pd.read_csv("../../../data/1_data_processing/1_rank_data/input/chars_summary.csv")
acronyms = chars["Acronym"].astype(str).str.rstrip().tolist()

# rank features
rank_features = ["size"] + [c for c in acronyms if c != "size"]

# check missing columns
missing_cols = [c for c in rank_features if c not in df.columns]
if len(missing_cols) > 0:
    raise ValueError(f"Missing columns in raw data: {missing_cols}")

# 0 to nan
# df = df.replace(0, np.nan)

# keep needed raw columns
keep_raw_cols = ["permno", "date", "ret", "size"] + [c for c in rank_features if c != "size"]
df = df[keep_raw_cols].copy()

# sort
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "permno"]).reset_index(drop=True)

# rank to [0, 1]
def cross_sectional_rank_0_1(x: pd.Series) -> pd.Series:
    notna = x.notna()
    n = notna.sum()

    out = pd.Series(np.nan, index=x.index, dtype=float)

    if n == 0:
        return out
    if n == 1:
        out.loc[notna] = 0.5
        return out

    ranks = x.loc[notna].rank(method="average")
    out.loc[notna] = (ranks - 1) / (n - 1)
    return out

# make rank columns
for col in rank_features:
    rank_col = f"{col}_rank"
    df[rank_col] = df.groupby("date", group_keys=False)[col].apply(cross_sectional_rank_0_1)

# keep final columns
rank_cols = [f"{c}_rank" for c in rank_features]
final_cols = ["permno", "date", "ret", "size"] + rank_cols
df = df[final_cols].copy()

# save full pickle
save_path = "../../../data/1_data_processing/1_rank_data/output/ranked_data.pkl.gz"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df.to_pickle(save_path)

# save by year parquet
save_dir = "../../../data/1_data_processing/1_rank_data/output/by_year"
os.makedirs(save_dir, exist_ok=True)

years = sorted(df["date"].dt.year.unique())

for year in years:
    df_year = df[df["date"].dt.year == year].copy()
    year_path = os.path.join(save_dir, f"{year}.parquet")
    df_year.to_parquet(year_path, index=False)
    print(f"Saved {year}: {df_year.shape}")

print("Saved successfully")
print(f"Final shape: {df.shape}")
print(f"Number of rank features: {len(rank_features)}")