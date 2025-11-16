import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_packs(in_csv: str, out_dir: str):
    out_dir = Path(out_dir)
    (out_dir / "features" / "trees").mkdir(parents=True, exist_ok=True)
    (out_dir / "features" / "nn").mkdir(parents=True, exist_ok=True)

    # Read
    df_raw = pd.read_csv(in_csv, sep=";", decimal=",", low_memory=False, encoding="utf-8")
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

    # Timestamp
    if {"Date","Time"}.issubset(df_raw.columns):
        dt = pd.to_datetime(
        df_raw["Date"] + " " + df_raw["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    if dt.isna().any():
        mask = dt.isna()
        dt2 = pd.to_datetime(
        (df_raw.loc[mask, "Date"] + " " + df_raw.loc[mask, "Time"]),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
        dayfirst=True,
    )
        dt.loc[mask] = dt2
        n_valid = int(dt.notna().sum())
        if n_valid < 100:
            raise RuntimeError(f"Only {n_valid} valid timestamps parsed. Check Date/Time format in CSV.")
        df_raw.insert(0, "timestamp", dt)
        df_raw = df_raw.drop(columns=["Date", "Time"])
    else:
        raise RuntimeError("Expected 'Date' and 'Time' columns")

    # Sentinel → NaN
    for c in df_raw.columns:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            df_raw[c] = df_raw[c].replace(-200, np.nan)

    # Base clean
    df = (df_raw.dropna(subset=["timestamp"])
                .sort_values("timestamp")
                .drop_duplicates(subset=["timestamp"]))

    num_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]

    # Impute + flags
    for c in num_cols:
        miss = df[c].isna()
        med = df[c].median()
        df[c] = df[c].fillna(med)
        df[f"{c}__was_missing"] = miss.astype("int8")

    # Winsorize (3×IQR)
    def cap_iqr(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return s.clip(q1 - 3*iqr, q3 + 3*iqr)
    for c in num_cols:
        df[c] = cap_iqr(df[c])

    # Hourly asfreq + gap flag + ffill
    df = df.set_index("timestamp").asfreq("H")
    first_num = num_cols[0]
    df["__gap"] = df[first_num].isna().astype("int8")
    for c in num_cols:
        df[c] = df[c].ffill()
    for c in df.columns:
        if str(c).endswith("__was_missing"):
            df[c] = df[c].fillna(0).astype("int8")
    df["__gap"] = df["__gap"].fillna(0).astype("int8")
    df["entity_id"] = "station_1"
    df = df.reset_index()

    # Save
    df.to_parquet(out_dir / "cleaned.parquet", index=False)

    # Targets
    co_candidates = [c for c in df.columns if "CO" in c.upper() and "(GT" in c]
    if not co_candidates:
        co_candidates = [c for c in df.columns if "CO" in c.upper()]
    co_col = co_candidates[0] if co_candidates else first_num
    df["target_reg_next_CO"] = df[co_col].shift(-1)

    # splits (pre-warmup)
    n = len(df)
    i_tr, i_va = int(0.7*n), int(0.85*n)
    splits = {"train": list(range(0, i_tr)),
              "val":   list(range(i_tr, i_va)),
              "test":  list(range(i_va, n))}

    # Tertile labels on train
    train_vals = df.loc[splits["train"], "target_reg_next_CO"].dropna()
    q1v, q2v = train_vals.quantile([0.33, 0.66]).values.tolist()
    def to_cls(v):
        if pd.isna(v): return np.nan
        if v <= q1v: return 0
        if v <= q2v: return 1
        return 2
    df["target_cls_next_CO3"] = df["target_reg_next_CO"].apply(to_cls)

    # Feature construction
    num_all = [c for c in df.columns if c not in ["timestamp","entity_id"] and pd.api.types.is_numeric_dtype(df[c])]
    keys = ["CO","PT08","NMHC","C6H6","NOX","NO2","T","RH","AH"]
    prios = []
    for k in keys:
        for c in num_all:
            if k.lower() in c.lower():
                prios.append(c)
    prios = sorted(set(prios))[:12] if prios else num_all[:12]

    lags  = [1,2,3,6,12,24]
    rolls = [3,6,12,24]

    X = df[["timestamp","entity_id"]].copy()
    for c in prios:
        X[c] = df[c]
        for L in lags:
            X[f"{c}__lag{L}"] = df[c].shift(L)
        for W in rolls:
            X[f"{c}__roll_mean{W}"] = df[c].rolling(W, min_periods=max(2,W//2)).mean()
            X[f"{c}__roll_std{W}"]  = df[c].rolling(W, min_periods=max(2,W//2)).std()

    ts = df["timestamp"]
    X["hour"] = ts.dt.hour
    X["dow"] = ts.dt.dayofweek
    X["month"] = ts.dt.month
    X["is_weekend"] = (X["dow"] >= 5).astype("int8")
    X["gap_flag"] = df["__gap"]

    min_lag = max(lags)
    X_feat = X.loc[min_lag:].reset_index(drop=True)
    y_reg  = df.loc[min_lag:, "target_reg_next_CO"].reset_index(drop=True)
    y_cls  = df.loc[min_lag:, "target_cls_next_CO3"].reset_index(drop=True)

    splits_trim = {k: [i - min_lag for i in idx if i >= min_lag and i < len(df)]
                   for k, idx in splits.items()}

    # Packs
    X_trees = X_feat.select_dtypes(include=[np.number])
    X_trees.to_parquet(out_dir/"features"/"trees"/"X.parquet", index=False)
    pd.DataFrame({"target_reg": y_reg, "target_cls": y_cls}).to_parquet(out_dir/"features"/"trees"/"y.parquet", index=False)

    scaler = StandardScaler()
    X_nn = pd.DataFrame(scaler.fit_transform(X_trees.values), columns=X_trees.columns)
    X_nn.to_parquet(out_dir/"features"/"nn"/"X.parquet", index=False)
    pd.DataFrame({"target_reg": y_reg, "target_cls": y_cls}).to_parquet(out_dir/"features"/"nn"/"y.parquet", index=False)

    # Metadata
    (out_dir/"splits.json").write_text(json.dumps(splits_trim))
    (out_dir/"artifacts.json").write_text(json.dumps({
        "co_column": co_col,
        "priority_features": prios,
        "lags": lags, "rolls": rolls,
        "rows_total": int(len(df)),
        "rows_after_warmup": int(len(X_feat)),
        "X_trees_shape": list(X_trees.shape),
        "X_nn_shape": list(X_nn.shape),
        "label_map": {"low":0,"mid":1,"high":2}
    }, indent=2))

    # Loader
    (out_dir/"loader.py").write_text(
        "import json\nfrom pathlib import Path\nimport pandas as pd\n\n"
        "def load_pack(base_dir: str, pack='trees', split='train'):\n"
        "    base = Path(base_dir)\n"
        "    X = pd.read_parquet(base/'features'/pack/'X.parquet')\n"
        "    y = pd.read_parquet(base/'features'/pack/'y.parquet')\n"
        "    splits = json.loads((base/'splits.json').read_text())\n"
        "    idx = splits[split]\n"
        "    return (X.iloc[idx].reset_index(drop=True),\n"
        "            y['target_reg'].iloc[idx].reset_index(drop=True),\n"
        "            y['target_cls'].iloc[idx].reset_index(drop=True))\n"
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="airquality_prepared")
    args = ap.parse_args()
    build_packs(args.csv, args.out)
    print("Done. Packs at:", args.out)
