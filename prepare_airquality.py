#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ----- Configuration that makes the pipeline PDF/spec compliant -----
CO_THRESHOLDS = (1.5, 2.5)  # mg/m^3 -> (low, mid) fixed cut points
VAL_FRACTION_2004 = 0.20    # last 20% of 2004 becomes validation


def _parse_timestamp(df_raw: pd.DataFrame) -> pd.Series:
    """Robust parse for UCI AirQuality 'Date' + 'Time' with European formats."""
    if {"Date", "Time"}.issubset(df_raw.columns):
        # Primary format in the CSV (e.g., 10/03/2004 18.00.00)
        dt = pd.to_datetime(
            df_raw["Date"] + " " + df_raw["Time"],
            format="%d/%m/%Y %H.%M.%S",
            errors="coerce",
        )
        if dt.isna().any():
            mask = dt.isna()
            # Fallback if some rows use colon time separators
            dt2 = pd.to_datetime(
                (df_raw.loc[mask, "Date"] + " " + df_raw.loc[mask, "Time"]),
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce",
                dayfirst=True,
            )
            dt.loc[mask] = dt2
        if int(dt.notna().sum()) < 100:
            raise RuntimeError(
                "Too few valid timestamps parsed. Check Date/Time format in CSV."
            )
        return dt
    raise RuntimeError("Expected 'Date' and 'Time' columns in input CSV.")


def build_packs(in_csv: str, out_dir: str):
    out_dir = Path(out_dir)
    (out_dir / "features" / "trees").mkdir(parents=True, exist_ok=True)
    (out_dir / "features" / "nn").mkdir(parents=True, exist_ok=True)

    # -------- Read raw CSV (European separators; avoid dtype guessing) --------
    df_raw = pd.read_csv(
        in_csv, sep=";", decimal=",", low_memory=False, encoding="utf-8"
    )
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains(r"^Unnamed")]

    # -------- Timestamp --------
    dt = _parse_timestamp(df_raw)
    df_raw.insert(0, "timestamp", dt)
    df_raw = df_raw.drop(columns=["Date", "Time"])

    # -------- Sentinel -> NaN; basic clean --------
    for c in df_raw.columns:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            df_raw[c] = df_raw[c].replace(-200, np.nan)

    df = (
        df_raw.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    num_cols = [
        c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])
    ]

    # -------- Impute + missing flags --------
    miss_flag_frames = []
    for c in num_cols:
        miss = df[c].isna()
        med = df[c].median()
        df[c] = df[c].fillna(med)
        miss_flag_frames.append(miss.astype("int8").rename(f"{c}__was_missing"))

    # -------- Winsorize numeric (3×IQR) --------
    def cap_iqr(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return s.clip(q1 - 3 * iqr, q3 + 3 * iqr)

    for c in num_cols:
        df[c] = cap_iqr(df[c])

    # -------- Hourly asfreq + gap flag + ffill --------
    df = df.set_index("timestamp").asfreq("H")
    first_num = num_cols[0]
    df["__gap"] = df[first_num].isna().astype("int8")
    for c in num_cols:
        df[c] = df[c].ffill()
    # fill flag NaNs (if any)
    for f in miss_flag_frames:
        name = f.name
        if name not in df.columns:
            df[name] = f.values
        df[name] = df[name].fillna(0).astype("int8")
    df["__gap"] = df["__gap"].fillna(0).astype("int8")
    df["entity_id"] = "station_1"
    df = df.reset_index()

    # -------- Save cleaned table --------
    df.to_parquet(out_dir / "cleaned.parquet", index=False)

    # -------- Target columns --------
    # Pick a CO column
    co_candidates = [c for c in df.columns if "CO" in c.upper() and "(GT" in c]
    if not co_candidates:
        co_candidates = [c for c in df.columns if "CO" in c.upper()]
    co_col = co_candidates[0] if co_candidates else first_num

    # Next-hour regression target
    df["target_reg_next_CO"] = df[co_col].shift(-1)

    # Fixed-threshold classification (PDF/spec requirement)
    low, mid = CO_THRESHOLDS

    def to_cls(v):
        if pd.isna(v):
            return np.nan
        if v <= low:
            return 0
        if v <= mid:
            return 1
        return 2

    df["target_cls"] = df["target_reg_next_CO"].apply(to_cls)

    # -------- Year-based splits --------
    years = df["timestamp"].dt.year
    mask_2004 = years == 2004
    mask_2005 = years == 2005

    idx_2004 = df.index[mask_2004].to_list()
    idx_2005 = df.index[mask_2005].to_list()

    # Chronological split inside 2004 for train/val
    n_2004 = len(idx_2004)
    cut = int((1.0 - VAL_FRACTION_2004) * n_2004)
    train_idx = idx_2004[:cut]
    val_idx = idx_2004[cut:]
    test_idx = idx_2005  # all of 2005

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # -------- Feature construction (concat once to avoid fragmentation) --------
    keys = ["CO", "PT08", "NMHC", "C6H6", "NOX", "NO2", "T", "RH", "AH"]
    num_all = [
        c
        for c in df.columns
        if c not in ["timestamp", "entity_id"] and pd.api.types.is_numeric_dtype(df[c])
    ]
    prios = []
    for k in keys:
        for c in num_all:
            if k.lower() in c.lower():
                prios.append(c)
    prios = sorted(set(prios))[:12] if prios else num_all[:12]

    lags = [1, 2, 3, 6, 12, 24]
    rolls = [3, 6, 12, 24]

    frames = []
    frames.append(df[["entity_id"]].reset_index(drop=True))  # keep entity id separately

    # base numeric (selected)
    frames.append(df[prios].reset_index(drop=True))

    # lags and rollings
    for c in prios:
        for L in lags:
            frames.append(df[c].shift(L).rename(f"{c}__lag{L}"))
        for W in rolls:
            frames.append(
                df[c]
                .rolling(W, min_periods=max(2, W // 2))
                .mean()
                .rename(f"{c}__roll_mean{W}")
            )
            frames.append(
                df[c]
                .rolling(W, min_periods=max(2, W // 2))
                .std()
                .rename(f"{c}__roll_std{W}")
            )

    # calendar + gap flag
    ts = df["timestamp"]
    cal = pd.DataFrame(
        {
            "hour": ts.dt.hour,
            "dow": ts.dt.dayofweek,
            "month": ts.dt.month,
            "is_weekend": (ts.dt.dayofweek >= 5).astype("int8"),
            "gap_flag": df["__gap"].astype("int8"),
        }
    ).reset_index(drop=True)
    frames.append(cal)

    # final X before warmup trim
    X_all = pd.concat(frames, axis=1)

    # trim for max lag
    min_lag = max(lags)
    X_feat = X_all.loc[min_lag:].reset_index(drop=True)
    y_reg = df.loc[min_lag:, "target_reg_next_CO"].reset_index(drop=True)
    y_cls = df.loc[min_lag:, "target_cls"].reset_index(drop=True)

    # shift splits to align with warmup trim
    splits_trim = {
        k: [i - min_lag for i in idx if min_lag <= i < len(df)] for k, idx in splits.items()
    }

    # -------- Packs: trees (raw scale) --------
    X_trees = X_feat.select_dtypes(include=[np.number])
    (out_dir / "features" / "trees" / "X.parquet").write_bytes(
        X_trees.to_parquet(index=False)
        if hasattr(pd.DataFrame, "to_parquet")
        else b""
    )
    pd.DataFrame({"target_reg": y_reg, "target_cls": y_cls}).to_parquet(
        out_dir / "features" / "trees" / "y.parquet",
        index=False,
    )

    # -------- Packs: nn (standardised) --------
    scaler = StandardScaler()
    X_nn = pd.DataFrame(scaler.fit_transform(X_trees.values), columns=X_trees.columns)
    X_nn.to_parquet(out_dir / "features" / "nn" / "X.parquet", index=False)
    pd.DataFrame({"target_reg": y_reg, "target_cls": y_cls}).to_parquet(
        out_dir / "features" / "nn" / "y.parquet",
        index=False,
    )

    # -------- Metadata + loader --------
    (out_dir / "splits.json").write_text(json.dumps(splits_trim))
    (out_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "co_column": co_col,
                "priority_features": prios,
                "lags": lags,
                "rolls": rolls,
                "rows_total": int(len(df)),
                "rows_after_warmup": int(len(X_feat)),
                "X_trees_shape": list(X_trees.shape),
                "X_nn_shape": list(X_nn.shape),
                "label_map": {"low": 0, "mid": 1, "high": 2},
                "cls_thresholds": {"low": CO_THRESHOLDS[0], "mid": CO_THRESHOLDS[1]},
                "splits_rule": "2004 train/val (last 20% val), 2005 test",
            },
            indent=2,
        )
    )

    (out_dir / "loader.py").write_text(
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
    ap.add_argument("--csv", required=True, help="Path to AirQualityUCI.csv")
    ap.add_argument("--out", default="airquality_prepared", help="Output folder")
    args = ap.parse_args()
    build_packs(args.csv, args.out)
    print("Done. Packs at:", args.out)
