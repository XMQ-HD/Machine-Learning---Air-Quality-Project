# COMP9417 — Air Quality Project
 
 This repository provides:
 - the raw dataset `data/AirQualityUCI.csv`,
 - the preparation script `prepare_airquality.py`,
 - a **pre-built** cleaned feature pack at `clean_data/airquality_prepared/` (for teammates who don’t want to run the prep script).
 
 The prepared pack and the script can co-exist: reruns of the script can write to another folder (e.g., `clean_data/airquality_prepared_v2`) without touching the existing pack.
 
 ---
 
 ## 1) Repository layout
 
.
├──  clean_data
│   └── airquality_prepared/ # ready-to-use feature pack
│   ├── artifacts.json # run metadata (seed, windows, lags, targets)
│   ├── cleaned.parquet # cleaned time-aligned table (ts + raw cols + flags)
│   ├── features/
│   │   ├── nn/ # scaled features for neural models
│   │   │   ├── X.parquet # shape (N, 185)
│   │   │   └── y.parquet # [y_reg, y_cls]
│   │   └── trees/ # raw-scale features for tree models
│   │   ├── X.parquet
│   │   └── y.parquet
│   ├── loader.py # load_pack(path, pack, split) helper
│   └── splits.json # indices for train/val/test
├──  data/
│   └── AirQualityUCI.csv # raw CSV
├──  prepare_airquality.py # make a new prepared pack from CSV
├──  requirements.txt
└──  README.md
 
 ### What’s inside `airquality_prepared`
 
 - **`cleaned.parquet`** — the cleaned, gap-aware table, one row per timestamp. Use this if you want to do your **own** feature engineering.
 - **`features/trees`** — lag/rolling features (no scaling) for RandomForest/GBM/XGBoost, with targets:
   - `X.parquet` shape `(N, 185)`, `y.parquet` shape `(N, 2)` = `[y_reg, y_cls]`.
 - **`features/nn`** — the **same feature set**, standardized by `StandardScaler` for neural models.
 - **`splits.json`** — index lists for `train`, `val`, `test`.
 - **`loader.py`** — convenience loader.
 
 ---
 
 ## 2) Quick use (no preparation run)
 
 If you just want to start modelling right away:
 
 ```python
 from pathlib import Path
 import sys
 # point Python to the pack's folder so we can import loader.py
 sys.path.append(str(Path("clean_data/airquality_prepared")))
 from loader import load_pack
 
 # tree models:
 X_tr, yreg_tr, ycls_tr = load_pack("clean_data/airquality_prepared", pack="trees", split="train")
 X_va, yreg_va, ycls_va = load_pack("clean_data/airquality_prepared", pack="trees", split="val")
 X_te, yreg_te, ycls_te = load_pack("clean_data/airquality_prepared", pack="trees", split="test")
 
 # neural models (scaled features):
 X_tr_nn, yreg_tr, ycls_tr = load_pack("clean_data/airquality_prepared", pack="nn", split="train")
 ```
 
 Targets:
 - `y_reg` — CO_GT next-hour **regression** target (continuous ppm).
 - `y_cls` — CO_GT status **classification** target (0/1 thresholded).
 
 ---
 
 ## 3) Build a new prepared pack yourself
 
 ### 3.1. Prerequisites
 
 - Python **3.8–3.12** recommended.
 - macOS / Linux / Windows supported.
 - Optional but recommended: **Git LFS** if you plan to version large CSV/parquet files.
 
 Install Python packages:
 
 ```bash
 # macOS / Linux
 python3 -m venv .venv
 source .venv/bin/activate
 python -m pip install -U pip
 pip install -r requirements.txt
 ```
 
 ```powershell
 # Windows PowerShell
 py -3 -m venv .venv
 .\.venv\Scripts\Activate.ps1
 python -m pip install -U pip
 pip install -r requirements.txt
 ```
 
 > If `py` is not available, use `python` in place of `py`.
 
 ### 3.2. Run the preparation script
 
 ```bash
 # write to a NEW folder so the shipped pack remains intact
 python prepare_airquality.py --csv data/AirQualityUCI.csv --out clean_data/airquality_prepared_v2
 ```
 
 - The script parses the semicolon-delimited CSV with comma decimals, cleans gaps, creates lag/rolling features, standardizes for NN, and writes `artifacts.json`, `cleaned.parquet`, `features/`, `splits.json`, and `loader.py` under the output folder you pass to `--out`.
 - Re-run with different `--out` names to create multiple versions side-by-side.
 
 ### 3.3. Use your newly built pack
 
 ```python
 import sys
 sys.path.append("clean_data/airquality_prepared_v2")
 from loader import load_pack
 
 X_tr, yreg_tr, ycls_tr = load_pack("clean_data/airquality_prepared_v2", pack="trees", split="train")
 ```
 
 ---
 
 ## 4) Reproducibility & notes
 
 - `artifacts.json` records window sizes, lags, seeds and column lists used to build the pack.
 - `splits.json` holds row indices for each split; we use **index-based** slicing to avoid leakage.
 - `cleaned.parquet` preserves a `ts`/time column plus original sensor fields and flags (e.g., `__gap`).
 
 ---
 
 ## 5) Optional: Git LFS for large files
 
 ```bash
 # once per machine
 git lfs install
 git lfs track "*.csv" "*.parquet"
 git add .gitattributes
 git commit -m "Track large data files with LFS"
 ```
 
 ---
 
 ## 6) Report (Overleaf)
 
 The shared Overleaf project link will be placed here once created: **TBD**.
 
 - P1 (Data engineering): cleaning, gap handling, lag/rolling features, scaling strategy, split design.
 - P2 (Feature/time-series): rationale for windows/lags, leakage prevention, ablation notes.
 
 ---
 
 ## 7) Common issues
 
 - **Windows execution policy** blocks venv activation: run PowerShell as Admin → `Set-ExecutionPolicy RemoteSigned` → `Y`.
 - **Locale parsing** warnings on dates: the script explicitly sets `dayfirst=True` and decimal=`,`; these warnings are safe.
 - **Slow DataFrame ops** warnings: informational only; the final artifacts are correct.
 
 ---
 
# Maintainers: P1/P2 owner — please open an issue for any dataset quirks or to request additional features.





