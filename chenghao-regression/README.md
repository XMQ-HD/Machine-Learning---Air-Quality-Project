# Multi-Horizon Air Quality Regression  
**Targets:** CO(GT), C6H6(GT), NOx(GT), NO2(GT)  
**Horizons:** 1h, 6h, 12h, 24h

Author: **Chenghao Xie**  
Repo root: `chenghao-regression/`  
Main notebook: `notebooks/regression.ipynb`  
Python: **3.10** (CPU ok; optional TF/Keras for RNNs)

---

## 📁 Repository Layout

chenghao-regression/
├─ notebooks/
│  └─ regression.ipynb  # End-to-end: data → models → merge → plots
├─ outputs/
│  ├─ results/  # Per-family CSV results (baseline runs)
│  │  ├─ linear_kernel_results.csv
│  │  ├─ tree_results.csv
│  │  ├─ mlp_svr_results.csv
│  │  ├─ dl_results.csv            # RNN, seq_len=24
│  │  └─ dl_results_nn48.csv       # RNN, seq_len=48, standardized
│  ├─ figures/  # All figures used in the report
│  │  ├─ final_overview.png
│  │  ├─ perf_bar_by_horizon_1h.png
│  │  ├─ perf_bar_by_horizon_6h.png
│  │  ├─ perf_bar_by_horizon_12h.png
│  │  ├─ perf_bar_by_horizon_24h.png
│  │  ├─ perf_heatmaps_by_family.png
│  │  ├─ diagnostics_winners_1h_compact.png
│  │  ├─ diagnostics_winners_6h_compact.png
│  │  ├─ diagnostics_winners_12h_compact.png
│  │  └─ diagnostics_winners_24h_compact.png
│  ├─ overall_winners.csv
│  ├─ family_best_by_cell.csv
│  ├─ final_leaderboard.csv
│  ├─ avg_by_pollutant.csv
│  └─ avg_by_horizon.csv
└─ README.md  # This file
常见踩坑：

---

## 🎯 Objectives

1. Predict the next value \(y_{t+h}\) for four pollutants at horizons **1/6/12/24h**.  
2. Compare **model families** under a **fair baseline** (default hyper-parameters).  
3. Quantify **difficulty** by pollutant and by horizon (average across models).  
4. Provide compact **diagnostics** (residuals, QQ, rolling RMSE/MAE) and **one-page overviews** for reporting.

---

## 🧰 Data & Splits

- **Prepared pack** under `clean_data/airquality_prepared/` (aligned features/targets).
- **Targets:** pollutant value at \(t+h\), \(h \in \{1,6,12,24\}\).
- **Features:** lags, rolling stats, calendar/time features; standardized tensors for NN/RNN.
- **Temporal split:** train / val / test in chronological order. Validation for early-stopping; test for final reporting.

---

## 🧪 Models & Families

We report results **by model** and also **by family**. 

### 1) Linear / Kernel family
- **Ridge Regression** (L2-regularized linear)
- **Lasso / Elastic Net** (L1 / L1+L2; where applicable in baseline runs)
- **LinearSVR** (ε-insensitive linear SVR)
- **SVR (RBF kernel)** (nonlinear kernel baseline)
  - **Purpose:** strong 1h/24h baselines when linear trend + daily cycle are informative; acts as a contrast to Trees/RNNs for nonlinearity.
  - **Config:** default hyper-parameters (scikit-learn) with standard preprocessing; no wide grid search in the final scoreboard.

### 2) Tree family
- **RandomForestRegressor** (RF)
- **GradientBoostingRegressor** (GBDT)
  - **Default versions** are used in the **final leaderboard**.
  - **Light-tuning variants (ablation only):** small grids on `n_estimators`, `max_depth`, `min_samples_leaf`, `subsample`, evaluated with `TimeSeriesSplit(cv=3)`, **reported separately** to check sensitivity; **not counted** in the cross-family ranking.

### 3) Neural MLP / SVR block (tabular NN + a second SVR baseline)
- **MLPRegressor (small)**: 1–2 hidden layers, ReLU, early-stopping on validation.
- **(Duplicate listing) SVR (RBF)**: included in `mlp_svr_results.csv` to compare a shallow NN vs. a strong kernel baseline under the same tabular pack.
  - **Purpose:** simple capacity control on standardized features; ensures we’re not attributing gains only to Trees/RNNs.

### 4) Deep sequence models (RNNs)
- **LSTM (seq_len = 24)** on standardized tensors (`pack='nn'`)
- **GRU  (seq_len = 24)** on standardized tensors
- **LSTM (seq_len = 48)** on standardized tensors (`dl_results_nn48.csv`)
- **GRU  (seq_len = 48)** on standardized tensor
---

## ⚙️ Reproducibility

### Environment
- Python **3.10**
- `pip install -r requirements.txt` (or conda equivalents)
- Optional for RNNs:
  - `tensorflow==2.15.1`, `keras==2.15.0`  
  - Windows users may need long-path enabled

### How to run
1. Open `notebooks/regression.ipynb`.
2. Run cells in order:
   - Load & align data.
   - Train/evaluate **Linear/Kernel**, **Tree** (default), **MLP/SVR** (default).
   - Train/evaluate **RNNs** (optional): generates `dl_results.csv` (24) and `dl_results_nn48.csv` (48).
   - **Merge & plot**:
     - `final_overview.png` (winner heatmap + family bars for 1/6/12/24h)
     - per-horizon bars, family heatmaps,
     - **diagnostics** mosaics for 1h/6h/12h/24h,
     - **average difficulty** tables/figure.
3. Artifacts appear in `outputs/results/` and `outputs/figures/`.

---

## 📈 Key Figures

- **One-page overview:** `outputs/figures/final_overview.png`  
  (Winners heatmap, annotated with best model & \(R^2\); plus family bars for 1/6/12/24h)
- **Diagnostics (compact):**  
  `diagnostics_winners_{1h,6h,12h,24h}_compact.png`  
  (residual hist+KDE, QQ, rolling RMSE/MAE, residual vs. prediction)
- **Average difficulty:** `avg_difficulty_baseline.png`  
  (mean \(R^2\) aggregated by pollutant and by horizon)

---

## 🔎 Results Summary (Default Hyper-parameters)

### Overall behavior
- **1h is the easiest**: highest \(R^2\) across pollutants.
- **6–12h is the hardest**: marked drop in \(R^2\) (weak short-term autocorrelation; exogenous noise dominates).
- **24h rebounds**: daily cycle becomes a usable anchor; several pollutants recover vs. 6–12h.

### By family
- **Linear/Kernel** & **Tree**: strong and stable for 1h; Trees remain robust at longer horizons.  
- **RNNs (LSTM/GRU)**: noticeably helpful at 1h and sometimes 24h (when daily cycle is strong), but **not systematically superior** overall under the current data/feature setup.

### Concrete exemplars (seq_len=48, standardized)
- **CO(GT)** 1h — **GRU(nn, 48)**: \(R^2 \approx 0.616\)  
- **NO2(GT)** 1h — **LSTM/GRU(nn, 48)**: \(R^2 \approx 0.589\)  
- **C6H6(GT)** 1h — **LSTM(nn, 48)**: \(R^2 \approx 0.556\)  
- **NOx(GT)** 1h — **GRU(nn, 48)**: \(R^2 \approx 0.468\)

### “Average difficulty” (across models)
- **By pollutant:** **CO(GT)** easiest; **NOx(GT)** hardest; **NO2(GT)** / **C6H6(GT)** in between.  
- **By horizon:** **1h > 24h > 6h ≈ 12h** in your baseline aggregation (24h gains from daily cycle; mid-range horizons suffer most).

---

## 🧭 Error Diagnostics (What the mosaics show)

- **Heavy tails** in QQ-plots → spikes/extremes matter (weather/holiday/operational anomalies).  
- **Rolling errors** are higher around late night / early morning → add time-of-day, weekend/holiday features.  
- **Residual vs. prediction** shows mild under-prediction at high concentrations → try **log-targets**, **quantile regression**, or **stacked ensembles**.
---


## ✅ Recommendations

- **Features:** add **forecast covariates** (future weather), **Fourier day/week terms**, **holiday/event dummies**, **traffic proxies**; consider **log-targets** or **quantile loss**.  
- **Models:** try **XGBoost/LightGBM**, **TCN/Transformer** for multi-step direct forecasting; **stacking/blending** helps at 24h.  
- **Evaluation:** rolling-origin CV; stratified error slices by day/night, weekday/weekend, season.

---



## 📚 References

- AirQualityUCI dataset
- Scikit-learn: LinearSVR, RandomForest, GradientBoosting.  
- Keras/TensorFlow: LSTM/GRU sequence modeling.  


**License:** Coursework (educational use)  
**Contact:** Chenghao Xie (UNSW)