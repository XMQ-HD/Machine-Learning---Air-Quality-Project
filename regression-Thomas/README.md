# Multi-Pollutant Multi-Time-Horizon Air Quality Regression Prediction

This project implements machine learning models for predicting air quality pollutant concentrations across multiple time horizons. The research evaluates 8 different models on 4 pollutants (CO, C6H6, NOx, NO2) for 4 prediction horizons (1h, 6h, 12h, 24h).

## 📁 Directory Structure

```
regression-Thomas/
├── HW3-regression.ipynb          # Main experiment notebook (English)
├── HW3-regression-zh.ipynb       # Experiment notebook (Chinese)
├── README.md                      # This file
├── Regression_Analysis_Report.md  # Complete analysis report (English)
├── 回归分析报告.md                 # Complete analysis report (Chinese)
└── outputs/
    ├── results/
    │   └── regression_results_all_pollutants.csv  # All model performance metrics
    ├── figures/
    │   ├── eda/                   # Exploratory data analysis plots
    │   ├── performance/           # Model performance comparison plots
    │   ├── diagnostics/           # Model diagnostic plots
    │   └── analysis/              # In-depth analysis plots
    └── models/                    # Trained model files (if saved)
```

## 🎯 Project Overview

### Research Objectives
- Build multiple machine learning models to predict pollutant concentrations at different time horizons
- Compare model performance across different pollutants and time windows
- Analyze prediction difficulty for different pollutants and time horizons
- Identify key factors affecting prediction performance

### Target Pollutants
1. **CO(GT)** - Carbon Monoxide
2. **C6H6(GT)** - Benzene
3. **NOx(GT)** - Nitrogen Oxides
4. **NO2(GT)** - Nitrogen Dioxide

### Time Horizons
- **1 hour**: Short-term prediction
- **6 hours**: Medium-term prediction
- **12 hours**: Medium-to-long-term prediction
- **24 hours**: Long-term prediction

## 🔬 Models Evaluated

The project evaluates 8 different models:

1. **Naive Baseline** - Persistence model
2. **Linear Regression** - Standard linear regression
3. **Ridge** - L2 regularized linear regression
4. **Lasso** - L1 regularized linear regression
5. **ElasticNet** - L1 and L2 mixed regularized linear regression
6. **Random Forest** - Random forest regressor
7. **XGBoost** - Gradient boosting decision tree
8. **MLP** - Multi-layer perceptron (neural network)

## 📊 Dataset

- **Data Source**: AirQualityUCI dataset
- **Training set**: 6,525 samples
- **Validation set**: 1,404 samples
- **Test set**: 1,404 samples
- **Features**: 185 features (lag features and rolling window features)
- **Time Lag**: Maximum lag of 24 hours

## 🚀 Getting Started

### Prerequisites

- Python 3.8-3.12
- Jupyter Notebook or JupyterLab
- Required packages (see `requirements.txt` in parent directory)

### Installation

1. Navigate to the parent directory:
```bash
cd ..
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiments

1. Open the notebook:
   - English version: `HW3-regression.ipynb`
   - Chinese version: `HW3-regression-zh.ipynb`

2. Run all cells to reproduce the experiments:
   - The notebook will automatically create the `outputs/` directory structure
   - All results and figures will be saved in `outputs/`

3. Expected outputs:
   - Model performance metrics in `outputs/results/regression_results_all_pollutants.csv`
   - Visualization figures in `outputs/figures/`
   - Trained models (if saved) in `outputs/models/`

### Key Configuration

The notebook uses the following configuration:
- **Random seed**: 42 (for reproducibility)
- **Parallel computing**: Uses all available CPU cores
- **Hyperparameter tuning**: 
  - GridSearchCV for linear and tree models
  - RandomizedSearchCV for neural networks

## 📈 Results and Reports

### Analysis Reports

Two comprehensive analysis reports are available:

1. **English Report**: `Regression_Analysis_Report.md`
   - Complete analysis in English
   - Includes all visualizations
   - Model performance comparisons
   - Key findings and recommendations

2. **Chinese Report**: `回归分析报告.md`
   - Complete analysis in Chinese
   - Same content as English version
   - Includes all visualizations

### Results File

- **`outputs/results/regression_results_all_pollutants.csv`**
  - Contains performance metrics (RMSE, MAE, MAPE, R²) for all models
  - Covers all pollutant and time horizon combinations
  - Includes training time for each model

### Visualization Figures

All figures are organized in `outputs/figures/`:

- **EDA** (`figures/eda/`): Exploratory data analysis plots
- **Performance** (`figures/performance/`): Model performance comparisons
- **Diagnostics** (`figures/diagnostics/`): Model diagnostic plots
- **Analysis** (`figures/analysis/`): In-depth analysis visualizations

## 🔍 Key Findings

### Model Performance
- **Best overall models**: Ridge, Lasso, and ElasticNet show excellent performance
- **Short-term prediction (1h)**: Lasso performs best on CO and C6H6, Ridge performs best on NOx and NO2
- **Medium-to-long-term prediction (6-24h)**: XGBoost performs best on CO, Lasso/ElasticNet perform best on C6H6, Ridge performs best on NOx and NO2
- **Ridge model**: Achieves best performance on all time horizons for NOx and NO2

### Pollutant Difficulty
- **Easiest to predict**: CO(GT) - relatively stable concentration
- **Most difficult**: NOx(GT) - large variation range

### Time Horizon Difficulty
- **Easiest**: 1-hour horizon (R² > 0.7)
- **Most difficult**: 6-hour horizon
- **Interesting finding**: 24-hour prediction performs better than 6-12 hour predictions (possibly due to daily cycle patterns)

## 📝 Evaluation Metrics

The project uses 4 evaluation metrics:

1. **RMSE** (Root Mean Squared Error)
2. **MAE** (Mean Absolute Error)
3. **MAPE** (Mean Absolute Percentage Error)
4. **R²** (R-squared / Coefficient of Determination)

## 🛠️ Technical Details

- **CPU cores**: 208 cores (utilized for parallel training)
- **Training time**:
  - Linear models: 0.05-4 seconds/task
  - Tree models: 5-200 seconds/task
  - Neural networks: 5-37 seconds/task

## 📚 References

- AirQualityUCI dataset
- Scikit-learn documentation
- XGBoost documentation
- Related time series prediction research literature

## 📄 License

This project is part of COMP9417 coursework.

## 👤 Author

Thomas

---

**Note**: Make sure to run the notebook from the `regression-Thomas/` directory, as the code uses relative paths to access data from the parent directory (`../clean_data/airquality_prepared/`).

