# Multi-Pollutant Multi-Horizon Air Quality Classification

This project implements a **3-class classification** system for predicting air quality levels (specifically CO(GT)) across multiple prediction horizons. The task involves classifying future pollutant concentrations into three categories: **Low**, **Mid**, and **High** based on fixed thresholds.

## Overview

The classification model predicts CO(GT) concentration levels at 1h, 6h, 12h, and 24h ahead, using historical air quality data and engineered features. The target variable is discretized into three classes:
- **Class 0 (Low)**: CO(GT) < 1.5 mg/m³
- **Class 1 (Mid)**: 1.5 ≤ CO(GT) < 2.5 mg/m³
- **Class 2 (High)**: CO(GT) ≥ 2.5 mg/m³

## Dataset

### Data Splitting Strategy (Year-Based Temporal Split)

- **Train**: 2004 data (March - October) → 5,622 samples (60.2%)
- **Val**: 2004 data (November - December) → 1,464 samples (15.7%)
- **Test**: 2005 data (January - April) → 2,247 samples (24.1%)
- **Total**: 9,333 samples

### Features

- Input features are loaded from the prepared feature pack (`clean_data/airquality_prepared/`)
- Base feature set contains ~185 lagged or statistical descriptors (minimum lag 24h)
- During training the notebook can inject additional enhancements:
  - Interaction, difference, and cyclical encoding features
  - Horizon-specific signals (e.g., 24h adds 48/72/168h long lags and rolling means, 1h adds short-term trends)
  - Optional feature selection (Mutual Information by default, top 100 features)

## Models Implemented

The notebook evaluates multiple classification models:

1. **Persistence Baseline**: Naïve baseline that discretizes current CO(GT) values
2. **Logistic Regression**: Multi-class logistic regression with L2 regularization
3. **Ridge Classifier**: Ridge-based linear classifier
4. **Random Forest**: Ensemble tree-based classifier with class balancing
5. **XGBoost**: Gradient boosting classifier (optional, requires `xgboost` package)
6. **MLP (Multi-Layer Perceptron)**: Neural network classifier

### Model Configuration

- All models are evaluated with **macro-averaged F1 score**
- Class-imbalance handling:
  - Linear and tree models enable `class_weight='balanced'`
  - `rebalance_training_data()` can oversample each horizon’s training set (matching the largest class by default)
- Horizon-specific feature injection and feature selection can be toggled per experiment
- Hyperparameter tuning uses GridSearchCV or RandomizedSearchCV with PredefinedSplit folds

## Evaluation Metrics

The following metrics are computed for each model:

- **Accuracy**: Overall classification accuracy
- **Precision** (macro): Macro-averaged precision across all classes
- **Recall** (macro): Macro-averaged recall across all classes
- **F1 Score** (macro): Macro-averaged F1 score
- **ROC-AUC** (macro OvR): One-vs-Rest ROC-AUC score (for models with probability outputs)

## Results Summary

### Best Models by Horizon

Based on F1 score (macro):

- **1h horizon**: XGBoost (F1: 0.7294, Accuracy: 0.7244, ROC-AUC: 0.8922)
- **6h horizon**: XGBoost (F1: 0.5801, Accuracy: 0.5819, ROC-AUC: 0.7629)
- **12h horizon**: XGBoost (F1: 0.5333, Accuracy: 0.5351, ROC-AUC: 0.7346)
- **24h horizon**: Persistence Baseline (F1: 0.5619, Accuracy: 0.5668)

### Class Distribution & Handling

- Original training-set composition:
  - Class 0 (Low): ~29%
  - Class 1 (Mid): ~48%
  - Class 2 (High): ~23%
- Imbalance ratio ≈ 2.15 (max/min)
- When enabled, the notebook oversamples minority classes per horizon before fitting models to keep training labels balanced

## Folder Layout

```
classification-model/
├── HW3-classification.ipynb      # Main classification notebook
├── README.md                      # This file
└── outputs/
    ├── figures/
    │   ├── performance/           # Performance comparison charts
    │   │   ├── CO_model_comparison.png
    │   │   ├── metrics_comparison_by_model_horizon.png
    │   │   ├── best_models_across_horizons.png
    │   │   ├── roc_auc_heatmap.png
    │   │   ├── f1_score_all_models.png
    │   │   └── model_ranking_table.png
    │   ├── diagnostics/           # Diagnostic visualizations
    │   │   ├── roc_pr_curves_top4.png
    │   │   ├── confusion_matrices_top4.png
    │   │   ├── feature_importance_CO.png
    │   │   ├── class_distribution_analysis.png
    │   │   ├── imbalance_ratio_heatmap.png
    │   │   └── error_type_analysis.png
    │   └── pollutant_distributions.png
    └── results/
        └── classification_results_all_pollutants.csv
```

## Running the Notebook

### Prerequisites

1. Python environment with required packages:
   - `numpy`, `pandas`, `matplotlib`, `seaborn`
   - `scikit-learn`
   - `xgboost` (optional, for XGBoost classifier)
   - `joblib` (for parallel processing)

2. Data preparation:
   - Ensure `clean_data/airquality_prepared/` directory exists with:
     - `cleaned.parquet`: Cleaned air quality data
     - `splits.json`: Train/val/test split indices
     - `artifacts.json`: Feature engineering metadata
     - `loader.py`: Data loading module

### Execution Steps

1. Activate the project virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Open `classification-ML.ipynb` (or run the equivalent `.py`) inside Jupyter / VS Code.

3. Execute all cells in order. The workflow will:
   - Load the dataset and base feature matrices
   - Construct 3-class CO(GT) targets for 1/6/12/24-hour horizons
   - Optionally inject horizon-specific features, rebalance the training data, and apply feature selection
   - Train and evaluate every model
   - Produce diagnostic visualizations
   - Save summary metrics to `outputs/results/classification_results_all_pollutants.csv`

### Cell Structure (classification-ML.ipynb)

- **Cells 1–4**: environment setup, data loading, EDA, target construction
- **Cells 5–8**: metric helpers, enhanced features, missing-value remediation
- **Cell 9**: CO(GT) three-class target creation
- **Cells 10–11**: training utilities and persistence baseline
- **Cells 12–16**: linear / RF / XGB / MLP training loops (with rebalancing and feature selection)
- **Cell 17 onward**: result consolidation, ROC/PR plots, confusion matrices, feature importance, imbalance & error analysis, performance visualizations

## Key Features

### 1. Multi-Class Classification
- Implements 3-class classification (Low/Mid/High) using fixed thresholds
- Uses macro-averaged metrics to handle class imbalance
- Supports probability outputs for ROC-AUC calculation (One-vs-Rest)

### 2. Comprehensive Evaluation
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Per-class performance analysis
- Error type analysis (over-prediction, under-prediction, severe errors)
- Confidence analysis for correct vs. incorrect predictions

### 3. Visualizations
- ROC and Precision-Recall curves (One-vs-Rest for multi-class)
- Confusion matrices for top models
- Feature importance analysis (for tree-based models)
- Class distribution analysis
- Model performance comparison charts
- Error type visualizations

### 4. Model Comparison
- Systematic comparison across all models and horizons
- Best model selection based on F1 score
- Performance ranking tables
- Heatmaps for metric comparison

## Notes

- **Thresholds**: Fixed thresholds (1.5 and 2.5 mg/m³) are used for class discretization. These can be adjusted in Cell 4 if needed.
- **XGBoost**: Optional dependency. Install with `pip install xgboost` to enable XGBoost classifier.
- **Class Imbalance**: The dataset shows moderate imbalance (~2.15 ratio). Models use `class_weight='balanced'` to handle this.
- **Parallel Processing**: The notebook uses `joblib` for parallel hyperparameter tuning (up to 8 cores).
- **Random State**: Fixed random seed (42) is used for reproducibility.

## Output Files

### Results CSV
- `outputs/results/classification_results_all_pollutants.csv`: Comprehensive results table with all metrics for all models and horizons

### Figures
All visualizations are saved to `outputs/figures/`:
- Performance comparison charts
- Diagnostic plots (ROC/PR curves, confusion matrices)
- Feature importance plots
- Class distribution and imbalance analysis
- Error analysis visualizations

## Performance Insights

1. **Short-term predictions (1h)**: Best performance with XGBoost (F1: 0.7294)
2. **Medium-term predictions (6h, 12h)**: XGBoost also performs best, but performance degrades with longer horizons
3. **Long-term predictions (24h)**: Baseline persistence model performs competitively, suggesting limited predictive power at this horizon
4. **Class-specific performance**: High class (Class 2) generally has better F1 scores than Mid class (Class 1), indicating better separation for extreme values

## Future Improvements

- Experiment with different threshold values for class discretization
- Implement SMOTE or other oversampling techniques for better class balance
- Try ensemble methods combining multiple models
- Feature selection to reduce dimensionality
- Time-series specific models (LSTM, GRU) for sequential patterns
