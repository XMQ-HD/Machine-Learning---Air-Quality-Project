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
- Feature set includes 185 features (lagged values, statistical aggregations, etc.)
- Minimum lag: 24 hours

## Models Implemented

The notebook evaluates multiple classification models:

1. **Persistence Baseline**: Naïve baseline that discretizes current CO(GT) values
2. **Logistic Regression**: Multi-class logistic regression with L2 regularization
3. **Ridge Classifier**: Ridge-based linear classifier
4. **Random Forest**: Ensemble tree-based classifier with class balancing
5. **XGBoost**: Gradient boosting classifier (optional, requires `xgboost` package)
6. **MLP (Multi-Layer Perceptron)**: Neural network classifier

### Model Configuration

- All models use **macro-averaged F1 score** for multi-class evaluation
- Class imbalance is handled via `class_weight='balanced'` for linear and tree models
- Hyperparameter tuning is performed using GridSearchCV or RandomizedSearchCV
- Validation strategy: PredefinedSplit (train/val split)

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

### Class Distribution

The training set shows moderate class imbalance:
- Class 0 (Low): ~29% of samples
- Class 1 (Mid): ~48% of samples
- Class 2 (High): ~23% of samples
- Imbalance ratio: ~2.15 (max/min class count)

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

2. Open `HW3-classification.ipynb` in Jupyter Notebook or JupyterLab

3. Run all cells sequentially. The notebook will:
   - Load and prepare the dataset
   - Construct 3-class classification targets for CO(GT) at each horizon
   - Train and evaluate all models
   - Generate performance visualizations
   - Save results to CSV and figures to output directories

### Cell Structure

- **Cell 1**: Environment setup and library imports
- **Cell 2**: Data loading and pollutant identification
- **Cell 3**: Exploratory data analysis (pollutant distributions)
- **Cell 4**: Classification target construction (3-class CO(GT))
- **Cell 5**: Multi-class classification metrics helper functions
- **Cell 6**: Baseline classification (Persistence)
- **Cell 7**: Linear classification models (Logistic Regression, Ridge)
- **Cell 8**: Random Forest classifier
- **Cell 9**: XGBoost classifier (optional)
- **Cell 10**: MLP classifier
- **Cell 11**: Results consolidation and saving
- **Cell 12**: ROC & PR curves visualization
- **Cell 13**: Model performance comparison charts
- **Cell 14**: Confusion matrices
- **Cell 15**: Feature importance analysis
- **Cell 16**: Class imbalance analysis
- **Cell 17**: Error analysis
- **Cell 18**: Model performance comparison visualizations

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
