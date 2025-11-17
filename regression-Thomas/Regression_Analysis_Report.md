# Multi-Pollutant Multi-Time-Horizon Air Quality Regression Prediction Analysis Report

## 1. Project Overview

This project aims to use machine learning methods for multi-pollutant, multi-time-horizon regression prediction of air quality data. The research covers concentration prediction tasks for 4 major pollutants (CO, C6H6, NOx, NO2) at 1-hour, 6-hour, 12-hour, and 24-hour future time horizons.

### 1.1 Research Objectives

- Build multiple machine learning models to predict concentrations of different pollutants at different time horizons
- Compare the performance of different models on various pollutant prediction tasks
- Analyze the prediction difficulty of different pollutants and time windows
- Identify key factors affecting prediction performance

### 1.2 Dataset

- **Data Source**: AirQualityUCI dataset
- **Data Scale**: 
  - Training set: 6,525 samples
  - Validation set: 1,404 samples
  - Test set: 1,404 samples
- **Feature Dimensions**: 185 features (including lag features and rolling window features)
- **Time Lag**: Maximum lag of 24 hours

## 2. Data Description

### 2.1 Target Pollutants

The research covers the following 4 pollutants:

1. **CO(GT)** - Carbon Monoxide
   - Mean: 2.08 ppm
   - Std: 1.28 ppm
   - Range: 0.1 - 6.8 ppm

2. **C6H6(GT)** - Benzene
   - Mean: 9.99 μg/m³
   - Std: 7.23 μg/m³
   - Range: 0.1 - 40.6 μg/m³

3. **NOx(GT)** - Nitrogen Oxides
   - Mean: 231.03 μg/m³
   - Std: 179.29 μg/m³
   - Range: 2.0 - 800.0 μg/m³

4. **NO2(GT)** - Nitrogen Dioxide
   - Mean: 112.31 μg/m³
   - Std: 43.69 μg/m³
   - Range: 2.0 - 274.0 μg/m³

![Pollutant Distribution](./figures/eda/pollutants_distribution.png)
*Figure 1: Distribution histograms of pollutants, showing basic statistical properties of the data*

### 2.2 Time Horizons

The prediction tasks cover 4 time horizons:
- **1 hour**: Short-term prediction
- **6 hours**: Medium-term prediction
- **12 hours**: Medium-to-long-term prediction
- **24 hours**: Long-term prediction

### 2.3 Feature Engineering

- Use lag features to capture temporal dependencies in time series
- Use rolling window features to capture local trends
- For neural network models, features are standardized using StandardScaler
- For tree models, use original scale features

![Feature Importance Analysis](./figures/analysis/feature_importance_multi_pollutant.png)
*Figure 2: Multi-pollutant feature importance analysis, showing the contribution of different features to prediction*

## 3. Methodology

### 3.1 Model Architecture

This study evaluates the following 8 models:

#### 3.1.1 Baseline Model
- **Naive Baseline**: Persistence model that uses the current time value to predict future time values

#### 3.1.2 Linear Models
- **Linear Regression**: Standard linear regression
- **Ridge**: L2 regularized linear regression
- **Lasso**: L1 regularized linear regression
- **ElasticNet**: L1 and L2 mixed regularized linear regression

#### 3.1.3 Tree Models
- **Random Forest**: Random forest regressor
- **XGBoost**: Gradient boosting decision tree

#### 3.1.4 Neural Network Model
- **MLP**: Multi-layer perceptron (using standardized features)

### 3.2 Hyperparameter Tuning

- **Linear models, tree models**: Use GridSearchCV for grid search
- **Neural network models**: Use RandomizedSearchCV for random search (due to longer training time)
- **Cross-validation**: Use predefined train/validation set splits
- **Parallel computing**: Utilize 208 CPU cores for parallel training

### 3.3 Evaluation Metrics

The following 4 metrics are used to evaluate model performance:

1. **RMSE** (Root Mean Squared Error): Root mean squared error
2. **MAE** (Mean Absolute Error): Mean absolute error
3. **MAPE** (Mean Absolute Percentage Error): Mean absolute percentage error
4. **R²** (R-squared): Coefficient of determination

## 4. Experimental Results

### 4.1 Best Models for Each Pollutant

#### CO(GT) - Carbon Monoxide
- **1 hour**: Lasso (RMSE: 0.6726, R²: 0.7409)
- **6 hours**: XGBoost (RMSE: 1.0398, R²: 0.3820)
- **12 hours**: XGBoost (RMSE: 1.0418, R²: 0.3807)
- **24 hours**: XGBoost (RMSE: 1.0376, R²: 0.3872)

#### C6H6(GT) - Benzene
- **1 hour**: Lasso (RMSE: 3.1866, R²: 0.7501)
- **6 hours**: Lasso (RMSE: 5.1600, R²: 0.3462)
- **12 hours**: Lasso (RMSE: 5.0922, R²: 0.3642)
- **24 hours**: ElasticNet (RMSE: 5.0977, R²: 0.3655)

#### NOx(GT) - Nitrogen Oxides
- **1 hour**: Ridge (RMSE: 91.0309, R²: 0.7825)
- **6 hours**: Ridge (RMSE: 157.1467, R²: 0.3529)
- **12 hours**: Ridge (RMSE: 166.1115, R²: 0.2780)
- **24 hours**: Ridge (RMSE: 165.2372, R²: 0.2860)

#### NO2(GT) - Nitrogen Dioxide
- **1 hour**: Ridge (RMSE: 23.7038, R²: 0.7745)
- **6 hours**: Ridge (RMSE: 38.4572, R²: 0.4074)
- **12 hours**: Ridge (RMSE: 39.4323, R²: 0.3773)
- **24 hours**: Ridge (RMSE: 39.3575, R²: 0.3824)

![Best Model Performance Heatmap](./figures/performance/performance_heatmap_best_models.png)
*Figure 3: Best model performance heatmap for each pollutant at different time horizons*

### 4.2 Average Model Performance (Across All Pollutants and Time Horizons)

| Model | Average RMSE | Average MAE | Average R² |
|-------|--------------|-------------|------------|
| **Ridge** | 40.93 | 30.84 | 0.506 |
| **Lasso** | 41.49 | 31.01 | 0.495 |
| **ElasticNet** | 41.58 | 31.11 | 0.496 |
| **XGBoost** | 41.71 | 30.45 | 0.490 |
| **Random Forest** | 43.13 | 31.82 | 0.421 |
| **Linear Regression** | 44.34 | 32.10 | 0.405 |
| **MLP** | 53.17 | 37.99 | 0.124 |
| **Naive Baseline** | 60.10 | 45.66 | -0.128 |

![Model Performance Heatmap](./figures/performance/performance_heatmap.png)
*Figure 4: Performance heatmap of all models on all tasks (RMSE), darker colors indicate smaller errors*

### 4.3 Pollutant Prediction Difficulty Analysis

| Pollutant | Average RMSE | Average MAE | Average R² |
|-----------|--------------|-------------|------------|
| **CO(GT)** | 0.99 | 0.76 | 0.371 |
| **C6H6(GT)** | 4.87 | 3.70 | 0.266 |
| **NO2(GT)** | 36.19 | 27.60 | 0.463 |
| **NOx(GT)** | 138.85 | 103.19 | 0.390 |

**Analysis**:
- **Easiest to predict**: CO(GT) - Carbon monoxide concentration is relatively stable with small variation range
- **Most difficult to predict**: NOx(GT) - Nitrogen oxides concentration has large variation range and strong volatility

![Multi-Pollutant Performance Comparison](./figures/performance/multi_pollutant_comparison.png)
*Figure 5: Performance comparison of different pollutants, showing prediction difficulty differences among pollutants*

### 4.4 Time Horizon Prediction Difficulty Analysis

| Time Horizon | Average RMSE | Average MAE | Average R² |
|--------------|--------------|-------------|------------|
| **1 hour** | 25.89 | 19.20 | 0.710 |
| **6 hours** | 48.42 | 37.15 | 0.270 |
| **12 hours** | 47.20 | 35.83 | 0.304 |
| **24 hours** | 45.12 | 34.30 | 0.333 |

**Analysis**:
- **Easiest to predict**: 1-hour horizon - Short-term prediction with strongest temporal dependency
- **Most difficult to predict**: 6-hour horizon - Medium-term prediction has the greatest uncertainty
- **Interesting finding**: 24-hour horizon prediction performance is actually better than 6-hour and 12-hour predictions, which may be related to daily cycle patterns

![Pollutant Comparison by Time Horizon](./figures/performance/pollutant_comparison_by_horizon.png)
*Figure 6: Pollutant performance comparison grouped by time horizon, showing prediction difficulty differences across time horizons*

## 5. Key Findings

### 5.1 Model Performance Insights

1. **Regularized linear models perform excellently**: Ridge, Lasso, and ElasticNet show the best overall performance, indicating that there is some collinearity among features, and regularization helps improve generalization ability.

2. **Lasso excels in short-term prediction**: In 1-hour prediction tasks, Lasso achieves the best performance on CO and C6H6 pollutants, demonstrating that L1 regularization can effectively handle feature selection and improve short-term prediction accuracy.

3. **XGBoost performs stably in medium-to-long-term prediction**: XGBoost achieves the best results on CO 6-24 hour prediction tasks consecutively, demonstrating that gradient boosting methods can effectively capture medium-to-long-term temporal dependency patterns.

4. **MLP performs poorly**: The neural network model performs worst on all tasks, possible reasons include:
   - Relatively small dataset, insufficient to train complex neural networks
   - Feature engineering may be more suitable for tree models
   - Hyperparameter search space may not be sufficient

![Error Comparison Analysis](./figures/analysis/error_comparison.png)
*Figure 7: Error comparison analysis of different models, showing performance differences across tasks*

### 5.2 Pollutant Characteristics Analysis

1. **CO(GT)**: Easiest to predict, small concentration range, relatively stable changes
2. **C6H6(GT)**: Medium prediction difficulty, but performs well on 1-hour prediction
3. **NO2(GT)**: Medium prediction difficulty, Ridge model performs stably
4. **NOx(GT)**: Most difficult to predict, large concentration variation range, requires stronger model capability

### 5.3 Time Horizon Characteristics Analysis

1. **Short-term prediction (1 hour)**: All models perform best, R² values generally above 0.7
2. **Medium-term prediction (6-12 hours)**: Prediction difficulty significantly increases, R² values drop to 0.3-0.4
3. **Long-term prediction (24 hours)**: Although the time span is longer, prediction performance is actually better than medium-term prediction, possibly due to daily cycle patterns

### 5.4 Model Selection Recommendations

Based on experimental results, the following recommendations are made:

- **Short-term prediction (1 hour)**: Prefer Lasso (for CO, C6H6) or Ridge (for NOx, NO2)
- **Medium-term prediction (6-12 hours)**: Prefer XGBoost (for CO) or Lasso (for C6H6) or Ridge (for NOx, NO2)
- **Long-term prediction (24 hours)**: Prefer XGBoost (for CO) or ElasticNet (for C6H6) or Ridge (for NOx, NO2)
- **Overall performance**: Ridge model achieves the best performance on all time horizons for NOx and NO2, Lasso performs excellently on multiple time horizons for C6H6, and XGBoost shows stable performance on medium-to-long-term prediction for CO

## 6. Visualization Results

The following visualization charts were generated (located in `outputs/figures/` directory):

### 6.1 Exploratory Data Analysis
- **pollutants_distribution.png**: Distribution histograms of pollutants, showing basic statistical properties of the data

### 6.2 Performance Comparison
- **performance_heatmap.png**: Performance heatmap of all models on all tasks
- **performance_heatmap_best_models.png**: Best model performance heatmap
- **multi_pollutant_comparison.png**: Multi-pollutant performance comparison chart
- **pollutant_comparison_by_horizon.png**: Pollutant comparison chart grouped by time horizon

### 6.3 Diagnostic Analysis

![Predicted vs Actual Scatter Plot](./figures/diagnostics/predicted_vs_actual.png)
*Figure 8: Predicted vs actual scatter plot, showing model prediction accuracy*

- **residual_vs_predicted.png**: Residual vs predicted value plot
- **residual_distributions.png**: Residual distribution plot
- **residual_qq_plots.png**: Residual Q-Q plot

### 6.4 In-Depth Analysis
- **feature_importance_multi_pollutant.png**: Multi-pollutant feature importance analysis (inserted as Figure 2)
- **error_comparison.png**: Error comparison analysis (inserted as Figure 7)
- **error_distributions.png**: Error distribution plot

![Time Series Prediction Visualization](./figures/analysis/time_series_prediction.png)
*Figure 9: Time series prediction visualization, showing model prediction effects on time series*

## 7. Conclusions

Through systematic experimental evaluation, this study provides an in-depth analysis of multi-pollutant multi-time-horizon air quality prediction tasks. The main conclusions are as follows:

1. **Regularized linear models (Ridge, Lasso, ElasticNet) show the best overall performance**, indicating that for this type of time series regression task, linear models with appropriate regularization can already achieve good results.

2. **Lasso excels in short-term prediction tasks**, achieving the best performance on CO and C6H6 1-hour predictions, demonstrating that L1 regularization can effectively perform feature selection and improve short-term prediction accuracy. **XGBoost shows stable performance in medium-to-long-term prediction**, achieving the best results consecutively on CO 6-24 hour predictions.

3. **Different pollutants have different prediction difficulties**, with CO being the easiest to predict and NOx being the most difficult, which is related to the physicochemical properties and variation patterns of each pollutant.

4. **Time horizon selection significantly affects prediction performance**, with short-term prediction (1 hour) clearly outperforming medium-to-long-term predictions, but 24-hour prediction performance is actually better than 6-12 hour predictions, possibly due to daily cycle patterns.

5. **Importance of feature engineering**: Through lag features and rolling window features, temporal dependencies in time series were successfully captured, providing effective inputs for models.

6. **Model selection should be task-specific**: The best models differ for different pollutants and time horizons, and in practical applications, the most suitable model should be selected based on specific scenarios.

## 8. Future Work Recommendations

1. **Feature engineering optimization**: Explore more time series features, such as Fourier transform features, wavelet features, etc.
2. **Model ensemble**: Try ensemble methods combining multiple models, which may further improve prediction performance
3. **Deep learning models**: Try LSTM, GRU and other deep learning models specifically designed for time series
4. **External features**: Consider introducing external features such as weather and traffic
5. **Online learning**: Explore online learning mechanisms to enable models to adapt to changes in data distribution

## 9. Technical Details

### 9.1 Experimental Environment
- **CPU cores**: 208 cores
- **Random seed**: 42
- **Parallel computing**: Fully utilize multi-core for model training and hyperparameter search

### 9.2 Data Splits
- **Training set**: 6,525 samples
- **Validation set**: 1,404 samples (for hyperparameter tuning)
- **Test set**: 1,404 samples (for final performance evaluation)

### 9.3 Model Training Time
- **Linear models**: 0.05-4 seconds/task
- **Tree models**: 5-200 seconds/task
- **Neural networks**: 5-37 seconds/task

## 10. References

- AirQualityUCI dataset
- Scikit-learn documentation
- XGBoost documentation
- Related time series prediction research literature

---

**Report Generation Time**: 2025
**Experimental Code**: `HW3-regression.ipynb`
**Results Data**: `outputs/results/regression_results_all_pollutants.csv`

