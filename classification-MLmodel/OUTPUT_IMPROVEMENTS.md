# Classification Notebook 输出检查与改进建议

基于对 `classification-ML.ipynb` 的全面检查，以下是发现的问题和改进建议。

## 📊 当前输出状态

### ✅ 正常工作的部分

1. **结果保存**：CSV文件正确保存到 `outputs/results/classification_results_all_pollutants.csv`
2. **可视化输出**：所有图表都正确保存到 `outputs/figures/` 目录
3. **模型训练输出**：每个模型的训练结果都有打印输出
4. **数据质量检查**：Cell 4 有完整的数据质量检查输出

### ⚠️ 发现的问题

## 🔴 严重问题

### 1. **24h预测性能严重不足**

**问题描述**：
- Persistence Baseline (0.5619) > XGBoost (0.5056) > Random Forest (0.4746)
- 机器学习模型在24h预测上表现不如简单的基线模型

**根本原因**：
- 特征工程缺少长期依赖特征（48h, 72h, 168h滞后）
- 所有时间范围使用相同的特征集，没有针对24h的特殊设计

**改进建议**：
```python
# 在特征工程中，为24h预测添加长期特征
def create_features_for_horizon(df, horizon):
    """为特定预测范围创建特征"""
    if horizon == 24:
        # 长期预测：强调长期模式和周期性
        lags = [24, 48, 72, 168]  # 1天, 2天, 3天, 1周
        windows = [24, 48, 72, 168]
        # 添加更多周期性特征
        features['day_of_year'] = df['timestamp'].dt.dayofyear
        features['week_of_year'] = df['timestamp'].dt.isocalendar().week
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # ... 其他时间范围的特征
```

### 2. **MLP模型表现异常差**

**问题描述**：
- 平均F1只有0.504，在所有模型中最低
- 12h F1只有0.4319，远低于其他模型

**当前代码问题**：
- 虽然已经添加了early_stopping，但网络结构可能还不够优化
- 超参数搜索空间可能不够充分

**改进建议**：
```python
# 1. 添加BatchNormalization和Dropout
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

# 2. 使用更深的网络结构
MLP_DIST = {
    'model__hidden_layer_sizes': [
        (512, 256, 128),           # 更深的三层
        (256, 256, 128, 64),       # 四层网络
        (512, 256, 128, 64),       # 更深更宽
    ],
    'model__learning_rate_init': [1e-4, 5e-4, 1e-3],
    'model__alpha': [1e-5, 1e-4, 1e-3],  # L2正则化
    'model__batch_size': [64, 128, 256],
    'model__activation': ['relu', 'tanh'],
    'model__beta_1': [0.9, 0.95],  # Adam参数
    'model__beta_2': [0.999, 0.9999]
}

# 3. 增加搜索迭代次数
search = RandomizedSearchCV(
    MLP_PIPE,
    MLP_DIST,
    n_iter=50,  # 从30增加到50
    scoring='f1_macro',
    cv=ps,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    verbose=1  # 显示进度
)
```

## 🟡 中等问题

### 3. **特征选择功能被禁用**

**问题描述**：
- `USE_FEATURE_SELECTION = False` - 特征选择功能被禁用
- 185个特征中可能存在冗余特征

**改进建议**：
```python
# 在Cell 6中启用特征选择
USE_FEATURE_SELECTION = True  # 改为True
FEATURE_SELECTION_METHOD = 'mutual_info'  # 或 'f_test', 'rfe'
N_SELECTED_FEATURES = 100  # 选择top 100特征

# 在模型训练前应用特征选择
if USE_FEATURE_SELECTION:
    selector = create_feature_selector(
        method=FEATURE_SELECTION_METHOD,
        n_features=N_SELECTED_FEATURES
    )
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
```

### 4. **概率校准功能被禁用**

**问题描述**：
- `USE_PROBABILITY_CALIBRATION = False` - 概率校准被禁用
- 模型输出的概率可能不够准确

**改进建议**：
```python
# 在Cell 14中启用概率校准
USE_PROBABILITY_CALIBRATION = True

# 对XGBoost和Random Forest进行校准
if USE_PROBABILITY_CALIBRATION:
    from sklearn.calibration import CalibratedClassifierCV
    
    calibrated_model = CalibratedClassifierCV(
        best_model,
        method='isotonic',  # 或 'sigmoid'
        cv=5
    )
    calibrated_model.fit(X_train, y_train)
    y_proba = calibrated_model.predict_proba(X_test)
```

### 5. **结果输出缺少详细信息**

**问题描述**：
- Cell 16只输出了最佳模型的简单信息
- 缺少每个类别的详细指标（per-class metrics）
- 缺少模型稳定性信息

**改进建议**：
```python
# 在Cell 16中添加更详细的输出
print('=== Best model per pollutant & horizon (by F1) ===')
print(best_models[['pollutant', 'horizon', 'model', 'F1', 'Accuracy', 'ROC_AUC']])

# 添加：每个类别的详细指标
from sklearn.metrics import classification_report
for idx, row in best_models.iterrows():
    # 获取该模型的预测结果
    model_name = row['model']
    horizon = row['horizon']
    pollutant = row['pollutant']
    
    # 从diagnostics_records中获取预测结果
    # ... 计算并打印per-class metrics
    print(f"\n{classification_report(y_true, y_pred, target_names=['Low', 'Mid', 'High'])}")

# 添加：模型稳定性评估
print('\n=== Model Stability Analysis ===')
# 多次运行模型，计算F1的标准差
```

### 6. **缺少针对不同时间范围的特征工程**

**问题描述**：
- 所有时间范围（1h, 6h, 12h, 24h）使用相同的特征集
- 短期预测和长期预测需要不同的特征

**改进建议**：
```python
# 为不同时间范围创建不同的特征集
def create_horizon_specific_features(df, horizon):
    """为特定预测范围创建特征"""
    features = {}
    
    if horizon == 1:
        # 短期预测：强调最近的特征
        lags = [1, 2, 3, 6]
        windows = [3, 6]
        # 添加短期趋势特征
        features['short_term_trend'] = df['CO(GT)'].diff(1)
        
    elif horizon == 6:
        lags = [1, 3, 6, 12]
        windows = [6, 12]
        
    elif horizon == 12:
        lags = [6, 12, 24]
        windows = [12, 24]
        
    else:  # 24h
        # 长期预测：强调长期模式和周期性
        lags = [24, 48, 72, 168]  # 1天, 2天, 3天, 1周
        windows = [24, 48, 72, 168]
        # 添加周期性特征
        features['day_of_year'] = df['timestamp'].dt.dayofyear
        features['week_of_year'] = df['timestamp'].dt.isocalendar().week
        features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # 添加长期趋势
        features['weekly_avg'] = df['CO(GT)'].rolling(168).mean()
    
    return create_features_with_lags(df, lags, windows, **features)
```

## 🟢 次要问题

### 7. **警告信息被完全屏蔽**

**问题描述**：
- `warnings.filterwarnings('ignore')` - 所有警告被屏蔽
- 可能隐藏重要的信息

**改进建议**：
```python
# 只屏蔽特定类型的警告
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
# 保留其他重要警告
```

### 8. **缺少模型集成**

**问题描述**：
- 只使用单个模型，没有尝试集成方法
- 集成可能提升性能

**改进建议**：
```python
# 添加模型集成
from sklearn.ensemble import VotingClassifier, StackingClassifier

# 针对不同时间范围的集成策略
if horizon == 1:
    # 1h: XGBoost + Random Forest
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', best_rf)
        ],
        voting='soft',
        weights=[2, 1]
    )
elif horizon == 24:
    # 24h: Persistence + XGBoost (因为Persistence在24h表现好)
    ensemble = VotingClassifier(
        estimators=[
            ('persistence', persistence_model),
            ('xgb', best_xgb)
        ],
        voting='soft',
        weights=[1.5, 1]
    )
```

### 9. **缺少错误分析的详细输出**

**问题描述**：
- Cell 22有错误分析，但输出可能不够详细
- 缺少按时间段、按类别的错误分析

**改进建议**：
```python
# 在Cell 22中添加更详细的错误分析
# 1. 按时间段的错误分析
df_errors['hour'] = df_errors['timestamp'].dt.hour
error_by_hour = df_errors.groupby('hour')['error_type'].value_counts()
print("\n错误按时间段分布:")
print(error_by_hour)

# 2. 按类别的错误分析
error_by_class = df_errors.groupby('true_class')['error_type'].value_counts()
print("\n错误按类别分布:")
print(error_by_class)

# 3. 高置信度错误分析
high_confidence_errors = df_errors[
    (df_errors['pred_proba'] > 0.9) & (df_errors['is_error'] == True)
]
print(f"\n高置信度错误数量: {len(high_confidence_errors)}")
```

### 10. **ROC_AUC列有缺失值**

**问题描述**：
- Persistence Baseline和Ridge Classifier的ROC_AUC为空
- 这会影响结果分析

**改进建议**：
```python
# 在结果保存前，填充缺失的ROC_AUC值
cls_results_df['ROC_AUC'] = cls_results_df['ROC_AUC'].fillna(0)
# 或者为不支持概率的模型计算其他指标
```

## 📋 改进优先级

### 🔴 高优先级（立即实施）

1. **修复24h预测问题** - 添加长期依赖特征
2. **改进MLP模型** - 优化网络结构和超参数
3. **启用特征选择** - 减少冗余特征
4. **启用概率校准** - 提升概率预测准确性

### 🟡 中优先级（短期实施）

5. **添加详细的结果输出** - per-class metrics, 稳定性分析
6. **针对不同时间范围的特征工程** - 为每个时间范围设计专门的特征
7. **模型集成** - 尝试Voting和Stacking

### 🟢 低优先级（长期优化）

8. **改进错误分析输出** - 更详细的错误模式分析
9. **处理缺失的ROC_AUC值** - 为所有模型计算ROC_AUC
10. **优化警告处理** - 只屏蔽特定警告

## 🎯 预期改进效果

如果实施所有高优先级改进：

| 时间范围 | 当前最佳F1 | 预期改进后F1 | 提升幅度 |
|---------|-----------|-------------|---------|
| 1h | 0.7294 (XGBoost) | 0.75-0.78 | +3-7% |
| 6h | 0.5801 (XGBoost) | 0.62-0.65 | +7-12% |
| 12h | 0.5333 (XGBoost) | 0.58-0.62 | +9-16% |
| 24h | 0.5619 (Persistence) | 0.60-0.65 | +7-16% |

**最关键**的是解决24h预测问题，因为Persistence Baseline超过了所有ML模型。

## 📝 总结

当前notebook的主要问题：

1. ✅ **代码运行正常** - 所有输出都正确保存
2. ⚠️ **24h预测能力弱** - 需要专门的特征工程
3. ⚠️ **MLP表现差** - 需要进一步优化
4. ⚠️ **功能被禁用** - 特征选择和概率校准需要启用
5. ⚠️ **输出不够详细** - 缺少per-class metrics和稳定性分析

建议按照优先级逐步实施改进，重点关注24h预测问题的解决。





