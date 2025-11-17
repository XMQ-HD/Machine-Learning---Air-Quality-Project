# 分类任务改进建议报告

基于代码审查和项目分析，以下是针对分类任务的详细改进建议，特别关注特征工程方面。

## 📊 当前状态总结

### 已实现的内容 ✅
- 基础特征工程：滞后特征（lag features）、滚动统计（rolling statistics）
- 时间特征：小时、天、月
- 数据预处理：缺失值处理、异常值处理（Winsorization）
- 多模型对比：Logistic Regression, Ridge, Random Forest, XGBoost, MLP
- 类别不平衡处理：class_weight='balanced'
- 验证集使用：已修复，使用 PredefinedSplit

### 当前特征集（基于 artifacts.json 分析）
- **特征数量**: 185个特征
- **特征类型**:
  - **滞后特征（lag features）**: [1, 2, 3, 6, 12, 24] 小时
  - **滚动窗口统计**: [3, 6, 12, 24] 小时窗口的 mean 和 std
  - **时间特征**: hour, dow (day of week), month, is_weekend
  - **缺失值标志**: `__was_missing` 标志（12个污染物）
  - **优先特征**: 12个污染物（CO, NO2, NOx, C6H6, NMHC, AH等）
  - **Gap标志**: `__gap` 标志（数据缺失间隔）

**当前特征工程代码位置**: `prepare_airquality.py` (lines 108-135)

---

## 🎯 改进建议（按优先级排序）

### 🔴 高优先级改进

#### 1. **特征工程增强**

##### 1.1 交互特征（Interaction Features）
**问题**: 当前特征主要是单变量特征，缺少变量间的交互信息。

**建议**:
```python
# 添加污染物之间的交互特征
# 例如：CO与其他污染物的比值、乘积等
features['CO_NO2_ratio'] = features['CO(GT)'] / (features['NO2(GT)'] + 1e-6)
features['CO_NOx_ratio'] = features['CO(GT)'] / (features['NOx(GT)'] + 1e-6)
features['CO_C6H6_ratio'] = features['CO(GT)'] / (features['C6H6(GT)'] + 1e-6)

# 添加乘积特征（可能捕获协同效应）
features['CO_NO2_product'] = features['CO(GT)'] * features['NO2(GT)']
```

**预期收益**: 捕获污染物之间的相关性，可能提升5-10%的F1分数

##### 1.2 滞后特征的优化
**问题**: 当前滞后特征为 [1, 2, 3, 6, 12, 24] 小时，缺少：
- 更细粒度的短期滞后（对1h预测重要）
- 长期滞后特征（对24h预测重要）
- 非整数小时滞后（如18h）

**建议**:
```python
# 在 prepare_airquality.py 中修改 lags 列表
# 当前: lags = [1, 2, 3, 6, 12, 24]
# 改进后:
lags = [1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 96, 168]  # 扩展到1周

# 或者针对不同预测范围使用不同的滞后集
lags_short = [1, 2, 3, 6, 12]  # 用于1h, 6h预测
lags_medium = [12, 18, 24, 36, 48]  # 用于12h预测
lags_long = [24, 48, 72, 96, 168]  # 用于24h预测
```

**预期收益**: 
- 短期预测（1h, 6h）可能提升3-8%的F1分数
- 长期预测（24h）可能提升2-5%的F1分数

##### 1.3 滚动窗口特征的多样化
**问题**: 当前只有 mean 和 std（在 prepare_airquality.py line 127-128），缺少：
- min, max（极值信息）
- median（对异常值更鲁棒）
- 趋势特征
- 变化率特征

**建议**:
```python
# 在 prepare_airquality.py 的滚动窗口部分（line 126-128）添加：
# 当前代码:
# X[f"{c}__roll_mean{W}"] = df[c].rolling(W, min_periods=max(2,W//2)).mean()
# X[f"{c}__roll_std{W}"]  = df[c].rolling(W, min_periods=max(2,W//2)).std()

# 改进后:
for W in rolls:
    roll = df[c].rolling(W, min_periods=max(2,W//2))
    X[f"{c}__roll_mean{W}"] = roll.mean()
    X[f"{c}__roll_std{W}"] = roll.std()
    X[f"{c}__roll_min{W}"] = roll.min()  # 新增
    X[f"{c}__roll_max{W}"] = roll.max()  # 新增
    X[f"{c}__roll_median{W}"] = roll.median()  # 新增
    X[f"{c}__roll_range{W}"] = roll.max() - roll.min()  # 新增：范围
    
    # 趋势特征（线性趋势斜率）
    X[f"{c}__roll_trend{W}"] = roll.apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else np.nan
    )
    
    # 变化率（相对于窗口起始值）
    X[f"{c}__roll_pct_change{W}"] = (
        (roll.apply(lambda x: x.iloc[-1] if len(x) > 0 else np.nan) - 
         roll.apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan)) /
        (roll.apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan) + 1e-6)
    )
```

**预期收益**: 捕获更复杂的时间模式，可能提升2-5%的F1分数

##### 1.4 周期性特征增强
**问题**: 当前只有简单的时间特征（hour, dow, month, is_weekend），缺少周期性编码。

**建议**:
```python
# 在 prepare_airquality.py 的时间特征部分（line 130-134）添加：
# 当前代码已有: hour, dow, month, is_weekend

# 添加周期性编码（sin/cos变换）- 对线性模型和神经网络特别重要
ts = df["timestamp"]
X["hour"] = ts.dt.hour
X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)  # 新增
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)  # 新增
X["dow"] = ts.dt.dayofweek
X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7)  # 新增
X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7)  # 新增
X["month"] = ts.dt.month
X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)  # 新增
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)  # 新增
X["is_weekend"] = (X["dow"] >= 5).astype("int8")

# 添加时间段特征（早晨、中午、晚上、夜间）
X["time_period"] = pd.cut(
    X["hour"], 
    bins=[0, 6, 12, 18, 24], 
    labels=[0, 1, 2, 3]  # 0=night, 1=morning, 2=afternoon, 3=evening
).astype(int)

# 添加是否为工作日早晨/晚上（交通高峰期）
X["is_rush_hour"] = ((X["hour"] >= 7) & (X["hour"] <= 9) | 
                     (X["hour"] >= 17) & (X["hour"] <= 19)).astype(int)
```

**预期收益**: 更好地捕获周期性模式，可能提升2-4%的F1分数（特别是对线性模型）

##### 1.5 差分特征（Difference Features）
**问题**: 缺少变化率信息。

**建议**:
```python
# 一阶差分（捕获变化趋势）
features['CO_diff_1h'] = df['CO(GT)'].diff(1)
features['CO_diff_6h'] = df['CO(GT)'].diff(6)
features['CO_diff_24h'] = df['CO(GT)'].diff(24)

# 二阶差分（捕获加速度）
features['CO_diff2_1h'] = features['CO_diff_1h'].diff(1)

# 百分比变化
features['CO_pct_change_1h'] = df['CO(GT)'].pct_change(1)
features['CO_pct_change_24h'] = df['CO(GT)'].pct_change(24)
```

**预期收益**: 捕获动态变化模式，可能提升3-6%的F1分数

#### 2. **特征选择（Feature Selection）**

**问题**: 185个特征可能包含冗余特征，影响模型性能和可解释性。

**建议**:
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 方法1: 基于统计的特征选择
selector = SelectKBest(score_func=mutual_info_classif, k=100)
X_selected = selector.fit_transform(X_train, y_train)

# 方法2: 基于模型的特征选择（RFE）
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
selector = RFE(estimator, n_features_to_select=100, step=10)
X_selected = selector.fit_transform(X_train, y_train)

# 方法3: 基于特征重要性的选择（XGBoost）
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
top_features = feature_importance.head(100)['feature'].tolist()
```

**预期收益**: 
- 减少过拟合风险
- 提升模型训练速度
- 可能提升2-5%的F1分数（通过去除噪声特征）

#### 3. **特征缩放优化**

**问题**: 当前只有树模型使用原始特征，神经网络使用标准化特征。

**建议**:
```python
# 尝试不同的缩放方法
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

# RobustScaler（对异常值更鲁棒）
scaler_robust = RobustScaler()
X_scaled_robust = scaler_robust.fit_transform(X_train)

# PowerTransformer（处理偏态分布）
scaler_power = PowerTransformer(method='yeo-johnson')
X_scaled_power = scaler_power.fit_transform(X_train)

# QuantileTransformer（转换为均匀分布）
scaler_quantile = QuantileTransformer(output_distribution='uniform')
X_scaled_quantile = scaler_quantile.fit_transform(X_train)
```

**预期收益**: 可能提升线性模型和神经网络的性能

---

### 🟡 中优先级改进

#### 4. **类别不平衡处理增强**

**问题**: 当前只使用 `class_weight='balanced'`，可能不够。

**建议**:
```python
# 方法1: SMOTE过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 方法2: ADASYN（自适应合成采样）
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# 方法3: 组合过采样和欠采样
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 方法4: 自定义类别权重（基于类别重要性）
class_weights = {0: 1.0, 1: 1.5, 2: 2.0}  # 给高污染类别更高权重
```

**预期收益**: 可能提升少数类（Class 2）的召回率，整体F1提升1-3%

#### 5. **阈值优化**

**问题**: 当前使用固定阈值（1.5和2.5 mg/m³），可能不是最优的。

**建议**:
```python
# 方法1: 基于分位数的动态阈值
# 使用训练集的33%和66%分位数
q33 = train_values.quantile(0.33)
q66 = train_values.quantile(0.66)

# 方法2: 基于F1分数优化的阈值
from sklearn.metrics import f1_score
thresholds = np.linspace(train_values.min(), train_values.max(), 100)
best_f1 = 0
best_threshold = None
for threshold in thresholds:
    y_pred = (y_val >= threshold).astype(int)
    f1 = f1_score(y_val_true, y_pred, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# 方法3: 基于业务需求的阈值
# 如果高污染预警更重要，可以调整阈值使Class 2更容易被识别
```

**预期收益**: 可能提升整体分类性能，特别是类别边界附近的样本

#### 6. **时间序列特定模型**

**问题**: 当前没有使用专门的时间序列模型。

**建议**:
```python
# 方法1: LSTM/GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

# 方法2: Transformer（时间序列）
from transformers import TimeSeriesTransformerModel

# 方法3: Prophet + 分类器组合
from prophet import Prophet
# 使用Prophet进行趋势分解，然后将残差用于分类
```

**预期收益**: 可能显著提升长期预测（12h, 24h）的性能

#### 7. **集成方法（Ensemble）**

**问题**: 当前只使用单个模型，没有集成。

**建议**:
```python
# 方法1: Voting Classifier
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('lr', lr_model)
    ],
    voting='soft',  # 使用概率投票
    weights=[2, 1.5, 1]  # 给XGBoost更高权重
)

# 方法2: Stacking
from sklearn.ensemble import StackingClassifier
stacking = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('mlp', mlp_model)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# 方法3: Blending（在验证集上训练元学习器）
# 使用验证集预测作为特征，训练元学习器
```

**预期收益**: 可能提升2-5%的F1分数，特别是通过结合不同模型的优势

---

### 🟢 低优先级改进

#### 8. **数据增强（Data Augmentation）**

**建议**:
```python
# 方法1: 时间窗口滑动
# 通过滑动窗口创建更多训练样本

# 方法2: 噪声注入
X_augmented = X_train + np.random.normal(0, 0.01, X_train.shape)

# 方法3: Mixup（用于时间序列）
# 混合两个样本创建新样本
```

#### 9. **超参数优化增强**

**建议**:
```python
# 使用Optuna进行更高效的超参数搜索
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
    }
    model = XGBClassifier(**params)
    # ... 训练和评估
    return f1_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### 10. **模型解释性增强**

**建议**:
```python
# 使用SHAP值进行特征重要性解释
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 使用LIME进行局部解释
from lime import lime_tabular
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values, 
    feature_names=X_train.columns
)
```

---

## 📈 预期改进效果汇总

| 改进项 | 预期F1提升 | 实施难度 | 优先级 |
|--------|-----------|---------|--------|
| 交互特征 | +5-10% | 中等 | 🔴 高 |
| 滞后特征优化 | +3-8% | 简单 | 🔴 高 |
| 滚动窗口多样化 | +2-5% | 中等 | 🔴 高 |
| 周期性特征增强 | +2-4% | 简单 | 🔴 高 |
| 差分特征 | +3-6% | 简单 | 🔴 高 |
| 特征选择 | +2-5% | 中等 | 🔴 高 |
| SMOTE过采样 | +1-3% | 简单 | 🟡 中 |
| 阈值优化 | +1-3% | 简单 | 🟡 中 |
| LSTM/GRU | +5-15% (长期) | 困难 | 🟡 中 |
| 集成方法 | +2-5% | 中等 | 🟡 中 |

**总体预期**: 通过实施高优先级改进，F1分数可能从当前的0.73（1h）提升到0.78-0.82

---

## 🛠️ 实施建议

### 阶段1: 快速改进（1-2天）
1. 添加短期滞后特征（1h, 3h, 6h, 12h）
2. 添加差分特征
3. 添加周期性编码（sin/cos）
4. 实施SMOTE过采样

### 阶段2: 特征工程增强（3-5天）
1. 添加交互特征
2. 增强滚动窗口统计
3. 实施特征选择
4. 优化阈值

### 阶段3: 高级方法（1-2周）
1. 实施LSTM/GRU模型
2. 实施集成方法
3. 使用Optuna进行超参数优化

---

## 📝 注意事项

1. **避免过拟合**: 添加更多特征时要小心过拟合，使用特征选择和交叉验证
2. **计算资源**: 某些改进（如LSTM）需要更多计算资源
3. **可解释性**: 在添加复杂特征时，保持模型的可解释性
4. **时间序列特性**: 确保所有特征工程都尊重时间序列的因果性（不使用未来信息）

---

## 🔍 验证方法

对于每个改进，建议：
1. 在验证集上测试性能提升
2. 进行A/B测试（对比改进前后的性能）
3. 使用交叉验证确保改进的稳定性
4. 分析改进对不同时间范围的影响

---

## 总结

当前分类任务已经有一个solid的基础，但通过系统性的特征工程改进，可以显著提升模型性能。建议优先实施高优先级的特征工程改进，这些改进通常能带来最大的性能提升，同时实施难度相对较低。

