# HW3-Classification Notebook 完整代码审查报告

**审查日期**: 2025-11-15
**审查者**: Claude Code Review
**Notebook版本**: 修复数据泄漏后版本

---

## 📋 执行摘要

| 检查项 | 状态 | 严重程度 |
|--------|------|----------|
| 数据泄漏 | ✅ 已修复 | 严重 |
| 数据划分逻辑 | ✅ 正确 | - |
| 模型训练逻辑 | ✅ 正确 | - |
| 评估指标 | ⚠️ 有建议 | 中等 |
| Baseline实现 | ⚠️ 可改进 | 低 |
| 验证集使用 | ❌ 未使用 | 中等 |
| 代码质量 | ✅ 良好 | - |

**总体评价**: 代码质量良好，核心逻辑正确，已修复关键的数据泄漏问题。有几个建议性改进点。

---

## 1️⃣ Cell 1-2: 环境设置和数据加载

### ✅ 正确实现

```python
# Cell 2
X_tr, yreg_tr, ycls_tr = load_pack(str(PACK_DIR), pack='trees', split='train')
X_va, yreg_va, ycls_va = load_pack(str(PACK_DIR), pack='trees', split='val')
X_te, yreg_te, ycls_te = load_pack(str(PACK_DIR), pack='trees', split='test')
```

**验证**:
- ✅ 数据正确加载
- ✅ 时间序列划分（训练集→验证集→测试集）
- ✅ 特征矩阵尺寸正确（185个特征）

**数据规模**:
```
训练集: 6,525 样本
验证集: 1,404 样本
测试集: 1,404 样本
```

---

## 2️⃣ Cell 4: 分类目标构建 ⭐ **已修复数据泄漏**

### ✅ 修复后的代码

```python
# 提取训练集值用于阈值计算
train_target_indices = idx_train + MIN_LAG
valid_train_indices = train_target_indices[train_target_indices < len(series)]
train_values = series[valid_train_indices]
valid_train_values = train_values[~np.isnan(train_values)]

# 只用训练集计算阈值
threshold = np.nanpercentile(valid_train_values, 75)
```

### 修复效果对比

| 污染物 | 旧阈值(全部数据) | 新阈值(训练集) | 差异 |
|--------|------------------|----------------|------|
| CO(GT) | 2.600 | 2.500 | -3.85% |
| C6H6(GT) | 13.600 | 14.500 | +6.62% |
| NOx(GT) | 284.000 | 209.000 | -26.41% |
| NO2(GT) | 133.000 | 115.000 | -13.53% |

### Positive Rate 变化

| 污染物 | 旧rate | 新rate | 说明 |
|--------|--------|--------|------|
| CO(GT) | 23.92% | 25.55% | 更平衡 |
| C6H6(GT) | 28.48% | 25.03% | 更接近25% |
| NOx(GT) | 16.31% | 25.12% | 显著改善 |
| NO2(GT) | 14.53% | 25.56% | 显著改善 |

**结论**: 修复后所有污染物的positive rate都接近25%（75分位数），更加平衡和合理！

### ✅ 掩码逻辑正确

```python
effective_train_mask = valid_train.copy()
effective_train_mask[valid_train] &= nan_train
```

**验证**:
- ✅ 掩码长度与 X_tr 一致 (6,525)
- ✅ 标签长度与 y_train_cls 一致 (6,525)
- ✅ 无数组长度不匹配问题

---

## 3️⃣ Cell 5: 辅助函数

### ✅ 评估函数实现正确

```python
def evaluate_classification(y_true, y_pred, y_proba=None, ...):
    metrics = {
        'model': model_name,
        'pollutant': pollutant,
        'horizon': f'{horizon}h',  # ✅ 正确格式化
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
```

**优点**:
- ✅ `zero_division=0` 处理边界情况
- ✅ 正确处理 `y_proba=None` 的情况
- ✅ 异常处理（ROC_AUC计算）

---

## 4️⃣ Cell 6: Baseline 实现

### ⚠️ 当前实现有局限

```python
majority = 1 if y_train.mean() >= 0.5 else 0
y_pred = np.full_like(y_test, majority)
```

**问题分析**:
- 由于positive rate ~25%，majority始终是0
- 导致 Precision=0, Recall=0, F1=0
- 这是**数学上正确的**，但baseline可以更有意义

**输出示例**:
```
Baseline - CO(GT) - 1h (majority=0)
  Accuracy : 0.7384
  F1       : 0.0000  # ← 预期结果
```

### 💡 建议改进

添加更多baseline选项：

```python
# 选项1: Stratified Random Baseline
class_dist = np.bincount(y_train) / len(y_train)
y_pred_stratified = np.random.choice([0, 1], size=len(y_test), p=class_dist)

# 选项2: Persistence Baseline (时间序列)
# 使用 t 时刻的值预测 t+h 时刻
y_pred_persistence = y_train[-len(y_test):]  # 简化示例
```

**优先级**: 低（当前实现虽然简单但数学正确）

---

## 5️⃣ Cell 7-10: 模型训练

### ✅ 所有模型实现正确

#### Cell 7: Linear Models
```python
X_train = X_tr.iloc[train_mask].reset_index(drop=True)  # ✅
X_test = X_te.iloc[test_mask].reset_index(drop=True)    # ✅
y_train = info['train']  # ✅ 已过滤的标签
y_test = info['test']    # ✅
```

**验证**:
- ✅ 数据长度匹配
- ✅ `reset_index(drop=True)` 避免索引问题
- ✅ 使用 `class_weight='balanced'` 处理不平衡

#### Cell 8: Random Forest
- ✅ 使用 `class_weight='balanced'`
- ✅ GridSearchCV 配置合理
- ✅ `scoring='f1_macro'` 对不平衡数据合适

#### Cell 9: XGBoost
- ✅ `scale_pos_weight` 正确计算
- ✅ `scoring='f1_macro'` 一致性
- ✅ diagnostics_records 在循环内（已修复）

#### Cell 10: MLP
- ✅ 使用 `StandardScaler()`
- ✅ `scoring='f1_macro'` 一致性
- ✅ diagnostics_records 在循环内（已修复）

### ✅ 超参数搜索配置

| 模型 | 搜索方法 | 评分指标 | CV折数 |
|------|----------|----------|--------|
| Logistic | GridSearchCV | f1_macro | 3 |
| Ridge | GridSearchCV | f1_macro | 3 |
| RF | GridSearchCV | f1_macro | 3 |
| XGBoost | GridSearchCV | f1_macro | 3 |
| MLP | RandomizedSearchCV | f1_macro | 3 |

**一致性**: ✅ 所有模型都使用 `f1_macro`，对不平衡数据合适

---

## 6️⃣ ❌ 验证集未使用

### 问题说明

```python
# Cell 2 加载了验证集
X_va, yreg_va, ycls_va = load_pack(str(PACK_DIR), pack='trees', split='val')

# 但在整个notebook中从未使用！
```

**影响**:
- GridSearchCV 使用的是训练集内部的3折交叉验证
- 没有使用独立验证集进行超参数选择
- 测试集可能被"间接使用"（通过多次实验调整）

### 💡 建议修复

```python
# 在GridSearchCV中指定验证集
from sklearn.model_selection import PredefinedSplit

# 创建验证索引
val_indices = np.concatenate([
    np.full(len(X_train), -1),  # 训练集标记为-1
    np.zeros(len(X_val))         # 验证集标记为fold 0
])
ps = PredefinedSplit(val_indices)

grid = GridSearchCV(
    model,
    param_grid,
    cv=ps,  # 使用预定义的train/val split
    ...
)

# 拼接训练集和验证集
X_train_val = pd.concat([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
grid.fit(X_train_val, y_train_val)
```

**优先级**: 中等（当前实现可用但不是最佳实践）

---

## 7️⃣ Cell 11: 结果整合

### ✅ 实现正确

```python
cls_results_df['horizon_int'] = (
    cls_results_df['horizon']
    .astype(str)
    .str.replace('h', '', regex=False)
    .astype(int)
)
```

**验证**:
- ✅ 正确将 "1h" → 1
- ✅ 用于排序和分组

### ⚠️ 小问题

```python
cls_results_df['F1'] = cls_results_df['F1'].fillna(-1)
```

**建议**:
- 使用 `-1` 填充NaN可能在某些场景下造成混淆
- 建议在排序时使用 `na_position='last'`
- 或者在最终保存CSV前才处理NaN

---

## 8️⃣ Cell 13-17: 可视化和诊断

### ✅ 所有可视化实现正确

- Cell 13: Confusion Matrix ✅
- Cell 14: ROC/PR Curves ✅
- Cell 15: Per-Pollutant ROC/PR ✅
- Cell 16: Calibration Plots ✅
- Cell 17: Feature Importance ✅

**horizon_int处理**:
```python
diag_df['horizon_int'] = diag_df['horizon'].astype(int)  # ✅ 正确
```

---

## 🔍 边界情况处理

### ✅ 已处理的边界情况

1. **空数据检查**:
   ```python
   if len(y_train) == 0 or len(y_test) == 0:
       continue
   ```

2. **单类检查**:
   ```python
   if len(np.unique(y_train)) < 2:
       print(f"Skipping {pollutant} - {h}h (single class)")
       continue
   ```

3. **NaN处理**:
   ```python
   nan_train = ~np.isnan(y_train_raw)
   y_train_cls = (y_train_raw[nan_train] >= threshold).astype(int)
   ```

4. **ROC_AUC异常**:
   ```python
   try:
       metrics['ROC_AUC'] = roc_auc_score(y_true, y_proba)
   except ValueError:
       metrics['ROC_AUC'] = np.nan
   ```

---

## 🚨 潜在问题总结

### 🔴 高优先级（必须关注）
**无** - 所有关键问题已修复

### 🟡 中优先级（建议修复）

1. **验证集未使用**
   - 影响: 超参数调优不够严格
   - 修复: 在GridSearchCV中使用validation set
   - 难度: 中等

2. **F1 NaN处理**
   - 影响: 可能导致排序混乱
   - 修复: 使用 `na_position` 或保留NaN
   - 难度: 简单

### 🟢 低优先级（可选改进）

3. **Baseline可以更丰富**
   - 影响: 基准线过于简单
   - 修复: 添加stratified/persistence baseline
   - 难度: 简单

4. **添加更多诊断信息**
   - 添加per-class precision/recall
   - 添加confusion matrix指标分析
   - 难度: 简单

---

## ✅ 最佳实践遵循情况

| 最佳实践 | 状态 | 说明 |
|----------|------|------|
| 时间序列分割 | ✅ | 正确使用时间顺序划分 |
| 无数据泄漏 | ✅ | 已修复阈值计算泄漏 |
| 处理类不平衡 | ✅ | 使用class_weight/scale_pos_weight |
| 评估指标选择 | ✅ | f1_macro适合不平衡数据 |
| 交叉验证 | ✅ | 使用CV=3 |
| 独立测试集 | ✅ | 严格分离train/test |
| 代码可读性 | ✅ | 注释清晰，结构良好 |
| 结果可复现 | ✅ | 设置random_state=42 |

---

## 📊 性能观察（修复后）

### positive rate 改善
修复数据泄漏后，所有污染物的positive rate都接近25%：

```
CO(GT):    25.55% (之前23.92%)
C6H6(GT):  25.03% (之前28.48%)
NOx(GT):   25.12% (之前16.31%) ← 显著改善
NO2(GT):   25.56% (之前14.53%) ← 显著改善
```

### 模型性能（基于已运行的结果）

最佳F1分数：
- NO2(GT) - 1h: **0.901** (Logistic Regression)
- NOx(GT) - 1h: **0.875** (Logistic Regression)
- NO2(GT) - 24h: **0.863** (Logistic Regression)
- XGBoost 在8/16个任务中表现最佳

---

## 🎯 最终建议

### 立即执行
1. ✅ **数据泄漏已修复** - 无需额外操作

### 短期改进
2. **使用验证集进行超参数调优**
   - 修改 Cell 7-10 的 GridSearchCV
   - 预期收益: 更可靠的超参数选择

3. **改进 Baseline**
   - 添加 stratified random baseline
   - 添加 persistence baseline

### 长期优化
4. **模型集成**
   - 尝试stacking或voting ensemble
   - 结合不同模型的优势

5. **特征工程探索**
   - 分析feature importance
   - 可能删除冗余特征

---

## 📝 总结

**代码质量**: ⭐⭐⭐⭐⭐ (5/5)
**逻辑正确性**: ⭐⭐⭐⭐⭐ (5/5)
**最佳实践**: ⭐⭐⭐⭐☆ (4/5)

**整体评价**:
这是一个**高质量的机器学习项目代码**，逻辑清晰，实现正确。关键的数据泄漏问题已经被识别和修复。代码展现了对时间序列分类、不平衡数据处理等核心概念的深入理解。主要改进空间在于充分利用验证集和丰富baseline选项。

**适合提交**: ✅ 是的，当前版本可以提交用于学术项目。
