# 特征工程增强使用指南

## 📋 概述

已在 `classification-ML.ipynb` 中添加了**Cell 6.5: Enhanced Feature Engineering**，用于在notebook中进行特征工程增强，无需修改 `prepare_airquality.py`。

## 🎯 新增特征类型

### 1. 交互特征 (Interaction Features)
- **污染物比值**：CO与其他污染物的比值（如 `CO_NO2_ratio`）
- **污染物乘积**：CO与其他污染物的乘积（如 `CO_NO2_product`）
- **作用**：捕获污染物之间的相关性和协同效应

### 2. 差分特征 (Difference Features)
- **短期差分**：1h-3h, 1h-6h的变化
- **长期差分**：6h-24h的变化
- **百分比变化**：24小时内的百分比变化
- **作用**：捕获时间序列的动态变化趋势

### 3. 周期性编码 (Cyclical Encoding)
- **小时周期性**：`hour_sin`, `hour_cos`（24小时周期）
- **星期周期性**：`dow_sin`, `dow_cos`（7天周期）
- **月份周期性**：`month_sin`, `month_cos`（12个月周期）
- **作用**：对线性模型和神经网络特别重要，能更好地捕获周期性模式

### 4. 时间范围专门特征
- **24h预测**：长期趋势特征、长期平均值
- **1h预测**：短期趋势特征
- **作用**：针对不同预测时间范围优化特征集

### 5. 滚动窗口增强特征
- **变异系数 (CV)**：`std / mean`，衡量相对波动性
- **作用**：补充已有的mean和std特征

## 🚀 使用方法

### 基本使用

1. **启用特征工程增强**：
   ```python
   USE_ENHANCED_FEATURES = True  # 在Cell 6.5中设置
   ```

2. **运行Cell 6.5**：
   - 该cell会自动对 `X_tr`, `X_va`, `X_te` 应用特征工程增强
   - 增强后的特征会自动替换原始特征

3. **继续后续训练**：
   - 后续的模型训练cell会自动使用增强后的特征
   - 无需修改其他代码

### 禁用特征工程

如果不想使用增强特征，只需设置：
```python
USE_ENHANCED_FEATURES = False
```

## 📊 预期效果

根据改进建议文档，预期改进效果：

| 时间范围 | 当前最佳F1 | 预期改进后F1 | 提升幅度 |
|---------|-----------|-------------|---------|
| 1h | 0.7294 | 0.75-0.78 | +3-7% |
| 6h | 0.5801 | 0.62-0.65 | +7-12% |
| 12h | 0.5333 | 0.58-0.62 | +9-16% |
| 24h | 0.5619 | 0.60-0.65 | +7-16% |

## 🔧 自定义特征工程

如果需要添加更多特征，可以修改 `create_enhanced_features()` 函数：

```python
def create_enhanced_features(X_base, df_clean=None, horizon=None):
    X_enhanced = X_base.copy()
    
    # 在这里添加你的自定义特征
    # 例如：
    # X_enhanced['my_custom_feature'] = ...
    
    return X_enhanced
```

## ⚠️ 注意事项

1. **特征数量**：增强后特征数量会增加（从185个增加到约200-220个）
2. **计算时间**：特征工程会增加一些计算时间，但通常可以忽略
3. **内存使用**：特征数量增加会略微增加内存使用
4. **特征选择**：如果使用特征选择（Cell 5.5），增强特征也会被考虑在内

## 📝 特征列表示例

增强后的特征包括：

**交互特征**：
- `CO_NO2_ratio`, `CO_NO2_product`
- `CO_NOx_ratio`, `CO_NOx_product`
- `CO_C6H6_ratio`, `CO_C6H6_product`
- 等等...

**差分特征**：
- `CO_diff_1h_3h`
- `CO_diff_1h_6h`
- `CO_diff_6h_24h`
- `CO_pct_change_24h`

**周期性特征**：
- `hour_sin`, `hour_cos`
- `dow_sin`, `dow_cos`
- `month_sin`, `month_cos`

**滚动窗口增强**：
- `CO(GT)__cv` (变异系数)
- 等等...

## 🔍 验证特征工程

运行Cell 6.5后，检查输出：
- 应该看到 "Total new features created: XX"
- 应该看到 "Enhanced features: XXX"（比原始特征多）

## 💡 下一步建议

1. **运行完整流程**：运行Cell 6.5后，继续运行后续的模型训练cell
2. **对比结果**：比较使用增强特征前后的模型性能
3. **特征重要性**：查看特征重要性分析，了解哪些新特征最重要
4. **进一步优化**：根据结果调整特征工程策略

## 📚 相关文档

- `OUTPUT_IMPROVEMENTS.md` - 完整的改进建议
- `IMPROVEMENT_RECOMMENDATIONS.md` - 特征工程详细建议
- `COMPLETE_PIPELINE_ANALYSIS.md` - 完整流程分析





