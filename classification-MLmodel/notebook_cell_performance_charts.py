# ======================================================
# Cell: Performance Metrics Line Charts - Multi-Model Comparison
# ======================================================
# 将此单元格添加到你的 Jupyter Notebook 中（Cell 15 或更后面）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 准备数据（使用已有的 cls_results_df）
# =========================

# 检查 cls_results_df 是否存在
if 'cls_results_df' not in locals():
    print("⚠ cls_results_df not found. Please run Cell 11 first.")
else:
    # 确保 horizon_int 列存在
    if 'horizon_int' not in cls_results_df.columns:
        cls_results_df['horizon_int'] = (
            cls_results_df['horizon']
            .astype(str)
            .str.replace('h', '', regex=False)
            .astype(int)
        )

    # 获取所有唯一的模型和时间范围
    models = sorted(cls_results_df['model'].unique())
    horizons = sorted(cls_results_df['horizon_int'].unique())

    print(f"📊 Data Summary:")
    print(f"  Models: {len(models)} models")
    print(f"  Horizons: {horizons}")

    # 定义指标映射
    metrics = {
        "Accuracy": "Accuracy",
        "F1": "F1 Score (Macro)",
        "Precision": "Precision (Macro)",
        "Recall": "Recall (Macro)",
    }

    # 定义模型颜色（与原notebook保持一致）
    model_colors = {
        'Logistic Regression': '#E63946',
        'Ridge Classifier': '#F77F00',
        'Random Forest': '#06A77D',
        'XGBoost': '#2E86AB',
        'MLP': '#A23B72',
        'Persistence Baseline': '#6C757D'
    }

    # =========================
    # 2. 绘制 2×2 指标折线图
    # =========================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (col_name, title) in zip(axes, metrics.items()):
        for model in models:
            # 获取该模型在各个 horizon 的数据
            model_data = cls_results_df[cls_results_df['model'] == model].sort_values('horizon_int')

            if len(model_data) == 0:
                continue

            # 绘制折线
            color = model_colors.get(model, '#000000')
            ax.plot(
                model_data['horizon_int'],
                model_data[col_name],
                marker='o',
                linewidth=2.5,
                label=model,
                color=color,
                markersize=8
            )

        # 设置标题和标签
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Forecast Horizon (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=11, fontweight='bold')

        # 设置 x 轴刻度
        ax.set_xticks(horizons)
        ax.set_xticklabels([f'{h}h' for h in horizons])

        # 添加网格
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)

        # 设置 y 轴范围
        ax.set_ylim([0, 1.05])

        # 添加图例（只在第一个子图）
        if ax == axes[0]:
            ax.legend(title="Model", fontsize=9, loc='lower left', framealpha=0.9)

    # 添加总标题
    fig.suptitle(
        'Classification Performance Comparison: All Models Across Forecast Horizons',
        fontsize=15,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # 保存图片
    output_path = PERFORMANCE_DIR / 'metrics_line_charts_all_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Saved line charts to: {output_path}")

    # =========================
    # 3. 额外：单个指标的详细折线图（F1 Score）
    # =========================

    fig, ax = plt.subplots(figsize=(12, 7))

    for model in models:
        model_data = cls_results_df[cls_results_df['model'] == model].sort_values('horizon_int')

        if len(model_data) == 0:
            continue

        color = model_colors.get(model, '#000000')
        ax.plot(
            model_data['horizon_int'],
            model_data['F1'],
            marker='o',
            linewidth=3,
            label=model,
            color=color,
            markersize=10
        )

        # 添加数值标签
        for x, y in zip(model_data['horizon_int'], model_data['F1']):
            ax.annotate(
                f'{y:.3f}',
                xy=(x, y),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color=color,
                fontweight='bold'
            )

    ax.set_title(
        'F1 Score (Macro) Comparison Across All Models and Horizons',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (Macro)', fontsize=12, fontweight='bold')
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'{h}h' for h in horizons])
    ax.legend(title="Model", fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()

    # 保存图片
    output_path_f1 = PERFORMANCE_DIR / 'f1_score_detailed_line_chart.png'
    plt.savefig(output_path_f1, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved F1 detailed chart to: {output_path_f1}")

    # =========================
    # 4. 打印数值汇总表
    # =========================

    print("\n" + "="*80)
    print("📋 Performance Summary Table (F1 Score)")
    print("="*80)

    # 创建透视表
    pivot_table = cls_results_df.pivot_table(
        index='model',
        columns='horizon_int',
        values='F1',
        aggfunc='mean'
    )

    # 添加平均分列
    pivot_table['Average'] = pivot_table.mean(axis=1)

    # 按平均分排序
    pivot_table = pivot_table.sort_values('Average', ascending=False)

    # 显示表格
    print(pivot_table.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n" + "="*80)
    print("✓ Line charts visualization completed!")
    print("="*80)
