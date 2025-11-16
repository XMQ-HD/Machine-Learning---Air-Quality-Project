import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from sklearn.metrics import confusion_matrix
import warnings


def plot_single_model_results(
    model_name: str,
    horizon: int,
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = [
        ('accuracy', 'Model Accuracy', 'Accuracy'),
        ('loss', 'Model Loss', 'Loss'),
        ('precision', 'Model Precision', 'Precision'),
        ('recall', 'Model Recall', 'Recall')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        axes[idx].plot(history[metric], label='Train', linewidth=2)
        axes[idx].plot(history[f'val_{metric}'], label='Validation', linewidth=2)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=10)
        axes[idx].set_ylabel(ylabel, fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_h{horizon}_history.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Low', 'Medium', 'High']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    threshold = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]),
                   ha="center", va="center",
                   color="white" if cm[i, j] > threshold else "black",
                   fontsize=14, fontweight='bold')
    
    ax.set_title(f'Confusion Matrix - {model_name} ({horizon}h Forecast)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_h{horizon}_confusion.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

# comparison
def plot_model_comparison(results: Dict, output_dir: Path):
    horizons = [1, 6, 12, 24]
    metrics_to_plot = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    metric_names = ['Accuracy', 'F1 Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        ax = axes[idx]
        
        for model_name, horizon_results in results.items():
            values = [horizon_results[h][metric] for h in horizons]
            ax.plot(horizons, values, marker='o', label=model_name, linewidth=2, markersize=8)
        
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Forecast Horizon (hours)', fontsize=10)
        ax.set_ylabel(name, fontsize=10)
        ax.set_xticks(horizons)
        ax.set_xticklabels([f'{h}h' for h in horizons])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()