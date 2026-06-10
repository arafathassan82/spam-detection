import json
from pathlib import Path
from typing import Any
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
 
 
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics
 
 
def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
 
 
def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def plot_roc_curve(y_true, y_prob, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def plot_model_comparison(results: dict, save_path: Path):
    """Bar chart comparing models across key metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    n_metrics = len(metrics)
    x = np.arange(len(models))
    width = 0.2
 
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        vals = [results[m].get(metric, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=metric.upper())
 
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.7, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
 
 
def plot_feature_importance(importances: np.ndarray, n_top: int, save_path: Path, title: str = "Feature Importance"):
    top_idx = np.argsort(importances)[-n_top:]
    top_vals = importances[top_idx]
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(n_top), top_vals, color="steelblue")
    ax.set_yticks(range(n_top))
    ax.set_yticklabels([f"feature_{i}" for i in top_idx], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)