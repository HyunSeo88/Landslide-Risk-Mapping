"""
Evaluation metrics for binary classification
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)
from typing import Dict, Tuple


def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        y_pred: Predicted probabilities (N, 1) or (N,)
        y_true: Ground truth labels (N, 1) or (N,)
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Flatten
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        'ap': average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    metrics['true_positive'] = int(tp)
    metrics['false_positive'] = int(fp)
    metrics['true_negative'] = int(tn)
    metrics['false_negative'] = int(fn)

    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Pretty print metrics

    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)

    # Main metrics
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print(f"AP:        {metrics['ap']:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positive']:4d}  |  FP: {metrics['false_positive']:4d}")
    print(f"  FN: {metrics['false_negative']:4d}  |  TN: {metrics['true_negative']:4d}")

    print("="*50 + "\n")


def compute_class_weights(labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute class weights for imbalanced dataset

    Args:
        labels: Binary labels (0 or 1)

    Returns:
        (weight_class_0, weight_class_1)
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (len(unique) * count)

    return weights.get(0, 1.0), weights.get(1, 1.0)
