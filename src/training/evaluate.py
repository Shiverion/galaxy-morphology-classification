"""
Evaluation metrics and utilities for galaxy morphology classification.

Implements comprehensive metrics following the project plan:
- Standard classification metrics (accuracy, F1, etc.)
- Calibration assessment (ECE, reliability diagrams)
- Class-wise analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    cohen_kappa_score,
    classification_report,
)
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_classes: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Evaluate model and collect predictions.
    
    Returns:
        Dictionary with predictions, probabilities, and labels
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
        "labels": np.array(all_labels),
    }


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str] = ["Elliptical", "Spiral"],
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Predicted class indices
        labels: True class indices
        probabilities: Prediction probabilities
        class_names: Names of classes
        
    Returns:
        Dictionary of metric names and values
    """
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None
    )
    
    # Weighted average metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    
    # Macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    
    # Cohen's kappa
    kappa = cohen_kappa_score(labels, predictions)
    
    # ROC-AUC (one-vs-rest)
    try:
        if len(class_names) == 2:
            roc_auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            roc_auc = roc_auc_score(labels, probabilities, multi_class="ovr")
    except ValueError:
        roc_auc = None
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    
    # ECE (Expected Calibration Error)
    ece = compute_ece(probabilities, predictions, labels)
    
    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "cohens_kappa": kappa,
        "roc_auc": roc_auc,
        "ece": ece,
        "confusion_matrix": conf_matrix,
    }
    
    # Per-class metrics
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = precision[i]
        metrics[f"recall_{name}"] = recall[i]
        metrics[f"f1_{name}"] = f1[i]
        metrics[f"support_{name}"] = support[i]
    
    return metrics


def compute_ece(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.
    
    ECE = sum(|B_m| / n * |acc(B_m) - conf(B_m)|)
    
    Args:
        probabilities: Prediction probabilities
        predictions: Predicted classes
        labels: True labels
        n_bins: Number of confidence bins
        
    Returns:
        ECE value
    """
    # Get confidence for predicted class
    confidences = np.max(probabilities, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: Class names
        save_path: Optional path to save figure
        normalize: Whether to normalize rows
        
    Returns:
        Matplotlib figure
    """
    if normalize:
        conf_matrix = conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(conf_matrix, cmap="Blues")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = conf_matrix.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                   ha="center", va="center",
                   color="white" if conf_matrix[i, j] > thresh else "black")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def plot_reliability_diagram(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot reliability diagram for calibration assessment.
    
    Args:
        probabilities: Prediction probabilities
        predictions: Predicted classes
        labels: True labels
        n_bins: Number of confidence bins
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    confidences = np.max(probabilities, axis=1)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            confidence = confidences[in_bin].mean()
            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7, 
           edgecolor="black", label="Outputs")
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, range=(0, 1), edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Histogram")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def generate_evaluation_report(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str] = ["Elliptical", "Spiral"],
    save_dir: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: Class names
        save_dir: Directory to save figures and report
        device: Computation device
        
    Returns:
        Dictionary with all metrics and paths to saved figures
    """
    # Get predictions
    results = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_all_metrics(
        results["predictions"],
        results["labels"],
        results["probabilities"],
        class_names,
    )
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            save_path=save_dir / "confusion_matrix.png",
        )
        
        # Save reliability diagram
        plot_reliability_diagram(
            results["probabilities"],
            results["predictions"],
            results["labels"],
            save_path=save_dir / "reliability_diagram.png",
        )
        
        # Save classification report
        report = classification_report(
            results["labels"],
            results["predictions"],
            target_names=class_names,
        )
        with open(save_dir / "classification_report.txt", "w") as f:
            f.write(f"Galaxy Morphology Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"Cohen's Kappa: {metrics['cohens_kappa']:.4f}\n")
            f.write(f"ECE: {metrics['ece']:.4f}\n")
    
    return metrics
