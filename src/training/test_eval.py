"""
Evaluate trained model on test set.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data.dataset import GalaxyZooDataset, get_transforms
from src.models.resnet import create_model


def evaluate_on_test(
    model_path: str,
    data_dir: str = "data/raw/images_training_rev1",
    labels_file: str = "data/splits/labels_with_splits.csv",
    model_type: str = "resnet18",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate model on test set."""
    
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    
    # Load test data
    test_transform = get_transforms("val", img_size=224)
    test_dataset = GalaxyZooDataset(
        images_dir=data_dir,
        labels_file=labels_file,
        transform=test_transform,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Load model
    model = create_model(model_type=model_type, num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of spiral
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = (all_preds == all_labels).mean() * 100
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, 
                                target_names=["Elliptical", "Spiral"]))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Elliptical", "Spiral"],
                yticklabels=["Elliptical", "Spiral"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix")
    
    # Plot prediction distribution
    axes[1].hist(all_probs[all_labels == 0], bins=50, alpha=0.5, label="Elliptical", color="blue")
    axes[1].hist(all_probs[all_labels == 1], bins=50, alpha=0.5, label="Spiral", color="orange")
    axes[1].set_xlabel("P(Spiral)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Probability Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    output_path = Path("outputs/figures") / f"{model_type}_test_eval.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved evaluation plots to {output_path}")
    
    return accuracy, roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/resnet18_best.pt")
    parser.add_argument("--model-type", type=str, default="resnet18")
    args = parser.parse_args()
    
    evaluate_on_test(model_path=args.model, model_type=args.model_type)
