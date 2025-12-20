"""
Train SimpleCNN baseline model for galaxy morphology classification.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataset import GalaxyZooDataset, get_transforms
from src.models.baseline import SimpleCNN


def train_baseline(
    data_dir: str = "data/raw/images_training_rev1",
    labels_file: str = "data/splits/labels_with_splits.csv",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs/models",
):
    """Train SimpleCNN baseline model."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Training SimpleCNN baseline for {epochs} epochs")
    
    # Data loaders
    train_transform = get_transforms("train", img_size=224)
    val_transform = get_transforms("val", img_size=224)
    
    train_dataset = GalaxyZooDataset(
        images_dir=data_dir,
        labels_file=labels_file,
        transform=train_transform,
        split="train",
    )
    val_dataset = GalaxyZooDataset(
        images_dir=data_dir,
        labels_file=labels_file,
        transform=val_transform,
        split="val",
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Model
    model = SimpleCNN(num_classes=2, dropout=0.5).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                "loss": train_loss / (pbar.n + 1),
                "acc": 100. * train_correct / train_total,
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "baseline_best.pt")
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / "baseline_final.pt")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path("outputs/figures") / "baseline_training.png", dpi=150)
    print(f"\nSaved training curves to outputs/figures/baseline_training.png")
    
    print(f"\n{'='*50}")
    print(f"BASELINE TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return history, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_baseline(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
