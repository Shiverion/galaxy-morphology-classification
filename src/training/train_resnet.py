"""
Train ResNet18 main model for galaxy morphology classification.
Uses transfer learning from ImageNet pretrained weights.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataset import GalaxyZooDataset, get_transforms
from src.models.resnet import create_model


def train_resnet(
    data_dir: str = "data/raw/images_training_rev1",
    labels_file: str = "data/splits/labels_with_splits.csv",
    model_type: str = "resnet18",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs/models",
):
    """Train ResNet main model with transfer learning."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Training {model_type} for {epochs} epochs")
    
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
    
    # Model with transfer learning
    model = create_model(
        model_type=model_type,
        num_classes=2,
        pretrained=True,
        dropout=0.5,
    ).to(device)
    
    print(f"Model: {model_type}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
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
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': model_type,
            }, output_dir / f"{model_type}_best.pt")
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'model_type': model_type,
    }, output_dir / f"{model_type}_final.pt")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_type} Training Loss")
    ax1.legend()
    
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_type} Training Accuracy")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(Path("outputs/figures") / f"{model_type}_training.png", dpi=150)
    print(f"\nSaved training curves to outputs/figures/{model_type}_training.png")
    
    print(f"\n{'='*50}")
    print(f"RESNET TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Model: {model_type}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return history, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", 
                       choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_resnet(model_type=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
