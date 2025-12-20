"""
Training Pipeline for Galaxy Morphology Classification

Features:
- Multi-GPU support
- Weights & Biases logging
- Early stopping
- Learning rate scheduling
- Class imbalance handling
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import wandb

from src.models.resnet import create_model
from src.models.baseline import SimpleCNN
from src.training.evaluate import evaluate_model, compute_all_metrics


class Trainer:
    """
    Trainer class for galaxy morphology classification.
    
    Supports:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Early stopping
    - Checkpoint saving
    - WandB logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup loss function
        if config.get("class_weights"):
            weights = torch.tensor(config["class_weights"]).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        elif config.get("focal_loss", False):
            self.criterion = FocalLoss(gamma=config.get("focal_gamma", 2.0))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.get("use_amp", True) and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = config.get("use_wandb", True)
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "galaxy-morphology"),
                config=config,
                name=config.get("run_name", None),
            )
            wandb.watch(model)
        
        self.current_epoch = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        opt_name = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if opt_name.lower() == "adam":
            return optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif opt_name.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif opt_name.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine")
        epochs = self.config.get("epochs", 50)
        
        if scheduler_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=epochs,
                eta_min=self.config.get("min_lr", 1e-6),
            )
        elif scheduler_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "acc": 100. * correct / total,
            })
        
        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc": 100. * correct / total,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_acc": 100. * correct / total,
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history dictionary
        """
        epochs = self.config.get("epochs", 50)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                history[key].append(value)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    **train_metrics,
                    **val_metrics,
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                
                # Save best model
                self.save_checkpoint("best.pt")
                print(f"✓ New best model saved (val_loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"train_loss={train_metrics['train_loss']:.4f}, "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_acc={val_metrics['val_acc']:.2f}%")
        
        # Save final model
        self.save_checkpoint("final.pt")
        
        if self.use_wandb:
            wandb.finish()
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha[targets]
            focal_loss = alpha * focal_loss
        
        return focal_loss.mean()


def train_from_config(config_path: str) -> Dict[str, Any]:
    """
    Train model from a YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Training history
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(
        model_type=config.get("model", "resnet18"),
        num_classes=config.get("num_classes", 2),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.5),
    )
    
    # Create dataloaders (assumes they're created elsewhere)
    from src.data.dataset import create_dataloaders
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config["data_dir"],
        labels_file=config["labels_file"],
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        img_size=config.get("img_size", 224),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    # Train
    history = trainer.train()
    
    return history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    train_from_config(args.config)
