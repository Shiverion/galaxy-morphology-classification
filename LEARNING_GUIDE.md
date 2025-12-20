# Galaxy Morphology Deep Learning - Step-by-Step Learning Guide

A hands-on tutorial for building an interpretable CNN classifier for galaxy morphology.

## 📚 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Building Models](#4-building-models)
5. [Training Pipeline](#5-training-pipeline)
6. [Interpretability Analysis](#6-interpretability-analysis)
7. [Evaluation & Metrics](#7-evaluation--metrics)

---

## 1. Project Overview

**Goal**: Classify galaxies as **Elliptical** (smooth) or **Spiral** (featured) using CNNs.

**Why interpretability?** Understanding *why* the model makes predictions is crucial for scientific trust.

**Key concepts**:
- Transfer learning (ImageNet → Galaxy images)
- Grad-CAM for visualizing model attention
- Morphological parameters (CAS, Gini-M20)

---

## 2. Environment Setup

### Step 2.1: Create Project with UV

```powershell
# Create project directory
mkdir astroML
cd astroML

# Initialize UV project
uv init

# Add dependencies
uv add torch torchvision timm numpy pandas matplotlib
uv add astropy photutils captum grad-cam wandb jupyter
```

### Step 2.2: Enable GPU (CUDA)

```powershell
# Check if NVIDIA GPU available
nvidia-smi

# Install PyTorch with CUDA (requires ~2GB free space)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --reinstall
```

### Step 2.3: Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 3. Data Preparation

### Step 3.1: Understand Galaxy Zoo 2 Labels

Galaxy Zoo 2 uses **vote fractions** from citizen scientists:
- `Class1.1`: Smooth (elliptical) vote fraction
- `Class1.2`: Featured (spiral) vote fraction

**Example**: If 80% of voters say "smooth", the galaxy is likely elliptical.

### Step 3.2: Create Dataset Class

```python
# src/data/dataset.py
from torch.utils.data import Dataset
from PIL import Image

class GalaxyZooDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None, split="train"):
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Load labels and filter by split
        df = pd.read_csv(labels_file)
        self.labels_df = df[df["split"] == split]
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / f"{row['GalaxyID']}.jpg"
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label
```

### Step 3.3: Data Augmentation

Galaxies are **rotationally invariant** - they can be any orientation!

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(180),  # Key: rotate freely
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
```

---

## 4. Building Models

### Step 4.1: Simple CNN Baseline

Start simple to establish a baseline:

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

### Step 4.2: ResNet with Transfer Learning

Use pretrained weights for better performance:

```python
import timm

# Create ResNet18 pretrained on ImageNet
model = timm.create_model('resnet18', pretrained=True, num_classes=2)

# Optionally freeze early layers
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True  # Only train classifier
```

---

## 5. Training Pipeline

### Step 5.1: Training Loop

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100 * correct / total
```

### Step 5.2: Validation Loop

```python
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100 * correct / total
```

### Step 5.3: Full Training Script

```python
# Train for N epochs
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}: train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), "best_model.pt")
        best_val_acc = val_acc
```

---

## 6. Interpretability Analysis

### Step 6.1: Grad-CAM

See which parts of the image the model focuses on:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Target the last conv layer
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Generate heatmap
grayscale_cam = cam(input_tensor=image)
visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
```

### Step 6.2: Integrated Gradients

Pixel-level attribution:

```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attributions = ig.attribute(image, target=predicted_class, n_steps=50)
```

---

## 7. Evaluation & Metrics

### Key Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| F1-Score | Harmonic mean of precision/recall |
| ROC-AUC | Area under ROC curve |
| ECE | Calibration error |

### Generate Evaluation Report

```python
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions on test set
predictions, labels = [], []
for images, lbls in test_loader:
    outputs = model(images.to(device))
    preds = outputs.argmax(dim=1)
    predictions.extend(preds.cpu().numpy())
    labels.extend(lbls.numpy())

# Print report
print(classification_report(labels, predictions, 
                            target_names=["Elliptical", "Spiral"]))
```

---

## 🚀 Quick Commands Reference

```powershell
# Run EDA
uv run python -m src.data.eda

# Train baseline
uv run python -m src.training.train_baseline --epochs 10

# Train ResNet
uv run python -m src.training.train_resnet --epochs 20

# Run interpretability
uv run python -m src.models.interpretability
```

---

## 📖 Further Reading

- [Galaxy Zoo Paper](https://arxiv.org/abs/1705.02000)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
