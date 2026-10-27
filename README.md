# Interpretable Deep Learning for Galaxy Morphology Classification

A CNN-based classifier for distinguishing galaxy morphologies (spiral vs elliptical) using Galaxy Zoo 2 data, with interpretability as a core objective.

## 🎯 Results

| Model | Test Accuracy | ROC-AUC |
|-------|---------------|---------|
| SimpleCNN (baseline) | 80.69% | - |
| **ResNet18** | **99.07%** | **0.9995** |

## Features

- **ResNet18** with ImageNet transfer learning
- **Interpretability**: Grad-CAM, Integrated Gradients
- **Binary classification**: Elliptical vs Spiral galaxies
- **High-confidence filtering**: Uses 80%+ vote consensus samples

## Project Structure

```
astroML/
├── data/
│   ├── raw/                  # Original images
│   └── splits/               # Train/val/test CSV
├── src/
│   ├── data/                 # Dataset and preprocessing
│   ├── models/               # CNN architectures + interpretability
│   └── training/             # Training and evaluation scripts
├── outputs/
│   ├── models/               # Saved checkpoints
│   └── figures/              # Visualizations
└── configs/                  # YAML configurations
```

## System Design

### Requirements

- Train a binary galaxy morphology classifier and compare evaluation metrics.
- Generate Grad-CAM and Integrated Gradients visualizations for model interpretation.

### Pipeline

```text
Galaxy Zoo images + label/split CSV → preprocessing and transforms
                                     → ResNet transfer-learning training
                                     → evaluation / attribution figures
```

The configuration points to `data/processed` and `data/labels.csv`, while training utilities also import `src.data.dataset`. Neither the data files nor a `src/data/` package is present in this checkout, so the full training path cannot be reproduced from the repository alone.

### Trade-offs

| Decision | Benefit | Cost |
| --- | --- | --- |
| Use pretrained ResNet-family models | Reuses learned visual features | Results depend on the external data and pretrained weights |
| Pair predictions with attribution tools | Makes image regions available for inspection | Heatmaps do not establish scientific validity on their own |

### What to Revisit

- Add the dataset/preprocessing module or document its separate source and expected files.
- Align README commands with the scripts that exist in `src/training/`.
- Record model checkpoints and data split provenance with reported metrics.

## Installation

```powershell
# Using UV
uv sync

# Install CUDA PyTorch (Windows)
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Quick Start

```powershell
# 1. Preprocess data
.\.venv\Scripts\python.exe -m src.data.preprocess

# 2. Train ResNet18
.\.venv\Scripts\python.exe -m src.training.train_resnet --epochs 20

# 3. Evaluate on test set
.\.venv\Scripts\python.exe -m src.training.test_eval

# 4. Generate Grad-CAM visualizations
.\.venv\Scripts\python.exe -m src.models.gradcam_viz
```

## Dataset

- **Source**: Galaxy Zoo 2 (SDSS images)
- **Total**: 61,578 labeled galaxies
- **Used**: 24,273 high-confidence samples (80%+ vote agreement)
- **Splits**: Train 16,991 / Val 3,640 / Test 3,642

## Visualizations

- `gradcam_visualization.png` - Model attention heatmaps
- `integrated_gradients.png` - Pixel-level attribution
- `resnet18_training.png` - Training curves
- `resnet18_test_eval.png` - Confusion matrix

## License

MIT
