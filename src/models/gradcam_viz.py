"""
Generate Grad-CAM visualizations for galaxy classification.
Shows which regions of the galaxy the model focuses on.
"""

import argparse
from pathlib import Path
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.dataset import GalaxyZooDataset, get_transforms
from src.models.resnet import create_model


def generate_gradcam_visualizations(
    model_path: str = "outputs/models/resnet18_best.pt",
    data_dir: str = "data/raw/images_training_rev1",
    labels_file: str = "data/splits/labels_with_splits.csv",
    model_type: str = "resnet18",
    num_samples: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs/figures",
):
    """Generate Grad-CAM visualizations for sample galaxies."""
    
    print(f"Device: {device}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = create_model(model_type=model_type, num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Get target layer for Grad-CAM (last conv layer)
    if "resnet" in model_type:
        target_layers = [model.model.layer4[-1]]
    elif "efficientnet" in model_type:
        target_layers = [model.model.conv_head]
    else:
        target_layers = [model.model.blocks[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Load test data
    transform = get_transforms("val", img_size=224)
    dataset = GalaxyZooDataset(
        images_dir=data_dir,
        labels_file=labels_file,
        transform=transform,
        split="test",
    )
    
    # Sample galaxies from each class
    elliptical_indices = [i for i, row in dataset.labels_df.iterrows() if row["label"] == 0]
    spiral_indices = [i for i, row in dataset.labels_df.iterrows() if row["label"] == 1]
    
    random.seed(42)
    sample_elliptical = random.sample(elliptical_indices[:100], min(num_samples//2, len(elliptical_indices)))
    sample_spiral = random.sample(spiral_indices[:100], min(num_samples//2, len(spiral_indices)))
    sample_indices = sample_elliptical + sample_spiral
    
    # Generate visualizations
    fig, axes = plt.subplots(4, num_samples//2, figsize=(3 * num_samples//2, 12))
    
    class_names = ["Elliptical", "Spiral"]
    
    for i, idx in enumerate(sample_indices):
        col = i % (num_samples // 2)
        is_spiral = i >= num_samples // 2
        
        # Get image and label
        image_tensor, label = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0, pred].item()
        
        # Generate Grad-CAM
        targets = [ClassifierOutputTarget(pred)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Get original image for overlay
        row = dataset.labels_df.iloc[idx]
        original_img = Image.open(Path(data_dir) / f"{row['GalaxyID']}.jpg")
        original_img = original_img.resize((224, 224))
        rgb_img = np.array(original_img).astype(np.float32) / 255.0
        
        # Create overlay
        cam_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Plot
        row_offset = 2 if is_spiral else 0
        
        # Original image
        axes[row_offset, col].imshow(original_img)
        axes[row_offset, col].set_title(f"True: {class_names[label]}", fontsize=10)
        axes[row_offset, col].axis("off")
        
        # Grad-CAM overlay
        axes[row_offset + 1, col].imshow(cam_overlay)
        pred_text = f"Pred: {class_names[pred]} ({prob*100:.1f}%)"
        axes[row_offset + 1, col].set_title(pred_text, fontsize=10, 
                                            color="green" if pred == label else "red")
        axes[row_offset + 1, col].axis("off")
    
    # Row labels
    fig.text(0.02, 0.85, "Elliptical\nOriginal", fontsize=12, fontweight="bold", va="center")
    fig.text(0.02, 0.65, "Elliptical\nGrad-CAM", fontsize=12, fontweight="bold", va="center")
    fig.text(0.02, 0.35, "Spiral\nOriginal", fontsize=12, fontweight="bold", va="center")
    fig.text(0.02, 0.15, "Spiral\nGrad-CAM", fontsize=12, fontweight="bold", va="center")
    
    plt.suptitle("Grad-CAM Visualization: Where Does the Model Look?", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    
    output_path = output_dir / "gradcam_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved Grad-CAM visualization to {output_path}")
    
    # Generate summary heatmap
    print("\n--- Grad-CAM Analysis ---")
    print("The heatmaps show which regions the model focuses on:")
    print("- Red/Yellow: High attention areas")
    print("- Blue/Green: Low attention areas")
    print("\nFor galaxies:")
    print("- Ellipticals: Model should focus on smooth central bulge")
    print("- Spirals: Model should focus on spiral arms and disk structure")
    
    plt.close()
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/resnet18_best.pt")
    parser.add_argument("--model-type", type=str, default="resnet18")
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    
    generate_gradcam_visualizations(
        model_path=args.model,
        model_type=args.model_type,
        num_samples=args.num_samples,
    )
