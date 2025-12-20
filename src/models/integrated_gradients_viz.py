"""
Generate Integrated Gradients visualizations for galaxy classification.
Pixel-level attribution showing which pixels contribute to the prediction.
"""

import argparse
from pathlib import Path
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz

from src.data.dataset import GalaxyZooDataset, get_transforms
from src.models.resnet import create_model


def generate_integrated_gradients(
    model_path: str = "outputs/models/resnet18_best.pt",
    data_dir: str = "data/raw/images_training_rev1",
    labels_file: str = "data/splits/labels_with_splits.csv",
    model_type: str = "resnet18",
    num_samples: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "outputs/figures",
):
    """Generate Integrated Gradients visualizations."""
    
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
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    # Load test data
    transform = get_transforms("val", img_size=224)
    dataset = GalaxyZooDataset(
        images_dir=data_dir,
        labels_file=labels_file,
        transform=transform,
        split="test",
    )
    
    # Sample galaxies
    elliptical_indices = [i for i, row in dataset.labels_df.iterrows() if row["label"] == 0]
    spiral_indices = [i for i, row in dataset.labels_df.iterrows() if row["label"] == 1]
    
    random.seed(42)
    sample_indices = (
        random.sample(elliptical_indices[:50], num_samples//2) +
        random.sample(spiral_indices[:50], num_samples//2)
    )
    
    class_names = ["Elliptical", "Spiral"]
    
    # Generate visualizations
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        # Get image and label
        image_tensor, label = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0, pred].item()
        
        # Compute Integrated Gradients (reduced steps for 4GB GPU)
        attributions = nt.attribute(
            image_tensor,
            nt_samples=5,
            nt_type="smoothgrad_sq",
            target=pred,
            n_steps=20,
        )
        
        # Get original image
        row = dataset.labels_df.iloc[idx]
        original_img = Image.open(Path(data_dir) / f"{row['GalaxyID']}.jpg")
        original_img = original_img.resize((224, 224))
        
        # Convert attributions for visualization
        attributions_np = attributions.squeeze().cpu().detach().numpy()
        attributions_np = np.transpose(attributions_np, (1, 2, 0))
        
        # Sum across channels for grayscale heatmap
        attr_sum = np.sum(np.abs(attributions_np), axis=2)
        attr_sum = (attr_sum - attr_sum.min()) / (attr_sum.max() - attr_sum.min() + 1e-8)
        
        # Plot
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Original\nTrue: {class_names[label]}", fontsize=10)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(attr_sum, cmap="hot")
        axes[i, 1].set_title(f"Integrated Gradients\nPred: {class_names[pred]} ({prob*100:.1f}%)", fontsize=10)
        axes[i, 1].axis("off")
        
        # Overlay
        overlay = np.array(original_img).astype(np.float32) / 255.0
        attr_colored = plt.cm.jet(attr_sum)[:, :, :3]
        blended = 0.6 * overlay + 0.4 * attr_colored
        blended = np.clip(blended, 0, 1)
        
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title("Overlay", fontsize=10)
        axes[i, 2].axis("off")
    
    plt.suptitle("Integrated Gradients: Pixel-Level Attribution", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / "integrated_gradients.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved Integrated Gradients visualization to {output_path}")
    
    plt.close()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/resnet18_best.pt")
    parser.add_argument("--model-type", type=str, default="resnet18")
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()
    
    generate_integrated_gradients(
        model_path=args.model,
        model_type=args.model_type,
        num_samples=args.num_samples,
    )
