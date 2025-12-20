"""
Interpretability Module for Galaxy Morphology Classification

Implements:
- Grad-CAM for activation visualization
- Integrated Gradients for pixel-level attribution
- Saliency map generation and analysis
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Grad-CAM
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Captum for Integrated Gradients
from captum.attr import (
    IntegratedGradients,
    LayerGradCam,
    NoiseTunnel,
    visualization as viz,
)


class GalaxyInterpretability:
    """
    Interpretability toolkit for galaxy morphology classifiers.
    
    Provides methods for:
    - Grad-CAM activation maps
    - Integrated Gradients attribution
    - Saliency map generation
    - Visualization utilities
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: Trained classification model
            target_layer: Layer for Grad-CAM. If None, attempts auto-detection.
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Set target layer for Grad-CAM
        if target_layer is None:
            self.target_layer = self._get_target_layer()
        else:
            self.target_layer = target_layer
    
    def _get_target_layer(self) -> nn.Module:
        """Auto-detect the target layer for Grad-CAM."""
        # Try common attribute names
        if hasattr(self.model, 'get_final_conv_layer'):
            return self.model.get_final_conv_layer()
        
        # Try to find last conv layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("Could not auto-detect target layer for Grad-CAM")
        
        return last_conv
    
    def grad_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "gradcam",
    ) -> np.ndarray:
        """
        Generate Grad-CAM activation map.
        
        Args:
            image: Input tensor [1, C, H, W]
            target_class: Target class for CAM. None = predicted class.
            method: One of "gradcam", "gradcam++", "scorecam"
            
        Returns:
            Grayscale CAM array [H, W] with values in [0, 1]
        """
        image = image.to(self.device)
        
        # Select CAM method
        if method == "gradcam":
            cam_class = GradCAM
        elif method == "gradcam++":
            cam_class = GradCAMPlusPlus
        elif method == "scorecam":
            cam_class = ScoreCAM
        else:
            raise ValueError(f"Unknown method: {method}")
        
        cam = cam_class(model=self.model, target_layers=[self.target_layer])
        
        # Set target
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        
        # Generate CAM
        grayscale_cam = cam(input_tensor=image, targets=targets)
        
        return grayscale_cam[0]  # Remove batch dimension
    
    def integrated_gradients(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        noise_tunnel: bool = False,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            image: Input tensor [1, C, H, W]
            target_class: Target class. None = predicted class.
            n_steps: Number of interpolation steps
            noise_tunnel: Apply noise tunnel for smoother attribution
            
        Returns:
            Attribution array [C, H, W]
        """
        image = image.to(self.device)
        image.requires_grad = True
        
        ig = IntegratedGradients(self.model)
        
        if noise_tunnel:
            ig = NoiseTunnel(ig)
        
        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Compute attribution
        if noise_tunnel:
            attributions = ig.attribute(
                image,
                target=target_class,
                nt_samples=10,
                nt_type='smoothgrad_sq',
            )
        else:
            attributions = ig.attribute(
                image,
                target=target_class,
                n_steps=n_steps,
            )
        
        return attributions.squeeze().cpu().detach().numpy()
    
    def visualize_cam(
        self,
        image: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image tensor [1, C, H, W] (normalized)
            cam: Grayscale CAM array [H, W]
            alpha: Opacity of overlay
            colormap: Matplotlib colormap name
            
        Returns:
            RGB image array with CAM overlay [H, W, 3]
        """
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img_np = image.squeeze().cpu().numpy()  # [C, H, W]
        img_np = img_np.transpose(1, 2, 0)  # [H, W, C]
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Apply colormap to CAM
        visualization = show_cam_on_image(img_np, cam, use_rgb=True)
        
        return visualization
    
    def generate_saliency_report(
        self,
        image: torch.Tensor,
        save_dir: Optional[Path] = None,
        image_id: str = "galaxy",
    ) -> dict:
        """
        Generate comprehensive saliency report for a single image.
        
        Args:
            image: Input tensor [1, C, H, W]
            save_dir: Directory to save visualizations
            image_id: Identifier for the image
            
        Returns:
            Dictionary with CAM arrays and predictions
        """
        image = image.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate different CAMs
        gradcam = self.grad_cam(image, pred_class, method="gradcam")
        gradcam_pp = self.grad_cam(image, pred_class, method="gradcam++")
        ig = self.integrated_gradients(image, pred_class)
        
        report = {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": probs.cpu().numpy(),
            "gradcam": gradcam,
            "gradcam_pp": gradcam_pp,
            "integrated_gradients": ig,
        }
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save visualizations
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original image
            img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title(f"Original (Pred: {pred_class}, Conf: {confidence:.2f})")
            axes[0, 0].axis("off")
            
            # Grad-CAM
            axes[0, 1].imshow(self.visualize_cam(image, gradcam))
            axes[0, 1].set_title("Grad-CAM")
            axes[0, 1].axis("off")
            
            # Grad-CAM++
            axes[1, 0].imshow(self.visualize_cam(image, gradcam_pp))
            axes[1, 0].set_title("Grad-CAM++")
            axes[1, 0].axis("off")
            
            # Integrated Gradients
            ig_vis = np.abs(ig).sum(axis=0)  # Sum across channels
            ig_vis = (ig_vis - ig_vis.min()) / (ig_vis.max() - ig_vis.min() + 1e-8)
            axes[1, 1].imshow(ig_vis, cmap="hot")
            axes[1, 1].set_title("Integrated Gradients")
            axes[1, 1].axis("off")
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{image_id}_saliency.png", dpi=150)
            plt.close()
        
        return report


def analyze_class_attention_patterns(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """
    Analyze attention patterns across different classes.
    
    Args:
        model: Trained model
        dataloader: DataLoader with labeled samples
        num_samples: Number of samples to analyze per class
        device: Computation device
        
    Returns:
        Dictionary with per-class aggregated CAM statistics
    """
    interpreter = GalaxyInterpretability(model, device=device)
    
    class_cams = {}  # {class_id: [cam1, cam2, ...]}
    counts = {}
    
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        
        for i in range(len(images)):
            label = labels[i].item()
            
            if label not in class_cams:
                class_cams[label] = []
                counts[label] = 0
            
            if counts[label] >= num_samples:
                continue
            
            cam = interpreter.grad_cam(images[i:i+1], target_class=label)
            class_cams[label].append(cam)
            counts[label] += 1
        
        # Check if we have enough samples
        if all(c >= num_samples for c in counts.values()):
            break
    
    # Compute statistics
    results = {}
    for class_id, cams in class_cams.items():
        cams_array = np.stack(cams)
        results[class_id] = {
            "mean_cam": np.mean(cams_array, axis=0),
            "std_cam": np.std(cams_array, axis=0),
            "n_samples": len(cams),
        }
    
    return results
