"""
ResNet-based Galaxy Morphology Classifier with Transfer Learning

Uses timm for pretrained models with easy customization.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple, List


class GalaxyResNet(nn.Module):
    """
    ResNet-based classifier for galaxy morphology.
    
    Uses ImageNet pretrained weights with fine-tuning options:
    - Full fine-tuning
    - Feature extraction (frozen backbone)
    - Gradual unfreezing
    """
    
    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            model_name: timm model name (resnet18, resnet34, resnet50, efficientnet_b0, etc.)
            num_classes: Number of output classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate before classifier
            freeze_backbone: If True, freeze all layers except classifier
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create model from timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for name, param in self.model.named_parameters():
            if "fc" not in name and "classifier" not in name and "head" not in name:
                param.requires_grad = False
    
    def unfreeze_layers(self, num_layers: int = -1):
        """
        Gradually unfreeze layers for fine-tuning.
        
        Args:
            num_layers: Number of final layers to unfreeze. -1 for all.
        """
        params = list(self.model.named_parameters())
        
        if num_layers == -1:
            for name, param in params:
                param.requires_grad = True
        else:
            # Unfreeze last n layers
            for name, param in params[-num_layers:]:
                param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the model (before classifier).
        Useful for interpretability and visualization.
        """
        return self.model.forward_features(x)
    
    def get_final_conv_layer(self) -> nn.Module:
        """
        Get the final convolutional layer for Grad-CAM.
        """
        if "resnet" in self.model_name:
            return self.model.layer4[-1]
        elif "efficientnet" in self.model_name:
            return self.model.conv_head
        elif "vit" in self.model_name:
            # For ViT, return the final norm layer
            return self.model.norm
        else:
            # Generic fallback - try to find last conv layer
            modules = list(self.model.modules())
            for m in reversed(modules):
                if isinstance(m, nn.Conv2d):
                    return m
            raise ValueError(f"Could not find conv layer for {self.model_name}")


class GalaxyEfficientNet(GalaxyResNet):
    """EfficientNet variant for galaxy classification."""
    
    def __init__(
        self,
        variant: str = "b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__(
            model_name=f"efficientnet_{variant}",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )


class GalaxyViT(nn.Module):
    """
    Vision Transformer for galaxy classification.
    
    Provides native attention mechanisms useful for interpretability.
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps from all transformer blocks.
        Useful for interpretability analysis.
        
        Returns:
            List of attention tensors, one per block
        """
        attention_maps = []
        
        # Hook to capture attention
        def get_attention(module, input, output):
            if hasattr(module, 'attn_drop'):
                # Get attention weights before dropout
                attention_maps.append(output.detach())
        
        hooks = []
        for block in self.model.blocks:
            hook = block.attn.register_forward_hook(get_attention)
            hooks.append(hook)
        
        # Forward pass
        _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps


def create_model(
    model_type: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create galaxy classification models.
    
    Args:
        model_type: One of "resnet18", "resnet34", "resnet50", 
                    "efficientnet_b0", "vit_small", or any timm model
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Configured model
    """
    if model_type.startswith("vit"):
        return GalaxyViT(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        return GalaxyResNet(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
