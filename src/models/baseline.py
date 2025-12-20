"""
Baseline Models for Galaxy Morphology Classification

Includes:
1. Traditional morphological parameters (CAS, Gini-M20)
2. Simple CNN baseline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN baseline for galaxy morphology classification.
    
    Architecture:
    - 4 convolutional blocks with BatchNorm and MaxPool
    - Global average pooling
    - Single fully connected layer for classification
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def compute_concentration(image: np.ndarray, r_80: float = 0.8, r_20: float = 0.2) -> float:
    """
    Compute Concentration Index (C) following Conselice (2003).
    
    C = 5 * log10(r_80 / r_20)
    where r_80 and r_20 are radii containing 80% and 20% of total light.
    
    Args:
        image: 2D numpy array (grayscale) or 3D (convert to grayscale)
        r_80: Fraction of light (default 0.8)
        r_20: Fraction of light (default 0.2)
        
    Returns:
        Concentration index value
    """
    if image.ndim == 3:
        # Convert to grayscale
        image = np.mean(image, axis=-1) if image.shape[-1] == 3 else image[0]
    
    # Find center (centroid)
    total_flux = np.sum(image)
    y_indices, x_indices = np.indices(image.shape)
    cx = np.sum(x_indices * image) / total_flux
    cy = np.sum(y_indices * image) / total_flux
    
    # Calculate radii from center
    radii = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    
    # Sort pixels by radius
    sorted_indices = np.argsort(radii.ravel())
    sorted_flux = image.ravel()[sorted_indices]
    sorted_radii = radii.ravel()[sorted_indices]
    
    # Cumulative flux
    cumsum = np.cumsum(sorted_flux)
    cumsum_normalized = cumsum / total_flux
    
    # Find radii containing r_80 and r_20 of light
    idx_20 = np.searchsorted(cumsum_normalized, r_20)
    idx_80 = np.searchsorted(cumsum_normalized, r_80)
    
    r_petro_20 = sorted_radii[min(idx_20, len(sorted_radii) - 1)]
    r_petro_80 = sorted_radii[min(idx_80, len(sorted_radii) - 1)]
    
    if r_petro_20 <= 0:
        return 0.0
    
    return 5 * np.log10(r_petro_80 / r_petro_20)


def compute_asymmetry(image: np.ndarray) -> float:
    """
    Compute Asymmetry index (A) following Conselice (2003).
    
    A = sum(|I - I_180|) / (2 * sum(|I|)) - A_bg
    where I_180 is the image rotated 180 degrees.
    
    Args:
        image: 2D numpy array
        
    Returns:
        Asymmetry value (0 = symmetric, higher = more asymmetric)
    """
    if image.ndim == 3:
        image = np.mean(image, axis=-1) if image.shape[-1] == 3 else image[0]
    
    # Rotate 180 degrees
    rotated = np.rot90(np.rot90(image))
    
    # Calculate asymmetry
    diff = np.abs(image - rotated)
    asymmetry = np.sum(diff) / (2 * np.sum(np.abs(image)) + 1e-10)
    
    return asymmetry


def compute_smoothness(
    image: np.ndarray, 
    smoothing_scale: int = 5
) -> float:
    """
    Compute Smoothness/Clumpiness (S) following Conselice (2003).
    
    S = sum(|I - I_s|) / sum(|I|) - S_bg
    where I_s is the smoothed image.
    
    Args:
        image: 2D numpy array
        smoothing_scale: Size of smoothing kernel
        
    Returns:
        Smoothness value (0 = smooth, higher = clumpy)
    """
    from scipy.ndimage import uniform_filter
    
    if image.ndim == 3:
        image = np.mean(image, axis=-1) if image.shape[-1] == 3 else image[0]
    
    # Smooth image
    smoothed = uniform_filter(image, size=smoothing_scale)
    
    # Calculate clumpiness
    diff = np.abs(image - smoothed)
    # Only count positive residuals (clumps)
    residual = np.clip(image - smoothed, 0, None)
    smoothness = np.sum(residual) / (np.sum(np.abs(image)) + 1e-10)
    
    return smoothness


def compute_gini(image: np.ndarray) -> float:
    """
    Compute Gini coefficient for light distribution.
    
    G = 1 / (mean * n * (n-1)) * sum((2i - n - 1) * x_i)
    where x_i are sorted pixel values.
    
    Args:
        image: 2D numpy array
        
    Returns:
        Gini coefficient (0 = uniform, 1 = concentrated)
    """
    if image.ndim == 3:
        image = np.mean(image, axis=-1) if image.shape[-1] == 3 else image[0]
    
    flat = image.ravel()
    # Only positive values
    flat = flat[flat > 0]
    
    if len(flat) == 0:
        return 0.0
    
    sorted_vals = np.sort(flat)
    n = len(sorted_vals)
    
    # Gini coefficient calculation
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n
    
    return gini


def compute_m20(image: np.ndarray) -> float:
    """
    Compute M20 statistic (second-order moment of brightest 20% of pixels).
    
    M20 = log10(sum(M_i for brightest 20%) / M_total)
    where M_i = flux_i * r_i^2
    
    Args:
        image: 2D numpy array
        
    Returns:
        M20 value
    """
    if image.ndim == 3:
        image = np.mean(image, axis=-1) if image.shape[-1] == 3 else image[0]
    
    # Find centroid
    total_flux = np.sum(image)
    if total_flux <= 0:
        return 0.0
        
    y_indices, x_indices = np.indices(image.shape)
    cx = np.sum(x_indices * image) / total_flux
    cy = np.sum(y_indices * image) / total_flux
    
    # Second-order moments
    r_squared = (x_indices - cx)**2 + (y_indices - cy)**2
    moments = image * r_squared
    
    # Total moment
    m_total = np.sum(moments)
    
    if m_total <= 0:
        return 0.0
    
    # Sort pixels by flux (descending)
    flat_flux = image.ravel()
    flat_moments = moments.ravel()
    sorted_indices = np.argsort(flat_flux)[::-1]
    
    # Find brightest 20%
    cumsum = np.cumsum(flat_flux[sorted_indices])
    threshold = 0.2 * total_flux
    n_brightest = np.searchsorted(cumsum, threshold) + 1
    
    # Sum moments of brightest pixels
    m_brightest = np.sum(flat_moments[sorted_indices[:n_brightest]])
    
    m20 = np.log10(m_brightest / m_total + 1e-10)
    
    return m20


def compute_morphological_params(image: np.ndarray) -> dict:
    """
    Compute all CAS and Gini-M20 parameters for an image.
    
    Args:
        image: Input image (2D or 3D numpy array)
        
    Returns:
        Dictionary with C, A, S, Gini, M20 values
    """
    return {
        "concentration": compute_concentration(image),
        "asymmetry": compute_asymmetry(image),
        "smoothness": compute_smoothness(image),
        "gini": compute_gini(image),
        "m20": compute_m20(image),
    }
