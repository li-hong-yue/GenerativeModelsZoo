"""
Metrics utilities for evaluating generative models.

This module provides wrapper functions around torchmetrics for computing:
- FID (Fréchet Inception Distance)
- Inception Score
- Additional metrics like Precision/Recall

Usage:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # These are the main metrics we use, directly from torchmetrics
"""

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def compute_fid_score(real_images, fake_images, device='cuda', batch_size=64, feature=2048):
    """
    Compute FID score using torchmetrics.
    
    Args:
        real_images: Real images [N, C, H, W] in range [0, 1]
        fake_images: Generated images [N, C, H, W] in range [0, 1]
        device: Device to use
        batch_size: Batch size for processing
        feature: Feature layer to use (64, 192, 768, or 2048)
        
    Returns:
        float: FID score (lower is better)
        
    Example:
        >>> real = torch.rand(1000, 3, 32, 32)  # Real images
        >>> fake = torch.rand(1000, 3, 32, 32)  # Generated images
        >>> fid = compute_fid_score(real, fake)
        >>> print(f"FID: {fid:.2f}")
    """
    from tqdm import tqdm
    
    fid = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
    
    # Convert to uint8 [0, 255]
    real_images = (real_images * 255).to(torch.uint8)
    fake_images = (fake_images * 255).to(torch.uint8)
    
    # Update with real images
    for i in tqdm(range(0, len(real_images), batch_size), desc="FID: Real images"):
        batch = real_images[i:i+batch_size].to(device)
        fid.update(batch, real=True)
    
    # Update with fake images
    for i in tqdm(range(0, len(fake_images), batch_size), desc="FID: Fake images"):
        batch = fake_images[i:i+batch_size].to(device)
        fid.update(batch, real=False)
    
    return fid.compute().item()


def compute_inception_score(images, device='cuda', batch_size=64, splits=10):
    """
    Compute Inception Score using torchmetrics.
    
    Args:
        images: Generated images [N, C, H, W] in range [0, 1]
        device: Device to use
        batch_size: Batch size for processing
        splits: Number of splits for computing mean and std
        
    Returns:
        tuple: (mean, std) of Inception Score
        
    Example:
        >>> images = torch.rand(1000, 3, 32, 32)
        >>> is_mean, is_std = compute_inception_score(images)
        >>> print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
    """
    from tqdm import tqdm
    
    inception = InceptionScore(normalize=True, splits=splits).to(device)
    
    # Convert to uint8 [0, 255]
    images = (images * 255).to(torch.uint8)
    
    # Update with images
    for i in tqdm(range(0, len(images), batch_size), desc="Inception Score"):
        batch = images[i:i+batch_size].to(device)
        inception.update(batch)
    
    mean, std = inception.compute()
    return mean.item(), std.item()


# Re-export from torchmetrics for convenience
__all__ = [
    'FrechetInceptionDistance',
    'InceptionScore',
    'compute_fid_score',
    'compute_inception_score'
]