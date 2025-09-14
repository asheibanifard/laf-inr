#!/usr/bin/env python3
import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio: Standard image quality metric.
    
    🎯 ANALOGY: Like measuring how clear a phone call is:
    - Higher dB = clearer call (better image quality)
    - 20dB = barely acceptable, 30dB = good, 40dB = excellent
    - PSNR measures signal (true image) vs noise (reconstruction errors)
    
    Args:
        pred: torch.Tensor [...] - Predicted image values
        target: torch.Tensor [...] - Ground truth image values (same shape as pred)
        data_range (float): Maximum possible pixel value (1.0 for normalized images)
    
    Returns:
        torch.Tensor scalar - PSNR in decibels (dB)
        
    Formula: PSNR = 20 * log10(data_range) - 10 * log10(MSE)
    Higher values indicate better image quality.
    """
    # Calculate Mean Squared Error between prediction and target
    # clamp_min(1e-12) prevents division by zero in log calculation
    mse = F.mse_loss(pred, target, reduction="mean").clamp_min(1e-12)
    
    # Convert data_range to tensor on same device as predictions for computation
    data_range_tensor = torch.tensor(data_range, device=pred.device)
    
    # Calculate PSNR using the standard formula:
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    # where MAX is the maximum possible pixel value
    return 20.0 * torch.log10(data_range_tensor) - 10.0 * torch.log10(mse)