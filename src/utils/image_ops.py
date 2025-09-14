#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from typing import Tuple


def conv_laplacian(x: torch.Tensor) -> torch.Tensor:
    """
    Laplacian operator using 2D convolution for edge detection.
    
    🎯 ANALOGY: Like a texture/edge detector tool:
    - Imagine running your finger over a surface - smooth areas feel flat (low response)
    - Bumps, edges, textures feel different (high response)
    - Laplacian finds "how much this pixel differs from its neighbors"
    - Perfect for detecting fine details, edges, and textural patterns
    
    Args:
        x: torch.Tensor [B, C, H, W] - Input tensor
    
    Returns:
        torch.Tensor [B, C, H, W] - Laplacian filtered tensor
        
    Uses 3x3 kernel: [[0,1,0], [1,-4,1], [0,1,0]]
    Highlights areas of rapid intensity change (edges, textures).
    """
    # Define the 3x3 Laplacian kernel for edge detection
    # Center coefficient = -4, direct neighbors = +1, diagonals = 0
    # This kernel detects the second derivative (curvature) in the image
    k = torch.tensor([[0, 1, 0],      # Top row: only vertical neighbor
                      [1,-4, 1],      # Middle row: center pixel and horizontal neighbors  
                      [0, 1, 0]],     # Bottom row: only vertical neighbor
                      dtype=x.dtype, device=x.device).view(1,1,3,3)  # Reshape for conv2d: [out_ch, in_ch, H, W]
    
    # Apply 2D convolution with padding=1 to maintain input spatial dimensions
    # The Laplacian will highlight regions where pixel values change rapidly
    if x.shape[1] > 1:
        k_expanded = k.expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, k_expanded, padding=1, groups=x.shape[1])
    else:
        return F.conv2d(x, k, padding=1)


def grad_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute image gradients in x and y directions using forward differences.
    
    Args:
        x: torch.Tensor [B, C, H, W] - Input tensor
    
    Returns:
        dx: torch.Tensor [B, C, H, W] - Horizontal gradient (rightward differences)
        dy: torch.Tensor [B, C, H, W] - Vertical gradient (downward differences)
        
    Gradients are computed as forward differences with zero-padding to maintain size.
    Used for edge detection and texture analysis.
    """
    # Compute horizontal gradient (x-direction): difference between adjacent pixels horizontally
    # x[..., :, 1:] takes all pixels except the leftmost column
    # x[..., :, :-1] takes all pixels except the rightmost column
    # This gives us the difference: pixel[i,j+1] - pixel[i,j] (rightward difference)
    dx = x[..., :, 1:] - x[..., :, :-1]
    
    # Compute vertical gradient (y-direction): difference between adjacent pixels vertically  
    # x[..., 1:, :] takes all pixels except the top row
    # x[..., :-1, :] takes all pixels except the bottom row
    # This gives us the difference: pixel[i+1,j] - pixel[i,j] (downward difference)
    dy = x[..., 1:, :] - x[..., :-1, :]
    
    # Pad gradients to maintain original spatial dimensions
    # dx loses 1 column (W-1), so pad 1 column on the right: (left=0, right=1, top=0, bottom=0)
    dx = F.pad(dx, (1,0,0,0))
    
    # dy loses 1 row (H-1), so pad 1 row on the bottom: (left=0, right=0, top=0, bottom=1)  
    dy = F.pad(dy, (0,0,1,0))
    
    return dx, dy


def gaussian_kernel_2d(sigma: float, kernel_size: int = None, device=None):
    """Create 2D Gaussian kernel for filtering operations"""
    if kernel_size is None:
        kernel_size = int(2 * (2 * sigma) + 1)
    
    # Create coordinate grids
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    
    # Calculate Gaussian kernel values
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def high_pass_filter(x: torch.Tensor, sigma: float = 1.0):
    """Apply high-pass filter using Gaussian blur subtraction"""
    # Create Gaussian kernel
    kernel = gaussian_kernel_2d(sigma, device=x.device)
    kernel = kernel.view(1, 1, kernel.size(0), kernel.size(1))
    
    # Apply Gaussian blur
    if x.shape[1] > 1:
        kernel = kernel.expand(x.shape[1], 1, kernel.size(2), kernel.size(3))
        blurred = F.conv2d(x, kernel, padding=kernel.size(2)//2, groups=x.shape[1])
    else:
        blurred = F.conv2d(x, kernel, padding=kernel.size(2)//2)
    
    # High-pass = original - low-pass (blurred)
    return x - blurred