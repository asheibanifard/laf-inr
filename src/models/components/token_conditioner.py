#!/usr/bin/env python3
"""
Token Conditioner: Computes global image statistics and ID embeddings for conditioning.

This module provides the TokenConditioner class which extracts statistical features 
from low-resolution inputs and combines them with learned image-specific embeddings 
to create conditioning vectors for rank token generation.
"""

import torch
import torch.nn as nn
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


class TokenConditioner(nn.Module):
    """
    Token Conditioner: Computes global image statistics and ID embeddings for conditioning.
    
    🎯 ANALOGY: Like a medical chart that summarizes a patient before treatment:
    - Takes vital signs: brightness (mean), contrast (std), edge sharpness (gradients)
    - Adds patient ID card: unique characteristics learned from past treatments
    - Doctors (rank tokens) use this summary to decide treatment approach
    - Each image gets personalized "diagnosis" influencing the restoration strategy
    
    Extracts statistical features from the low-resolution input and combines with learned
    image-specific embeddings. These features condition the rank token generation.
    
    Args:
        num_images (int): Total number of images in dataset (for ID embedding)
        id_dim (int): Dimension of per-image ID embedding
        use_grad_stats (bool): Whether to include gradient and Laplacian statistics
    
    Input:
        lr_luma: torch.Tensor [B, 1, H, W] - Low-resolution luminance channel
                B = batch size, 1 = single channel, H = height, W = width
        img_id: torch.Tensor [B] - Image ID indices (integers in [0, num_images-1])
    
    Output:
        torch.Tensor [B, Cc] - Conditioning vector
        Cc = 4 + id_dim if use_grad_stats else 2 + id_dim
        Contains: [mean, std, grad_mean, lap_mean, id_embedding...]
        - mean: spatial average intensity
        - std: spatial standard deviation  
        - grad_mean: average gradient magnitude (if enabled)
        - lap_mean: average Laplacian magnitude (if enabled)
        - id_embedding: learned per-image features
    """
    def __init__(self, num_images: int, id_dim: int, use_grad_stats: bool, num_channels: int = 3, num_ranks: int = 20, use_id_embed: bool = True):
        super().__init__()
        
        # Store parameters  
        self.use_grad_stats = use_grad_stats
        self.num_channels = num_channels
        self.num_ranks = num_ranks
        self.base_id_dim = id_dim if use_id_embed else 0
        self.use_id_embed = use_id_embed
        
        # Create learnable embedding lookup table for per-image conditioning (optional)
        # As recommended: Remove if not needed (when num_images=1 it's just a constant bias)
        if use_id_embed and num_images > 1:
            self.id_embed = nn.Embedding(num_images, id_dim)
        else:
            self.id_embed = None
            
        # Additional embeddings for channel IDs as recommended (optional)
        # NOTE: Rank embeddings should be handled per-token in RankTokenBank, not in global conditioning
        if use_id_embed and id_dim > 0:
            # Channel ID embedding for RGB channels (3 channels)
            self.channel_embed = nn.Embedding(num_channels, id_dim//4)  # Smaller dimension for channel info
        else:
            self.channel_embed = None
        
        # Remove rank embeddings from global conditioning - they belong in per-token processing
        self.rank_embed = None

    def forward(self, lr_luma: torch.Tensor, img_id: torch.Tensor, channel_id: torch.Tensor = None, rank_id: torch.Tensor = None) -> torch.Tensor:
        # Extract batch size from input tensor
        # lr_luma: [B,1,H,W] - luminance channel of low-resolution input
        # img_id: [B] - image indices for embedding lookup
        B = lr_luma.shape[0]
        
        # Compute basic statistical features from the luminance channel
        # mean(): spatial average intensity (brightness measure)
        mu = lr_luma.mean(dim=(2,3))             # [B,1] - average across H,W dimensions
        
        # var(): spatial variance + sqrt() gives standard deviation (contrast measure)  
        # unbiased=False uses N denominator instead of N-1, add small epsilon for numerical stability
        sd = (lr_luma.var(dim=(2,3), unbiased=False) + 1e-8).sqrt()  # [B,1] - spatial std dev
        
        if self.use_grad_stats:
            # Compute gradient-based features for texture/edge information
            dx, dy = grad_xy(lr_luma)  # Get horizontal and vertical gradients
            
            # Average gradient magnitude: measures overall "edginess" of the image
            # abs() removes direction info, mean() across spatial dims gives average edge strength
            grad_mean = (dx.abs().mean(dim=(2,3)) + dy.abs().mean(dim=(2,3)))/2.0  # [B,1]
            
            # Average Laplacian magnitude: measures texture and fine detail content
            # Laplacian detects second-order changes (curvature), useful for texture analysis
            lap_mean = conv_laplacian(lr_luma).abs().mean(dim=(2,3))               # [B,1]
            
            # Concatenate all statistical features into conditioning vector
            stats = torch.cat([mu, sd, grad_mean, lap_mean], dim=1)                # [B,4]
        else:
            # Use only basic statistics if gradient stats are disabled
            stats = torch.cat([mu, sd], dim=1)                                     # [B,2]
        
        # Add learned embeddings in controlled manner
        embeddings = []
        
        if self.id_embed is not None:
            # Get learned per-image embedding vectors using image IDs as indices
            ide = self.id_embed(img_id)                                            # [B,id_dim]
            embeddings.append(ide)
        
        if self.channel_embed is not None and channel_id is not None:
            # Add channel-specific embeddings (e.g., for RGB channel identification)
            channel_emb = self.channel_embed(channel_id)                          # [B, id_dim//4]
            embeddings.append(channel_emb)
        
        # Combine statistical features with embeddings
        if embeddings:
            # Concatenate all embeddings
            combined_embeddings = torch.cat(embeddings, dim=1)                    # [B, total_emb_dim]
            # Final conditioning vector contains both global statistics and image-specific features
            return torch.cat([stats, combined_embeddings], dim=1)                 # [B, Cc] where Cc = stats_dim + total_emb_dim
        else:
            # Return only statistical features if no embeddings are used
            return stats                                                           # [B, stats_dim] where stats_dim = 2 or 4