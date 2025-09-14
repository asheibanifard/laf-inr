#!/usr/bin/env python3
"""
Residual Composer: Intelligent residual composition with progressive alpha scaling.

This module provides the ResidualComposer class which combines base predictions 
with residual corrections using a learnable and adaptive alpha parameter for 
training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualComposer(nn.Module):
    """
    Residual Composer: Intelligent residual composition with progressive alpha scaling
    
    🎯 BALANCED APPROACH: This module combines base predictions with residual corrections
    using a learnable and adaptive alpha parameter for training stability.
    
    Key Features:
    - Conservative alpha initialization to prevent training instability
    - Progressive warmup schedule for gradual residual integration
    - Learnable alpha parameter with bounded growth
    - Epoch-aware scaling for training dynamics adaptation
    
    Mathematical Formula:
    output = base + α * residual, where α ∈ [alpha_min, alpha_max]
    
    Args:
        alpha_init (float): Initial alpha value (not directly used due to softplus)
        alpha_min (float): Minimum alpha value for stability
        warmup_epochs (int): Number of epochs for progressive alpha warmup
    """
    def __init__(self, alpha_init: float, alpha_min: float, warmup_epochs: int):
        super().__init__()
        # Initialize learnable parameter very conservatively to avoid instability
        init_val = -2.5  # Gives softplus(-2.5) ≈ 0.08
        self.a = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))  # Learnable alpha parameter
        # Store configuration parameters as plain attributes
        self.alpha_min = float(alpha_min)
        self.warmup_epochs = int(warmup_epochs)
        # Register epoch counter as buffer (not trainable, but part of state)
        self.register_buffer('epoch', torch.tensor(0.0))
        
    def set_epoch(self, epoch: int):
        """
        Update the current epoch for progressive alpha scheduling
        
        Args:
            epoch (int): Current training epoch
        """
        self.epoch.fill_(epoch)
        
    def forward(self, base: torch.Tensor, resid_rgb: torch.Tensor, training: bool = True, target_size: tuple = None) -> torch.Tensor:
        """
        Forward pass: Properly upsample canvas, broadcast to 3 channels, and add residual
        
        Process:
        1. Handle canvas upsampling to HR size if needed
        2. Broadcast single-channel canvas to 3 channels (RGB) if needed  
        3. Reshape residual from [B,H*W,3] to [B,3,H,W] format
        4. Compute alpha and apply residual composition
        5. Ensure output is in correct [B,3,H,W] format as recommended
        
        Args:
        base (torch.Tensor): Base prediction - can be:
            - [B,1,h,w] low-res single channel (needs upsampling + broadcast)
            - [B,3,H,W] upsampled RGB canvas  
            - [B,H,W,3] spatial-last format
        resid_rgb (torch.Tensor): Residual correction [B,H*W,3] from cross-attention
        training (bool): Whether model is in training mode
        target_size (tuple, optional): (H, W) target output size. Required if input is not already [B,3,H,W].

        Returns:
            Tuple[torch.Tensor, float]: 
                - Final composed output [B,3,H,W] in proper format
                - Current alpha value for monitoring
        """
        
        # Step 1: Handle canvas upsampling and broadcasting as recommended
        H, W = None, None
        if target_size is not None:
            H, W = target_size
        # Handle base shape
        if base.dim() == 4:
            if base.shape[1] == 1:
                if H is None or W is None:
                    raise ValueError("target_size must be provided when upsampling base from [B,1,h,w]")
                base_upsampled = F.interpolate(base, size=(H, W), mode='bilinear', align_corners=False)
                base = base_upsampled.repeat(1, 3, 1, 1)
            elif base.shape[1] == 3:
                if H is not None and (base.shape[2] != H or base.shape[3] != W):
                    base = F.interpolate(base, size=(H, W), mode='bilinear', align_corners=False)
            elif base.shape[-1] == 3:
                base = base.permute(0, 3, 1, 2)
                if H is not None and (base.shape[2] != H or base.shape[3] != W):
                    base = F.interpolate(base, size=(H, W), mode='bilinear', align_corners=False)

        # Step 2: Reshape residual from [B,H*W,3] to [B,3,H,W] format
        if resid_rgb.dim() == 3 and resid_rgb.shape[-1] == 3:
            B, HW, C = resid_rgb.shape
            if H is None or W is None:
                raise ValueError("target_size must be provided when reshaping residual from [B,HW,3]")
            resid_rgb = resid_rgb.view(B, H, W, C).permute(0, 3, 1, 2)
        elif resid_rgb.dim() == 4 and resid_rgb.shape[-1] == 3:
            resid_rgb = resid_rgb.permute(0, 3, 1, 2)
            if H is not None and (resid_rgb.shape[2] != H or resid_rgb.shape[3] != W):
                resid_rgb = F.interpolate(resid_rgb, size=(H, W), mode='bilinear', align_corners=False)

        # Step 3: Compute conservative alpha gate with bounded range
        alpha = self.alpha_min + F.softplus(self.a)
        if self.epoch < self.warmup_epochs:
            progress = self.epoch / self.warmup_epochs
            warmup_boost = 0.15 * (1.0 - progress)
            alpha = alpha + warmup_boost
        else:
            post_warmup_epochs = self.epoch - self.warmup_epochs
            post_warmup_progress = min(post_warmup_epochs / (self.warmup_epochs * 0.5), 1.0)
            growth_boost = 0.05 * post_warmup_progress
            alpha = alpha + growth_boost
        
        out = base + alpha * resid_rgb
        if not training:
            out = out.clamp(0, 1)
        return out, alpha