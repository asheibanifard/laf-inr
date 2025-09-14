#!/usr/bin/env python3
import math
import torch
import torch.nn as nn


class FourierCoordEncoder(nn.Module):
    """
    Fourier Coordinate Encoder: Positional encoding using Fourier features.
    
    🎯 ANALOGY: Like a sophisticated GPS system for pixels:
    - Each pixel location gets a unique "address" made of sine and cosine waves
    - Low frequencies = "what country/state" (global position)
    - High frequencies = "what street/building number" (precise location)  
    - Multiple frequencies together create a unique fingerprint for each position
    
    Maps 2D coordinates to high-dimensional Fourier features for implicit neural representations.
    Uses sinusoidal functions at multiple frequencies to encode spatial positions.
    
    Args:
        num_freqs (int): Number of frequency bands (K in the paper)
        include_input (bool): Whether to include raw coordinates in output
    
    Input (forward):
        coords_xy: torch.Tensor [B, H, W, 2] or [..., 2] - Normalized coordinates
               Last dimension contains [x, y] coordinates in [0, 1]
    
    Output:
        torch.Tensor [..., 4*K] or [..., 4*K+2] - Fourier features
        Contains [sin(2^0*π*x), cos(2^0*π*x), ..., sin(2^K*π*y), cos(2^K*π*y)]
        If include_input=True, raw [x,y] are prepended
    
    Grid method:
        Input: B (batch_size), H (height), W (width), device
        Output: torch.Tensor [B, H, W, 2] - Coordinate grid for spatial locations
    """
    def __init__(self, num_freqs: int, include_input: bool):
        super().__init__()
        # Store whether to include raw coordinates in the output features
        self.include_input = include_input
        
        # Create frequency scales: [2^0*π, 2^1*π, 2^2*π, ..., 2^(K-1)*π]
        # Higher frequencies capture finer spatial details, lower frequencies capture global structure
        # register_buffer makes this part of the model state but not a trainable parameter
        self.register_buffer("omegas", 
                           torch.tensor([(2.0**k)*math.pi for k in range(num_freqs)], dtype=torch.float32), 
                           persistent=False)  # Don't save in model checkpoints
        
        # Calculate output dimension: 4 features per frequency (sin/cos for x and y)
        # Plus 2 raw coordinates if include_input=True
        self.out_dim = 4*num_freqs + (2 if include_input else 0)

    @torch.no_grad()  # Grid generation doesn't need gradients
    def grid(self, B: int, H: int, W: int, device) -> torch.Tensor:
        # CRITICAL: Use pixel centers (i+0.5)/H for align_corners=False consistency
        # This prevents bias toward copying LR by ensuring proper coordinate alignment
        
        # Create pixel index arrays for height and width
        yy = torch.arange(H, device=device, dtype=torch.float32)  # [0, 1, 2, ..., H-1]
        xx = torch.arange(W, device=device, dtype=torch.float32)  # [0, 1, 2, ..., W-1]
        
        # Convert to normalized coordinates [0,1] using pixel centers
        # Adding 0.5 ensures we sample at pixel centers, not corners
        yy = (yy + 0.5) / H  # Pixel centers: (i+0.5)/H ∈ [0.5/H, (H-0.5)/H]
        xx = (xx + 0.5) / W  # Pixel centers: (j+0.5)/W ∈ [0.5/W, (W-0.5)/W]
        
        # Create 2D coordinate grids using "ij" indexing (matrix indexing)
        # yy becomes [H, W] with row-wise coordinates, xx becomes [H, W] with column-wise coordinates
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        
        # Stack to create [H, W, 2] coordinate tensor, then expand for batch dimension
        # Final shape: [B, H, W, 2] where last dim is [x_coord, y_coord]
        return torch.stack([xx,yy], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)

    def forward(self, coords_xy: torch.Tensor) -> torch.Tensor:
        # Extract x and y coordinates from the last dimension
        x, y = coords_xy[...,0], coords_xy[...,1]  # Both have shape [...] (same as input without last dim)
        
        # Compute frequency-modulated coordinates for all frequencies simultaneously
        # Broadcasting: x[...] * omegas[K] -> [..., K] for all K frequencies
        xw = x.unsqueeze(-1)*self.omegas  # [..., K] - x coordinates at all frequencies
        yw = y.unsqueeze(-1)*self.omegas  # [..., K] - y coordinates at all frequencies
        
        # Apply sinusoidal functions to create Fourier features
        # Each coordinate gets sin and cos components at each frequency
        feats = [torch.sin(xw),     # [..., K] - sin components for x
                 torch.cos(xw),     # [..., K] - cos components for x  
                 torch.sin(yw),     # [..., K] - sin components for y
                 torch.cos(yw)]     # [..., K] - cos components for y
        
        # Concatenate all features: [..., 4*K] total Fourier features
        out = torch.cat(feats, dim=-1)
        
        # Optionally prepend raw coordinates to the feature vector
        if self.include_input: 
            out = torch.cat([coords_xy, out], dim=-1)  # [..., 2 + 4*K]
        
        return out