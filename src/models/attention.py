#!/usr/bin/env python3
"""
Attention Module: Combined attention classes and coordinate encoding for LAF-INR.

This module provides the high-level Query, Attend, and Residual classes that combine
multiple components for the complete attention pipeline, as well as the FourierCoordEncoder
for positional encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the component classes
from .components.token_conditioner import TokenConditioner
from .components.rank_token_bank import RankTokenBank
from .components.query_builder import QueryBuilder
from .components.attention_refiner import CrossAttentionRefiner
from .components.residual_composer import ResidualComposer


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


class Query(torch.nn.Module):
    """
        FourierCoordEncoder: Positional Encoding (PE)
        of input coordinates.
        TokenConditioner: Conditions on low-res image
        and image ID to produce conditioning vector
        RankTokenBank: Produces a bank of tokens
        conditioned on the conditioning vector
        QueryBuilder: Combines PE and extra
        scalars to produce query vectors
        
        Inputs:
            pe_dim: int - dimension of positional encoding.
            shape = [B, H, W, pe_dim]
            d_model: int - dimension of model (token and
            query vectors). shape = [B, num_tokens, d_model]
            extra_scalars: int - number of extra scalar inputs
            (e.g., scale, rotation). shape = [B, num_scalars]
            num_tokens: int - number of tokens in the token
            bank. shape = [B, num_tokens]
            cond_dim: int - dimension of conditioning vector
            from TokenConditioner. shape = [B, cond_dim]
            rank_start: int - starting rank for low-rank
            token adaptation. shape = [int]
            id_dim: int - dimension of image ID embedding.
            shape = [B, id_dim]
            use_grad_stats: bool - whether to use gradient
            statistics in TokenConditioner. shape = [bool]
            num_images: int - total number of images for
            ID embedding. shape = [int]
            dropout: float - dropout rate in QueryBuilder.
            shape = [float]
    """
    def __init__(self,
                 pe_dim,
                 d_model,
                 extra_scalars,
                 num_tokens,
                 cond_dim,
                 rank_start,
                 id_dim,
                 use_grad_stats,
                 num_images,
                 dropout=0.0):
        super().__init__()
        self.encoder = FourierCoordEncoder(num_freqs=pe_dim//4,
                                           include_input=False)
        self.token_conditioner = TokenConditioner(num_images=num_images,
                                                  id_dim=id_dim,
                                                  use_grad_stats=use_grad_stats,
                                                  use_id_embed=num_images > 1)  # Disable id_embed if only 1 image
        self.token_bank = RankTokenBank(num_tokens=num_tokens,
                                        d_model=d_model,
                                        rank_start=rank_start,
                                        cond_dim=cond_dim)
        self.query_builder = QueryBuilder(pe_dim=pe_dim,
                                           d_model=d_model,
                                           extra_scalars=extra_scalars,
                                           dropout=dropout)

    def forward(self,
                coords_xy,
                lr_luma,
                img_id,
                scalars):
        '''
            coords_xy: [B, H, W, 2] - normalized coordinates in [-1, 1]
            lr_luma: [B, 1, h, w] - low-res grayscale image
            img_id: [B] - image IDs
            scalars: [B, S, H, W] or [B, H, W, S] - local per-pixel features (e.g., luma, |dx|, |dy|, |laplacian|)
                S = number of local scalar features per pixel
            # Do NOT use for global extra scalars; use a separate argument if needed.
        '''
        # 1. Encode coordinates
        pe = self.encoder(coords_xy)  # [B, H, W, pe_dim]
        # 2. Token conditioning
        B = coords_xy.shape[0]
        device = coords_xy.device
        # Create channel and rank IDs for embeddings
        channel_id = torch.zeros(B, dtype=torch.long, device=device)  # Default channel 0
        rank_id = torch.zeros(B, dtype=torch.long, device=device)     # Default rank 0
        cond = self.token_conditioner(lr_luma, img_id, channel_id, rank_id)  # [B, cond_dim]
        # 3. Token bank
        tokens = self.token_bank(B=coords_xy.shape[0],
                                 cond=cond)  # [B, num_tokens, d_model]
        # 4. Query building with canvas sample as recommended
        B, H, W = pe.shape[:3]
        canvas = F.interpolate(lr_luma, size=(H, W), mode='bilinear', align_corners=False)  # [B, 1, H, W]
        canvas = canvas.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        queries = self.query_builder(pe, scalars, canvas)  # [B, H*W, d_model] with canvas
        return queries, tokens


class Attend(torch.nn.Module):
    """
        CrossAttentionRefiner: Applies cross-attention between queries and tokens to produce refined features
        
        Inputs:
            d_model: int - dimension of model (token and query vectors). shape = [B, num_tokens, d_model]
            n_heads: int - number of attention heads. shape = [int]
            mlp_hidden: int - hidden layer size in MLP. shape = [int]
            out_ch: int - output channel dimension after refinement. shape = [int]
            dropout: float - dropout rate in attention and MLP layers. shape = [float]
    """
    def __init__(self,
                 d_model,
                 n_heads,
                 mlp_hidden,
                 out_ch,
                 dropout=0.0):
        super().__init__()
        self.refiner = CrossAttentionRefiner(d_model=d_model,
                                             n_heads=n_heads,
                                             mlp_hidden=mlp_hidden,
                                             out_ch=out_ch,
                                             dropout=dropout)

    def forward(self, queries, tokens):
        '''
            queries: [B, H*W, d_model] - query vectors from QueryBuilder
            tokens: [B, num_tokens, d_model] - token bank from RankTokenBank
        '''
        # queries: [B, H*W, d_model], tokens: [B, num_tokens, d_model]
        residual, attn_weights = self.refiner(queries, tokens)
        return residual, attn_weights


class Residual(torch.nn.Module):
    """ 
        Inputs:
            alpha_init: float - initial alpha blending factor. shape = [float]
            alpha_min: float - minimum alpha blending factor after warmup. shape = [float]
            warmup_epochs: int - number of epochs for alpha warmup. shape = [int]

    """
    def __init__(self, alpha_init=0.8, alpha_min=0.5, warmup_epochs=10):
        super().__init__()
        self.composer = ResidualComposer(alpha_init=alpha_init,
                                         alpha_min=alpha_min,
                                         warmup_epochs=warmup_epochs)

    def set_epoch(self, epoch: int):
        """Set the current epoch for alpha warmup scheduling."""
        self.composer.set_epoch(epoch)

    def forward(self, base, resid_rgb, training=True, target_size=None):
        '''
            base: [B, 3, H, W] - base image (upsampled LR)
            resid_rgb: [B, 3, H, W] - predicted residual RGB to add to base
            training: bool - whether in training mode (for alpha warmup)
            target_size: (H, W) - required for non-square or ambiguous shapes
        '''
        # Compose the final super-resolved output
        sr_output, alpha = self.composer(base, resid_rgb, training=training, target_size=target_size)
        return sr_output, alpha