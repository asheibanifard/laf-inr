#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..config import LAFINRConfig
from .components.rank1_canvas import Rank1Canvas
from .components.fourier_encoder import FourierCoordEncoder
from .attention import Query, Attend, Residual
from ..utils.image_ops import grad_xy, conv_laplacian


class LAFINR(nn.Module):
    """
    LAF-INR: Low-Rank Attention for Implicit Neural Representations
    Main model class combining all components for super-resolution.
    Args:
        cfg: LAFINRConfig - Configuration dataclass with model hyperparameters
        num_images: int - Total number of images for ID embedding
    Inputs:
        x_lp: torch.Tensor [B, 3, H, W] - Low-resolution input image
        img_id: torch.Tensor [B] - Image IDs for conditioning
        target_coords: Optional[torch.Tensor] [B, Ht, Wt, 2] - Target coordinates for super-resolution
        target_size: Optional[Tuple[int, int]] - Target output size (Ht, Wt)
    Outputs:
        Dict[str, torch.Tensor] with keys:
            "pred": [B, 3, Ht, Wt] - Super-resolved output
            "residual": [B, 3, Ht, Wt] - Predicted RGB residual
            "attn": [B, Ht*Wt, K] - Attention weights
            "canvas_base": [B, 3, Ht, Wt] - Base upsampled image
            "canvas_rank1": [B, 1 , Ht, Wt] - Rank-1 canvas (luminance)
            "canvas_rankN": [B, N, Ht, Wt] - Rank-N canvas (color)
    Note:
        - If neither target_coords nor target_size is provided, uses input size (same-size reconstruction).
        - target_coords should be normalized to [0, 1] range.

    """
    def __init__(self, cfg: LAFINRConfig, num_images: int):
        super().__init__()
        self.canvas = Rank1Canvas(use_luma=True)
        print(self.canvas)
        self.pe = FourierCoordEncoder(num_freqs=cfg.fourier_K, include_input=cfg.include_input)
        # Automatically set cond_dim to match TokenConditioner output
        # Only add id_dim if id embeddings are actually used (avoid no-op when num_images=1)
        stats_dim = 4 if cfg.use_grad_stats else 2
        # For meaningful id_embed, check if num_images > 1, otherwise it's just a constant bias
        id_emb_dim = cfg.id_dim if num_images > 1 else 0
        channel_emb_dim = cfg.id_dim // 4 if id_emb_dim > 0 else 0
        # Rank embeddings removed from global conditioning - handled per-token in RankTokenBank
        cond_dim = stats_dim + id_emb_dim + channel_emb_dim
        self.query = Query(
            pe_dim=self.pe.out_dim,
            d_model=cfg.d_model,
            extra_scalars=4,  # 4 LR-derived features: Y, |dx|, |dy|, |lap|
            num_tokens=cfg.num_rank_tokens,
            cond_dim=cond_dim,
            rank_start=cfg.rank_start,
            id_dim=cfg.id_dim,
            use_grad_stats=cfg.use_grad_stats,
            num_images=num_images,
            dropout=cfg.dropout
        )
        self.attend = Attend(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            mlp_hidden=cfg.mlp_hidden,
            out_ch=3,
            dropout=cfg.dropout
        )
        self.residual = Residual(
            alpha_init=cfg.alpha_init,
            alpha_min=cfg.alpha_min,
            warmup_epochs=cfg.alpha_warmup_epochs
        )

    def forward(self, x_lp: torch.Tensor, img_id: torch.Tensor, 
                target_coords: Optional[torch.Tensor] = None, 
                target_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with super-resolution support.
        
        Args:
            x_lp: [B,3,H,W] - Low-resolution input
            img_id: [B] - Image IDs for conditioning
            target_coords: [B,Ht,Wt,2] - Target coordinates (overrides target_size if provided)
            target_size: (Ht, Wt) - Target output size (creates regular grid if target_coords not provided)
        
        If neither target_coords nor target_size provided, uses input size (same-size reconstruction).
        """
        B,C,H,W = x_lp.shape
        
        # Determine target coordinates - CREATE HR COORDS FROM TARGET SIZE  
        if target_coords is not None:
            target_coords_norm = target_coords  # Assume already normalized to [0,1]
            Ht, Wt = target_coords.shape[1:3]
        elif target_size is not None:
            Ht, Wt = target_size
            target_coords_norm = self.pe.grid(B, Ht, Wt, x_lp.device)
        else:
            Ht, Wt = H, W
            target_coords_norm = self.pe.grid(B, Ht, Wt, x_lp.device)
        
        # PE on target coordinates
        # pe = self.pe(target_coords_norm)  # [B,Ht,Wt,Pe]
        
        # Compute LR-derived features
        Y = 0.299*x_lp[:,0:1]+0.587*x_lp[:,1:2]+0.114*x_lp[:,2:3]  # [B,1,H,W]
        dx, dy = grad_xy(Y); lap = conv_laplacian(Y)
        lr_scalars = torch.cat([Y, dx.abs(), dy.abs(), lap.abs()], dim=1)  # [B,4,H,W]
        
        # Sample LR scalars at target coordinates using grid_sample
        coords_flat = target_coords_norm.view(B, Ht*Wt, 2)
        coords_grid_sample = coords_flat.view(B, Ht, Wt, 2) * 2.0 - 1.0
        sampled_scalars = F.grid_sample(
            lr_scalars,
            coords_grid_sample,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [B,4,Ht,Wt]
        
        # Build queries and tokens using modular Query
        # Pass the luminance channel for TokenConditioner, and sampled_scalars for QueryBuilder
        queries, tokens = self.query(
            target_coords_norm, # [B,Ht,Wt,2]
            Y,  # Pass luminance [B,1,H,W] for TokenConditioner conditioning
            img_id,
            sampled_scalars  # Pass sampled scalars [B,4,Ht,Wt] for QueryBuilder
        )  # queries: [B, Ht*Wt, d_model], tokens: [B, num_tokens, d_model]
        
        # Cross-attention → residual using modular Attend
        resid_vec, attn_w = self.attend(queries, tokens)  # [B,Ht*Wt,3], [B,Nq,K]
        resid = resid_vec.transpose(1,2).reshape(B,3,Ht,Wt)  # [B,3,Ht,Wt]
        
        # Base image at target resolution (bicubic upsampling with proper alignment)
        if Ht != H or Wt != W:
            base_target = F.interpolate(x_lp, size=(Ht, Wt), mode='bicubic', 
                                      align_corners=False, antialias=True)
        else:
            base_target = x_lp
        base_detached = base_target.detach()
        
        # Compose with DETACHED base using modular Residual
        sr_out, alpha = self.residual(base_detached, resid, training=self.training, target_size=(Ht, Wt))
        canvas_rank1 = self.canvas(base_target)  # [B,1,Ht,Wt]

        return {
            "pred": sr_out, "residual": resid, "attn": attn_w,
            "canvas_base": base_detached, "canvas_rank1": canvas_rank1, "alpha": alpha
        }