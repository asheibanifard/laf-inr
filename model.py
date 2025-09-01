from ranks import RankTokenBank, rank1_canvas
from input_processing import FourierCoordEncoder
from queries import QueryBuilder
from attention import CrossAttentionRefiner

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class LAFINRModel(nn.Module):
    """Implements LAF-INR with atomic singular layer decomposition and constrained residual learning:
    
    Architecture:
    - L_1 (rank-1 canvas): Explicitly reconstructed first atomic layer σ_1*u_1*v_1^T
      containing coarse global structure, upsampled to HR as foundation
    - L_2, L_3, ..., L_n: Higher-order atomic layers compressed into RankTokenBank
      as learnable spectral descriptors representing essence of each singular component
    - Cross-attention: Spatially selective incorporation of higher-order details
    - Constrained residual: Canvas + small learned residuals for stable training
    """
    def __init__(self, scale: int = 4, num_rank_tokens: int = 8, d_model: int = 128,
                 n_heads: int = 4, mlp_hidden: int = 256, fourier_K: int = 14,
                 use_luma_canvas: bool = True, out_ch: int = 3, rank_start: int = 2, 
                 preserve_color: bool = False, residual_clamp: float = 0.05):
        super().__init__()
        self.scale = scale
        self.use_luma_canvas = use_luma_canvas
        self.preserve_color = preserve_color
        self.residual_clamp = residual_clamp
        self.num_rank_tokens = num_rank_tokens
        self.rank_start = rank_start
        self.coord_enc = FourierCoordEncoder(num_freqs=fourier_K, include_input=False)
        # Token bank for L_2, L_3, ..., L_{rank_start + num_rank_tokens - 1}
        self.token_bank = RankTokenBank(num_tokens=num_rank_tokens, d_model=d_model, rank_start=rank_start)
        self.query = QueryBuilder(pe_dim=self.coord_enc.out_dim, d_model=d_model)
        # For reconstruction task, always use 1-channel residuals for better control
        # They will be broadcast to RGB channels as needed
        self.refiner = CrossAttentionRefiner(d_model=d_model, n_heads=n_heads, mlp_hidden=mlp_hidden,
                                             out_ch=1)  # Always single channel for residuals
        self.out_ch = out_ch
        
        # Constrained residual learning: start at zero, gradually learn
        self.residual_scale = nn.Parameter(torch.tensor(0.15))  # Start larger for meaningful updates
        
        # Initialize attention to output near-zero initially
        self._init_small_residuals()
    
    def _init_small_residuals(self):
        """Initialize the refiner to output very small residuals initially"""
        with torch.no_grad():
            # Scale down the final MLP layers much more to start with tiny outputs
            if hasattr(self.refiner.mlp, '__iter__'):
                for i, layer in enumerate(self.refiner.mlp):
                    if isinstance(layer, nn.Linear):
                        # Make the final layer especially small
                        scale = 0.01 if i == len(self.refiner.mlp) - 1 else 0.1
                        layer.weight.data *= scale
                        if layer.bias is not None:
                            layer.bias.data.zero_()
    
    def forward(self, x_lr: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x_lr.shape
        Hs, Ws = H, W  # Same resolution for reconstruction task
            
        # Extract L_1: first atomic layer (coarse global structure)
        canvas_lr = rank1_canvas(x_lr, use_luma=self.use_luma_canvas)  # [B,1,H,W] or [B,C,H,W]

        # Build coords and Fourier encoding
        coords = self.coord_enc.grid(B, Hs, Ws, x_lr.device)           # [B,Hs,Ws,2]
        pe = self.coord_enc(coords)                                    # [B,Hs,Ws,Pe]

        # Build queries from PE + consistent canvas scalar (always use luma for consistency)
        if self.use_luma_canvas:
            canvas_scalar = canvas_lr  # Already luma [B,1,H,W]
        else:
            # Convert RGB canvas to luma for consistent scalar representation
            canvas_scalar = 0.299 * canvas_lr[:, 0:1] + 0.587 * canvas_lr[:, 1:2] + 0.114 * canvas_lr[:, 2:3]
        
        Q = self.query(pe, canvas_scalar)  # [B,Nq,D]
        Nq = Q.shape[1]
        assert Nq == Hs * Ws, f"Expected Nq={Hs*Ws}, got {Nq}"

        # Get spectral descriptors for L_2, L_3, ..., L_n
        T = self.token_bank(B)  # [B,Tk,D] representing essence of higher-order atomic layers

        # Cross-attention: selectively incorporate higher-order details
        resid_vec, attn_w = self.refiner(Q, T)  # [B,Nq,out_ch'] , [B,Nq,Tk]
        B_, Nq, Cout = resid_vec.shape
        assert Cout == 1, f"Expected single-channel residuals, got {Cout}"
        resid = resid_vec.transpose(1, 2).reshape(B_, Cout, Hs, Ws)  # [B,1,Hs,Ws]

        # Apply constrained residual learning with single clamp mechanism
        resid = self.residual_scale * resid * self.residual_clamp  # Removed tanh over-clamping

        # Final reconstruction: L_1 canvas + constrained residuals
        if self.use_luma_canvas:
            # For single image reconstruction, we want to preserve color
            # Combine luma enhancement with original color information
            if self.preserve_color:
                # Get original color ratios from LR input
                lr_upsampled = F.interpolate(x_lr, size=(Hs, Ws), mode="bicubic", align_corners=False)
                
                # Convert LR to luma for ratio calculation
                lr_luma = 0.299 * lr_upsampled[:, 0:1] + 0.587 * lr_upsampled[:, 1:2] + 0.114 * lr_upsampled[:, 2:3]
                
                # Calculate color ratios (avoid division by zero)
                eps = 1e-6
                r_ratio = lr_upsampled[:, 0:1] / (lr_luma + eps)
                g_ratio = lr_upsampled[:, 1:2] / (lr_luma + eps)
                b_ratio = lr_upsampled[:, 2:3] / (lr_luma + eps)
                
                # Enhanced luma from our model
                enhanced_luma = (canvas_lr + resid).clamp(0, 1)
                
                # Apply color ratios to enhanced luma
                r_channel = (enhanced_luma * r_ratio).clamp(0, 1)
                g_channel = (enhanced_luma * g_ratio).clamp(0, 1)
                b_channel = (enhanced_luma * b_ratio).clamp(0, 1)
                
                out = torch.cat([r_channel, g_channel, b_channel], dim=1)
            else:
                # Original luma-only mode (grayscale output)
                luma_combined = (canvas_lr + resid).clamp(0, 1)  # [B,1,Hs,Ws]
                out = luma_combined.repeat(1, self.out_ch, 1, 1)  # [B,3,Hs,Ws]
        else:
            # RGB mode: residuals are always single-channel, broadcast to RGB
            resid_rgb = resid.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
            out = (canvas_lr + resid_rgb).clamp(0, 1)

        # Prepare canvas output for analysis (convert to RGB if needed)
        if self.use_luma_canvas and canvas_lr.shape[1] == 1:
            canvas_base_output = canvas_lr.repeat(1, self.out_ch, 1, 1)  # Convert luma to RGB for analysis
        else:
            canvas_base_output = canvas_lr
            
        return {
            "pred": out,               # [B,3,Hs,Ws] final reconstruction
            "canvas_base": canvas_base_output,    # [B,3,Hs,Ws] L_1 foundation (pure canvas, no residuals)
            "attn": attn_w,            # [B,Nq,Tk] attention weights over spectral descriptors
            "residual": resid,         # [B,1,Hs,Ws] learned residuals
            "residual_scale": self.residual_scale,  # Current scale factor
            "rank_contributions": {    # For analysis: which ranks are being used
                "rank_start": self.rank_start,
                "num_tokens": self.num_rank_tokens,
                "token_ranks": [self.token_bank.get_rank_for_token(i) for i in range(self.num_rank_tokens)]
            }
        }
