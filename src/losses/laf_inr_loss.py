#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict

from ..utils.image_ops import grad_xy, high_pass_filter
from ..utils.math_utils import cosine_similarity
from .charbonnier import CharbonnierLoss
from .loss_utils import attn_entropy


class LAFINRLoss(nn.Module):
    """
    LAF-INR Loss: Exact implementation matching the mathematical formulation.
    
    Loss Components:
    - L_rec: ρ(Ŷ^i - Y_i) - Charbonnier reconstruction loss
    - L_grad: ||∇Ŷ^i - ∇Y_i||_1 - Gradient preservation loss
    - L_ent: 1/(HW) ∑_x [-∑_{k=1}^K A^i(x)_k log A^i(x)_k] - Attention entropy
    - L_HP: ||G_σ * r_i||_1 - High-pass filter on residual (discourages low-freq energy)
    - L_ortho: |cos(r_i, C_i)| - Makes residual orthogonal to rank-1 canvas
    
    Total: L = E_i[L_rec + λ_g L_grad + λ_ent L_ent + λ_hp L_HP + λ_ortho L_ortho]
    """
    def __init__(self, lambda_g=0.3, lambda_ent=0.1, lambda_hp=0.15, lambda_ortho=0.05, hp_sigma=1.0):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.lambda_g = lambda_g
        self.lambda_ent = lambda_ent  
        self.lambda_hp = lambda_hp
        self.lambda_ortho = lambda_ortho
        self.hp_sigma = hp_sigma
        
    def forward(self, out: Dict[str,torch.Tensor], hr: torch.Tensor, lp: torch.Tensor, return_components: bool = False) -> Dict[str,torch.Tensor]:
        pred, resid, attn = out["pred"], out["residual"], out["attn"]
        canvas_rank1 = out["canvas_rank1"]  # [B, 1, H, W] - Rank-1 canvas
        
        # L_rec: ρ(Ŷ^i - Y_i) - Charbonnier reconstruction loss
        L_rec = self.charb(pred, hr)
        L = L_rec
        
        # L_grad: ||∇Ŷ^i - ∇Y_i||_1 - Gradient preservation loss
        L_grad = torch.tensor(0.0, device=hr.device)
        if self.lambda_g > 0:
            pred_dx, pred_dy = grad_xy(pred)
            hr_dx, hr_dy = grad_xy(hr)
            L_grad = (pred_dx - hr_dx).abs().mean() + (pred_dy - hr_dy).abs().mean()
            L = L + self.lambda_g * L_grad
            
        # L_ent: 1/(HW) ∑_x [-∑_{k=1}^K A^i(x)_k log A^i(x)_k] - Attention entropy
        # Also includes L1 sparsity loss on attention weights as recommended
        L_ent = torch.tensor(0.0, device=hr.device)
        if self.lambda_ent > 0:  
            # Entropy loss for attention diversity
            L_ent_entropy = attn_entropy(attn)
            # L1 sparsity loss for attention sparsity (encourages sparse attention)
            L_ent_l1 = attn.abs().mean()  # L1 penalty on attention weights
            L_ent = L_ent_entropy + 0.1 * L_ent_l1  # Combine entropy and L1 losses
            L = L + self.lambda_ent * L_ent
            
        # L_HP: ||G_σ * r_i||_1 - High-pass filter on residual (discourage low-frequency energy)
        L_HP = torch.tensor(0.0, device=hr.device)
        if self.lambda_hp > 0:
            # Apply high-pass filter to residual
            resid_hp = high_pass_filter(resid, sigma=self.hp_sigma)
            L_HP = resid_hp.abs().mean()
            L = L + self.lambda_hp * L_HP
            
        # L_ortho: |cos(r_i, C_i)| - Make residual orthogonal to rank-1 canvas
        L_ortho = torch.tensor(0.0, device=hr.device)
        if self.lambda_ortho > 0:
            # Expand rank-1 canvas to RGB for comparison with residual
            canvas_rgb = canvas_rank1.expand(-1, 3, -1, -1)  # [B, 3, H, W]
            L_ortho = cosine_similarity(resid, canvas_rgb)
            L = L + self.lambda_ortho * L_ortho
        
        if return_components:
            return {
                "loss": L,
                "L_rec": float(L_rec.item()),
                "L_grad": float(L_grad.item()), 
                "L_ent": float(L_ent.item()),
                "L_HP": float(L_HP.item()),
                "L_ortho": float(L_ortho.item())
            }
            
        return {"loss": L}