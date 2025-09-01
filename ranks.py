import torch
import torch.nn as nn
from typing import Dict, List, Optional

@torch.no_grad()
def svd_decompose_to_atomic_layers(lr: torch.Tensor, max_rank: int = 32, use_luma: bool = True, 
                                 add_noise: bool = False, noise_scale: float = 1e-6) -> Dict[int, torch.Tensor]:
    """Decompose image into atomic singular triplet layers L_i = σ_i * u_i * v_i^T.
    
    Each layer L_i captures a specific rank-1 structure from the i-th singular triplet (u_i, σ_i, v_i^T).
    L_1 contains coarse global structure (largest singular value), while higher layers L_2, L_3, ... 
    encode increasingly finer details and textures.
    
    Args:
      lr: [B,C,H,W] input image in [0,1]
      max_rank: maximum number of singular components to extract
      use_luma: if True, convert to Y (luma) and decompose single-channel; else per-channel
      add_noise: if True, add small random noise to avoid SVD degeneracies (breaks determinism)
      noise_scale: scale of regularization noise when add_noise=True
    Returns:
      atomic_layers: Dict[rank, tensor] where each tensor is [B,1,H,W] (luma) or [B,C,H,W] (rgb)
                    atomic_layers[i] = L_i = σ_i * u_i * v_i^T (the i-th atomic layer)
    """
    B, C, H, W = lr.shape
    device = lr.device
    atomic_layers = {}
    
    # Important: Ensure sufficient image resolution for meaningful SVD
    if min(H, W) < 32:
        print(f"Warning: Image resolution {H}x{W} may be too small for meaningful SVD decomposition")
    
    if use_luma:
        # Convert to Y (BT.601) then decompose on Y channel
        if C == 3:
            R, G, Bc = lr[:, 0], lr[:, 1], lr[:, 2]
            y = 0.299 * R + 0.587 * G + 0.114 * Bc  # [B,H,W]
        else:
            y = lr[:, 0]
            
        for b in range(B):
            X = y[b]  # [H,W]
            # Optional regularization to avoid numerical issues
            if add_noise:
                X_reg = X + noise_scale * torch.randn_like(X)
            else:
                # Deterministic regularization: add tiny constant to diagonal
                X_reg = X + noise_scale * torch.eye(min(X.shape), device=X.device, dtype=X.dtype)
            
            U, S, Vh = torch.linalg.svd(X_reg, full_matrices=False)  # X = U @ diag(S) @ Vh
            
            # Extract atomic layers for each rank
            for rank in range(1, min(max_rank + 1, len(S) + 1)):
                if rank not in atomic_layers:
                    atomic_layers[rank] = torch.zeros((B, 1, H, W), device=device, dtype=lr.dtype)
                
                # Atomic layer L_rank = σ_rank * u_rank * v_rank^T (proper SVD form)
                if S[rank-1] > 1e-8:  # Only include significant singular values
                    L_i = S[rank-1] * torch.outer(U[:, rank-1], Vh[rank-1, :])  # [H,W]
                    atomic_layers[rank][b, 0] = L_i
    else:
        # Per-channel decomposition
        for b in range(B):
            for c in range(C):
                X = lr[b, c]  # [H,W]
                # Optional regularization to avoid numerical issues
                if add_noise:
                    X_reg = X + noise_scale * torch.randn_like(X)
                else:
                    # Deterministic regularization: add tiny constant to diagonal
                    X_reg = X + noise_scale * torch.eye(min(X.shape), device=X.device, dtype=X.dtype)
                
                U, S, Vh = torch.linalg.svd(X_reg, full_matrices=False)
                
                for rank in range(1, min(max_rank + 1, len(S) + 1)):
                    if rank not in atomic_layers:
                        atomic_layers[rank] = torch.zeros((B, C, H, W), device=device, dtype=lr.dtype)
                    
                    # Atomic layer L_rank = σ_rank * u_rank * v_rank^T (proper SVD form)
                    if S[rank-1] > 1e-8:  # Only include significant singular values
                        L_i = S[rank-1] * torch.outer(U[:, rank-1], Vh[rank-1, :])  # [H,W]
                        atomic_layers[rank][b, c] = L_i
    
    # CRITICAL: No layer-wise normalization - preserve SVD properties!
    # Only clamp extreme values for numerical stability
    for rank in atomic_layers:
        layer = atomic_layers[rank]
        # Only clamp extreme outliers, don't normalize
        atomic_layers[rank] = torch.clamp(layer, -10.0, 10.0)
    
    return atomic_layers

@torch.no_grad()
def rank1_canvas(lr: torch.Tensor, use_luma: bool = True, add_noise: bool = False, 
                 noise_scale: float = 1e-6) -> torch.Tensor:
    """Extract L_1: the first atomic layer containing coarse global structure.
    
    This is the rank-1 canvas dominated by the largest singular value σ_1.
    L_1 = σ_1 * u_1 * v_1^T captures the most significant structural information.
    
    Args:
      lr: [B,C,H,W] in [0,1]
      use_luma: if True, convert to Y (luma) and compute single-channel L_1; else per-channel L_1.
      add_noise: if True, add small random noise to avoid SVD degeneracies (breaks determinism)
      noise_scale: scale of regularization when add_noise=True
    Returns:
      canvas: [B,1,H,W] if use_luma else [B,C,H,W] - the L_1 atomic layer
    """
    atomic_layers = svd_decompose_to_atomic_layers(lr, max_rank=1, use_luma=use_luma, 
                                                 add_noise=add_noise, noise_scale=noise_scale)
    return atomic_layers[1]  # Return L_1 (first atomic layer)


# ---------------------------
# Rank Token Bank for Higher-Order Layers (L_2, L_3, ...)
# ---------------------------
class RankTokenBank(nn.Module):
    """Learnable embeddings representing the essence of higher-order singular components.
    
    Instead of explicitly reconstructing all atomic layers L_2, L_3, L_4, ... in pixel space,
    LAF-INR compresses their contributions into compact spectral descriptors. Each token
    represents the "essence" of a specific singular component beyond rank-1, enabling 
    efficient attention across different rank contributions.
    
    These tokens act as:
    - Compact representations of L_i = σ_i * u_i * v_i^T for i ≥ 2
    - Spectral descriptors capturing rank-specific image structures
    - Keys/values for cross-attention to selectively incorporate higher-order details
    """
    def __init__(self, num_tokens: int, d_model: int, rank_start: int = 2):
        super().__init__()
        self.num_tokens = num_tokens
        self.rank_start = rank_start  # Start from L_2 (since L_1 is explicit canvas)
        
        # Learnable tokens representing L_2, L_3, ..., L_{rank_start + num_tokens - 1}
        self.tokens = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        nn.init.normal_(self.tokens, std=0.02)
        
        # Optional: learnable rank embeddings to distinguish different singular components
        self.rank_embeddings = nn.Parameter(torch.randn(num_tokens, d_model) * 0.01)
        nn.init.normal_(self.rank_embeddings, std=0.01)

    def forward(self, B: int) -> torch.Tensor:
        """Return spectral descriptors for higher-order atomic layers.
        
        Args:
            B: batch size
        Returns:
            tokens: [B, num_tokens, d_model] representing essence of L_2, L_3, ..., L_n
        """
        # Combine base tokens with rank-specific embeddings
        enhanced_tokens = self.tokens + self.rank_embeddings  # [num_tokens, d_model]
        
        # Broadcast to batch dimension
        return enhanced_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tokens, d_model]
    
    def get_rank_for_token(self, token_idx: int) -> int:
        """Get the singular value rank corresponding to a token index."""
        return self.rank_start + token_idx

# ---------------------------
# Utility Functions for Atomic Layer Analysis
# ---------------------------
@torch.no_grad()
def analyze_singular_spectrum(lr: torch.Tensor, use_luma: bool = True, add_noise: bool = False) -> Dict[str, torch.Tensor]:
    """Analyze the singular value spectrum of an image for rank selection.
    
    Args:
        lr: [B,C,H,W] input image
        use_luma: if True, analyze luma channel; else analyze per-channel
        add_noise: if True, add small random noise to avoid SVD degeneracies (breaks determinism)
    Returns:
        spectrum_info: Dict containing:
            - 'singular_values': [B, max_rank] singular values for each image
            - 'energy_ratios': [B, max_rank] normalized energy contribution per rank
            - 'cumulative_energy': [B, max_rank] cumulative energy up to each rank
    """
    B, C, H, W = lr.shape
    device = lr.device
    max_possible_rank = min(H, W)
    
    singular_values = torch.zeros((B, max_possible_rank), device=device)
    
    if use_luma:
        # Convert to Y (BT.601)
        if C == 3:
            R, G, Bc = lr[:, 0], lr[:, 1], lr[:, 2]
            y = 0.299 * R + 0.587 * G + 0.114 * Bc
        else:
            y = lr[:, 0]
            
        for b in range(B):
            X = y[b]
            # Use deterministic SVD by default for reproducibility
            if add_noise:
                X = X + 1e-8 * torch.randn_like(X)
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            singular_values[b, :len(S)] = S
    else:
        # Average across channels for spectrum analysis
        for b in range(B):
            channel_svs = []
            for c in range(C):
                X = lr[b, c]
                # Use deterministic SVD by default for reproducibility
                if add_noise:
                    X = X + 1e-8 * torch.randn_like(X)
                U, S, Vh = torch.linalg.svd(X, full_matrices=False)
                channel_svs.append(S)
            # Average singular values across channels
            avg_sv = torch.stack(channel_svs).mean(dim=0)
            singular_values[b, :len(avg_sv)] = avg_sv
    
    # Compute energy ratios and cumulative energy
    total_energy = (singular_values ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    energy_ratios = (singular_values ** 2) / (total_energy + 1e-8)  # [B, max_rank]
    cumulative_energy = torch.cumsum(energy_ratios, dim=1)  # [B, max_rank]
    
    return {
        'singular_values': singular_values,
        'energy_ratios': energy_ratios,
        'cumulative_energy': cumulative_energy
    }
