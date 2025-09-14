#!/usr/bin/env python3
import torch
import torch.nn as nn


class Rank1Canvas(nn.Module):
    """
    Rank-1 Canvas: Creates a rank-1 approximation of the input image.

    ✅ Now robust for training:
    - Handles small/degenerate inputs gracefully
    - Falls back to zeros if SVD fails
    - Prevents crashes during interruptions or weird batches

    Args:
        use_luma (bool): If True, converts RGB to luminance; if False, uses first channel.

    Input:
        x: torch.Tensor [B, C, H, W] - Input image tensor
           B = batch size, C = channels, H = height, W = width

    Output:
        torch.Tensor [B, 1, H, W] - Rank-1 approximation (safe fallback if SVD fails)
    """
    def __init__(self, use_luma: bool = True):
        super().__init__()
        self.use_luma = use_luma

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # Early exit for degenerate inputs
        if H < 2 or W < 2:
            return out

        # Convert to luminance or use first channel
        if self.use_luma and C == 3:
            y = 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]  # [B,H,W]
        else:
            y = x[:,0]  # [B,H,W]

        for b in range(B):
            try:
                X = y[b] + 1e-6*torch.randn_like(y[b])  # jitter for numerical stability
                U, S, Vh = torch.linalg.svd(X, full_matrices=False)

                if S.numel() > 0:
                    rank1 = S[0] * torch.outer(U[:,0], Vh[0,:])
                    out[b,0] = rank1
                else:
                    out[b,0] = torch.zeros(H, W, device=x.device, dtype=x.dtype)

            except Exception as e:
                # Graceful fallback: no crash
                print(f"[Rank1Canvas] ⚠️ SVD failed on sample {b}: {e}")
                out[b,0] = torch.zeros(H, W, device=x.device, dtype=x.dtype)

        return out
