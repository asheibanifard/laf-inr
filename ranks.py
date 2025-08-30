import torch
import torch.nn as nn

@torch.no_grad()
def rank1_canvas(lr: torch.Tensor, use_luma: bool = True) -> torch.Tensor:
    """Compute rank-1 canvas on LR.
    Args:
      lr: [B,C,H,W] in [0,1]
      use_luma: if True, convert to Y (luma) and compute a single-channel rank-1; else per-channel rank-1.
    Returns:
      canvas: [B,1,H,W] if use_luma else [B,C,H,W]
    """
    B, C, H, W = lr.shape
    device = lr.device
    if use_luma:
        # Convert to Y (BT.601) then rank-1 on Y
        if C == 3:
            R, G, Bc = lr[:, 0], lr[:, 1], lr[:, 2]
            y = 0.299 * R + 0.587 * G + 0.114 * Bc  # [B,H,W]
        else:
            y = lr[:, 0]
        canvas = torch.empty((B, 1, H, W), device=device, dtype=lr.dtype)
        for b in range(B):
            X = y[b]
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # X ~ U @ diag(S) @ Vh
            # Rank-1 reconstruction
            X1 = (U[:, :1] * S[:1]) @ Vh[:1, :]
            canvas[b, 0] = X1
        return canvas.clamp(0, 1)
    else:
        canvas = torch.empty_like(lr)
        for b in range(B):
            for c in range(C):
                X = lr[b, c]
                U, S, Vh = torch.linalg.svd(X, full_matrices=False)
                X1 = (U[:, :1] * S[:1]) @ Vh[:1, :]
                canvas[b, c] = X1
        return canvas.clamp(0, 1)


# ---------------------------
# Rank Token Bank (K=2..n)
# ---------------------------
class RankTokenBank(nn.Module):
    """Learned token bank for ranks 2..n."""
    def __init__(self, num_tokens: int, d_model: int):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, B: int) -> torch.Tensor:
        # Broadcast tokens to batch: [B, T, D]
        return self.tokens.unsqueeze(0).expand(B, -1, -1)
