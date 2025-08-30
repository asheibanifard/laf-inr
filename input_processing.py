import math
import torch
import torch.nn as nn

class FourierCoordEncoder(nn.Module):
    """Maps (x,y) in [0,1]^2 to multi-band sin/cos features.
    Output dim = 4 * K  (2 bands per axis * sin/cos)."""
    def __init__(self, num_freqs: int = 14, include_input: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.out_dim = 4 * num_freqs + (2 if include_input else 0)
        self.register_buffer(
            "omegas",
            torch.tensor([(2.0 ** k) * math.pi for k in range(num_freqs)], dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def grid(self, B: int, Hs: int, Ws: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, Hs, device=device),
            torch.linspace(0, 1, Ws, device=device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        return coords  # [B,Hs,Ws,2]

    def forward(self, coords_xy: torch.Tensor) -> torch.Tensor:
        x, y = coords_xy[..., 0], coords_xy[..., 1]
        xw = x.unsqueeze(-1) * self.omegas
        yw = y.unsqueeze(-1) * self.omegas
        feats = [torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)]
        out = torch.cat(feats, dim=-1)
        if self.include_input:
            out = torch.cat([coords_xy, out], dim=-1)
        return out  # [B,Hs,Ws,4K(+2)]
