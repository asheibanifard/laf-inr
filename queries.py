import torch
import torch.nn as nn

class QueryBuilder(nn.Module):
    def __init__(self, pe_dim: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim + 1, d_model), 
            nn.GELU(),  # More stable activation
            nn.LayerNorm(d_model),  # Add normalization
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )

    def forward(self, pe: torch.Tensor, canvas_hr: torch.Tensor) -> torch.Tensor:
        """
        pe:        [B,Hs,Ws,Pe]
        canvas_hr: [B,1,Hs,Ws]  (sample value at each (x,y))
        returns:   [B, Nq, D] where Nq = Hs*Ws
        """
        B, Hs, Ws, Pe = pe.shape
        v = canvas_hr.permute(0, 2, 3, 1)  # [B,Hs,Ws,1]
        q = torch.cat([pe, v], dim=-1).view(B, Hs * Ws, Pe + 1)
        q = self.mlp(q)
        return q  # [B,Nq,D]
