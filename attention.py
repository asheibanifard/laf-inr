from typing import Tuple
import torch
import torch.nn as nn

class CrossAttentionRefiner(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_hidden: int = 256, out_ch: int = 1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden), 
            nn.GELU(),  # More stable than ReLU
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, out_ch)
        )

    def forward(self, Q: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Q: [B,Nq,D] queries
        T: [B,Tk,D] token bank (keys/values)
        returns residual: [B,out_ch,Nq], attn_weights: [B,Nq,Tk]
        """
        # Normalize inputs for stability
        Qn = self.norm_q(Q)
        Tn = self.norm_kv(T)
        
        # Attention with dropout
        attn_out, attn_w = self.mha(Qn, Tn, Tn, need_weights=True, average_attn_weights=False)
        attn_out = self.dropout(attn_out)
        
        # Residual connection + normalization
        attn_out = self.norm_out(attn_out + Q)
        
        # attn_w: [B, n_heads, Nq, Tk] -> average heads for loss/inspection
        attn_w_mean = attn_w.mean(dim=1)  # [B,Nq,Tk]
        
        # MLP with residual
        H = torch.cat([Q, attn_out], dim=-1)  # [B,Nq,2D]
        resid = self.mlp(H)  # [B,Nq,out_ch]
        
        return resid, attn_w_mean