from ranks import RankTokenBank, rank1_canvas
from input_processing import FourierCoordEncoder
from queries import QueryBuilder
from attention import CrossAttentionRefiner

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class LAFINRModel(nn.Module):
    """Implements the diagrammed LAF-INR variant:
    - Rank-1 canvas path
    - Rank tokens (2..n) as a token bank
    - HR queries from Fourier PE + sampled canvas value
    - Cross-attention + residual MLP
    - Add canvas back and (optionally) broadcast to 3 channels
    """
    def __init__(self, scale: int, num_rank_tokens: int, d_model: int = 128,
                 n_heads: int = 4, mlp_hidden: int = 256, fourier_K: int = 14,
                 use_luma_canvas: bool = True, out_ch: int = 3):
        super().__init__()
        self.scale = scale
        self.use_luma_canvas = use_luma_canvas
        self.coord_enc = FourierCoordEncoder(num_freqs=fourier_K, include_input=False)
        self.token_bank = RankTokenBank(num_tokens=num_rank_tokens, d_model=d_model)
        self.query = QueryBuilder(pe_dim=self.coord_enc.out_dim, d_model=d_model)
        # predict 1-channel residual then broadcast if luma; otherwise predict 3
        self.refiner = CrossAttentionRefiner(d_model=d_model, n_heads=n_heads, mlp_hidden=mlp_hidden,
                                             out_ch=1 if use_luma_canvas else out_ch)
        self.out_ch = out_ch

    def forward(self, x_lr: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x_lr.shape
        Hs, Ws = H * self.scale, W * self.scale

        # Rank-1 canvas on LR
        canvas_lr = rank1_canvas(x_lr, use_luma=self.use_luma_canvas)  # [B,1,H,W] or [B,C,H,W]
        # Upsample canvas to HR
        canvas_hr = F.interpolate(canvas_lr, size=(Hs, Ws), mode="bicubic", align_corners=False)

        # Build HR coords and Fourier encoding
        coords = self.coord_enc.grid(B, Hs, Ws, x_lr.device)           # [B,Hs,Ws,2]
        pe = self.coord_enc(coords)                                    # [B,Hs,Ws,Pe]

        # Build queries from PE + sampled canvas value
        Q = self.query(pe, canvas_hr if self.use_luma_canvas else canvas_hr[:, :1])  # [B,Nq,D]

        # Rank token bank (2..n)
        T = self.token_bank(B)  # [B,Tk,D]

        # Cross-attention and residual MLP
        resid_vec, attn_w = self.refiner(Q, T)  # [B,Nq,out_ch'] , [B,Nq,Tk]
        B_, Nq, Cout = resid_vec.shape
        resid = resid_vec.transpose(1, 2).reshape(B_, Cout, Hs, Ws)  # [B,out_ch',Hs,Ws]

        # Add canvas back and broadcast if needed
        if self.use_luma_canvas:
            # canvas_hr: [B,1,Hs,Ws] -> broadcast to 3 channels
            base = canvas_hr.repeat(1, self.out_ch, 1, 1)
            out = (base + resid.repeat(1, self.out_ch, 1, 1)).clamp(0, 1)
        else:
            out = (canvas_hr + resid).clamp(0, 1)

        return {
            "pred": out,               # [B,3,Hs,Ws]
            "canvas_hr": canvas_hr,    # [B,1 or C,Hs,Ws]
            "attn": attn_w,            # [B,Nq,Tk]
        }
