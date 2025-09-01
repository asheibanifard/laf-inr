from typing import Tuple
import torch
import torch.nn as nn

class CrossAttentionRefiner(nn.Module):
    """Enhanced cross-attention for atomic layer refinement.
    
    Processes spatial queries with rank token bank to selectively incorporate
    higher-order spectral components with improved stability and expressiveness.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_hidden: int = 256, 
                 out_ch: int = 1, dropout: float = 0.1, use_spectral_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Multi-head attention with improved stability
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Layer normalization for better training stability
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
        # Improved dropout strategy
        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout * 0.5)  # Lower dropout for attention
        
        # Enhanced MLP with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden),
            nn.GELU(),  # Better than ReLU for this task
            nn.LayerNorm(mlp_hidden),  # Additional normalization
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, out_ch)
        )
        
        # Spectral normalization for better stability
        if use_spectral_norm:
            for module in self.mlp:
                if isinstance(module, nn.Linear):
                    nn.utils.spectral_norm(module)
        
        # Learnable scaling for residual connection
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization with slight modification
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, Q: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with better stability and expressiveness.
        
        Q: [B,Nq,D] queries (spatial locations)
        T: [B,Tk,D] token bank (spectral descriptors for L_2, L_3, ...)
        returns: 
            residual: [B,Nq,out_ch] - refined spectral contributions
            attn_weights: [B,Nq,Tk] - attention weights for analysis
        """
        # Input validation
        if Q.size(-1) != self.d_model or T.size(-1) != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got Q:{Q.size(-1)}, T:{T.size(-1)}")
        
        # Normalize inputs for better stability
        Qn = self.norm_q(Q)
        Tn = self.norm_kv(T)
        
        # Cross-attention: Q attends to T (queries attend to spectral descriptors)
        attn_out, attn_w = self.mha(Qn, Tn, Tn, need_weights=True, average_attn_weights=False)
        
        # Apply attention dropout
        attn_out = self.dropout_attn(attn_out)
        
        # Residual connection with learnable scaling (scale attention contribution, not original query)
        attn_out = self.norm_out(Q + self.residual_scale * attn_out)
        
        # Process attention weights: [B, n_heads, Nq, Tk] -> [B, Nq, Tk]
        if attn_w.dim() == 4:  # Multi-head attention weights
            attn_w_mean = attn_w.mean(dim=1)  # Average across heads
        else:
            attn_w_mean = attn_w
        
        # Enhanced MLP with concatenated features
        H = torch.cat([Q, attn_out], dim=-1)  # [B,Nq,2D] - original + attended features
        residual = self.mlp(H)  # [B,Nq,out_ch]
        
        # Apply final dropout
        residual = self.dropout(residual)
        
        return residual, attn_w_mean