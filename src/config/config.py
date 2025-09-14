#!/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class LAFINRConfig:
    """
    Configuration class for LAF-INR model hyperparameters.
    
    Attributes:
        d_model (int): Model embedding dimension (D) - shared across all components
        n_heads (int): Number of attention heads in CrossAttentionRefiner
        mlp_hidden (int): Hidden dimension in MLP layers
        fourier_K (int): Number of Fourier frequency bands for positional encoding
        include_input (bool): Whether to include raw coordinates in Fourier features
        num_rank_tokens (int): Number of rank tokens (K) in RankTokenBank
        rank_start (int): Starting rank number for token indexing
        cond_dim (int): Conditioning dimension for RankTokenBank
        id_dim (int): Dimension of per-image ID embeddings
        use_grad_stats (bool): Whether to include gradient and Laplacian statistics in conditioning
        dropout (float): Dropout probability for regularization
        alpha_min (float): Minimum residual gate strength in ResidualComposer
        alpha_warmup_epochs (int): Number of epochs with boosted residual strength
    """
    d_model: int = 192          # Increased model dimension for better capacity
    n_heads: int = 6            # More attention heads for finer patterns
    mlp_hidden: int = 384       # Proportionally increased MLP
    fourier_K: int = 14         # Number of Fourier frequency bands
    include_input: bool = False # Whether to include raw coordinates in Fourier features
    num_rank_tokens: int = 20   # More tokens for better frequency separation
    rank_start: int = 1         # Start from rank-1 for finer control
    cond_dim: int = 12          # Conditioning dimension for RankTokenBank
    id_dim: int = 8             # Dimension of per-image ID embeddings
    use_grad_stats: bool = True # Include gradient and Laplacian statistics in conditioning
    dropout: float = 0.1        # Dropout probability for regularization
    alpha_init: float = 0.5     # Slightly higher initial - still conservative
    alpha_min: float = 0.05     # Very low base alpha for stability
    alpha_warmup_epochs: int = 15   # Longer warmup for gradual alpha growth
    warmup_epochs: int = 15   # Longer warmup for gradual alpha growth