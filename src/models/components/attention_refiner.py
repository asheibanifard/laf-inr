#!/usr/bin/env python3
"""
Cross-Attention Refiner: Multi-head attention between queries and rank tokens.

This module provides the CrossAttentionRefiner class which performs cross-attention 
between spatial queries (from QueryBuilder) and rank tokens (from RankTokenBank) 
to produce RGB residual corrections for each pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CrossAttentionRefiner(nn.Module):
    """
    Cross-Attention Refiner: Multi-head attention between queries and rank tokens.
    
    🎯 ANALOGY: Like a consultation between pixels and restoration experts:
    - Each pixel (query) asks all experts (tokens): "How should I be enhanced?"  
    - Expert 1 says: "For global consistency, adjust brightness like this"
    - Expert 8 says: "For medium details, add this texture pattern"
    - Expert 16 says: "For fine edges, sharpen like this"
    - Pixel listens to all experts, weights their advice, and gets RGB correction
    - Attention weights show which expert each pixel trusted most
    
    Performs cross-attention between spatial queries (from QueryBuilder) and rank tokens
    (from RankTokenBank) to produce RGB residual corrections for each pixel.
    
    Args:
        d_model (int): Model dimension (must match query and token dimensions)
        n_heads (int): Number of attention heads
        mlp_hidden (int): Hidden dimension in the output MLP
        out_ch (int): Number of output channels (typically 3 for RGB)
        dropout (float): Dropout probability for regularization
    
    Input:
        Q: torch.Tensor [B, Nq, D] - Query vectors from QueryBuilder
           Nq = H*W (number of spatial locations), D = d_model
        T: torch.Tensor [B, K, D] - Rank tokens from RankTokenBank  
           K = num_tokens (number of frequency ranks), D = d_model
    
    Output:
        resid_vec: torch.Tensor [B, Nq, 3] - RGB residual for each spatial location
        attn_w: torch.Tensor [B, Nq, K] - Attention weights (averaged over heads)
        
    Attention Mechanism:
        - Q serves as queries, T serves as both keys and values
        - Each spatial location attends to all frequency ranks
        - Output MLP maps attended features to RGB residuals
        - Residual connection preserves query information
    """
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, out_ch: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for multi-head Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Normalization
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        # MLP to map concatenated [Q, attended] → RGB residual
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, out_ch),
        )

        self.drop = nn.Dropout(dropout)

        # Interpretability
        self.interpretability_enabled = False
        self.attention_history, self.temperature_history = [], []
        self.query_statistics, self.token_activation_patterns = [], []

    def forward(self, Q: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Apply cross-attention between queries and rank tokens
        
        Process:
        1. Normalize query and token inputs
        2. Compute attention scores with temperature scaling
        3. Apply attention to get weighted token combinations
        4. Combine with residual connection and generate RGB corrections
        
        Args:
            Q (torch.Tensor): Query vectors [B,Nq,D] from QueryBuilder
            T (torch.Tensor): Rank tokens [B,K,D] from RankTokenBank
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - resid_vec: RGB residuals [B,Nq,3] for each spatial location
                - attn_w: Attention weights [B,Nq,K] showing token importance
        """
        B, Nq, D = Q.shape
        K = T.size(1)

        # Normalize
        Qn, Tn = self.norm_q(Q), self.norm_kv(T)

        # Project to multi-head
        Qh = self.q_proj(Qn).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,Nq,Dh]
        Kh = self.k_proj(Tn).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,K,Dh]
        Vh = self.v_proj(Tn).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)   # [B,H,K,Dh]

        # Attention scores
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,H,Nq,K]
        scores = scores / torch.clamp(self.temperature, min=0.1, max=5.0)
        attn_w = torch.softmax(scores, dim=-1)  # [B,H,Nq,K]

        # Weighted sum of values
        attn_out = torch.matmul(attn_w, Vh)  # [B,H,Nq,Dh]

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, D)
        attn_out = self.out_proj(attn_out)

        # Residual + normalization
        attn_out = self.norm_out(attn_out + Qn)

        # Concatenate Q + attended
        H = torch.cat([Q, attn_out], dim=-1)  # [B,Nq,2D]

        # Predict residual
        resid_vec = self.mlp(H)  # [B,Nq,out_ch]

        # Average attention weights across heads for interpretability
        attn_mean = attn_w.mean(dim=1)  # [B,Nq,K]

        if self.interpretability_enabled:
            self._store_interpretability_data(Q, T, attn_mean, scores, resid_vec)

        return self.drop(resid_vec), attn_mean

    # --- interpretability utilities (same as your version) ---
    def enable_interpretability(self, enabled: bool = True):
        self.interpretability_enabled = enabled
        if not enabled:
            self.clear_interpretability_data()

    def clear_interpretability_data(self):
        self.attention_history.clear()
        self.temperature_history.clear()
        self.query_statistics.clear()
        self.token_activation_patterns.clear()

    def _store_interpretability_data(self, Q, T, attn_w, scores, resid_vec):
        with torch.no_grad():
            self.attention_history.append({
                'attention_weights': attn_w.detach().cpu(),
                'raw_scores': scores.detach().cpu(),
                'temperature': self.temperature.item()
            })
            self.query_statistics.append({
                'norm_mean': torch.norm(Q, dim=-1).mean().item(),
                'activation_mean': Q.mean().item(),
                'activation_std': Q.std().item()
            })
            token_norms = torch.norm(T, dim=-1).mean(dim=0).detach().cpu()
            token_means = T.mean(dim=(0, 1)).detach().cpu()
            self.token_activation_patterns.append({
                'token_norms': token_norms,
                'token_means': token_means,
                'residual_magnitude': torch.norm(resid_vec, dim=-1).mean().item()
            })


    def get_attention_entropy(self) -> torch.Tensor:
        """Compute attention entropy for the last forward pass."""
        if not self.attention_history:
            return torch.tensor(0.0)

        last_attention = self.attention_history[-1]['attention_weights']
        entropy = -(last_attention * torch.log(last_attention + 1e-8)).sum(dim=-1)
        return entropy

    def get_token_specialization_matrix(self) -> torch.Tensor:
        """Compute token specialization matrix from attention history."""
        if not self.attention_history:
            return torch.tensor([[]])

        # Use last attention weights
        attn_w = self.attention_history[-1]['attention_weights']  # [B, Nq, K]
        B, Nq, K = attn_w.shape

        # Compute co-activation matrix
        specialization = torch.zeros(K, K)
        for b in range(B):
            attn_batch = attn_w[b]  # [Nq, K]
            for k1 in range(K):
                for k2 in range(K):
                    # How much do tokens k1 and k2 co-activate?
                    correlation = torch.corrcoef(torch.stack([attn_batch[:, k1], attn_batch[:, k2]]))[0, 1]
                    if not torch.isnan(correlation):
                        specialization[k1, k2] += correlation.item()

        return specialization / B

    def analyze_attention_patterns(self):
        """Analyze attention patterns and return summary statistics."""
        if not self.attention_history:
            return {}

        stats = {}

        # Temperature evolution
        temps = [h['temperature'] for h in self.attention_history]
        stats['temperature'] = {
            'current': temps[-1],
            'mean': np.mean(temps),
            'std': np.std(temps),
            'trend': 'increasing' if temps[-1] > temps[0] else 'decreasing'
        }

        # Attention diversity
        last_attn = self.attention_history[-1]['attention_weights']
        entropy = self.get_attention_entropy().mean().item()
        max_entropy = np.log(last_attn.shape[-1])

        stats['attention_diversity'] = {
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
            'max_attention': last_attn.max().item(),
            'min_attention': last_attn.min().item()
        }

        # Query evolution
        if self.query_statistics:
            query_norms = [q['norm_mean'] for q in self.query_statistics]
            stats['query_evolution'] = {
                'norm_trend': 'increasing' if query_norms[-1] > query_norms[0] else 'decreasing',
                'current_norm': query_norms[-1],
                'norm_stability': np.std(query_norms[-10:]) if len(query_norms) >= 10 else np.std(query_norms)
            }

        return stats