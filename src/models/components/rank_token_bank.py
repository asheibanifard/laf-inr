#!/usr/bin/env python3
"""
Rank Token Bank: Learnable tokens representing different frequency ranks with FiLM conditioning.

This module provides the RankTokenBank class which maintains a bank of learnable tokens,
each representing a different spatial frequency rank. Uses Feature-wise Linear Modulation (FiLM)
to condition tokens based on input statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RankTokenBank(nn.Module):
    """
    Rank Token Bank: Learnable tokens representing different frequency ranks with FiLM conditioning.
    
    🎯 ANALOGY: Like a team of specialized art restoration experts:
    - Token 1: "Global structure expert" (handles overall composition)
    - Token 2: "Large feature expert" (handles major objects) 
    - Token 3-8: "Medium detail experts" (handle textures, patterns)
    - Token 9-16: "Fine detail experts" (handle edges, small features)
    Each expert gets personalized briefing (FiLM) based on the image's "medical chart"
    
    Maintains a bank of learnable tokens, each representing a different spatial frequency rank.
    Uses Feature-wise Linear Modulation (FiLM) to condition tokens based on input statistics.
    
    Args:
        num_tokens (int): Number of rank tokens (K)
        d_model (int): Token embedding dimension (D)
        rank_start (int): Starting rank number for token indexing
        cond_dim (int): Dimension of conditioning vector from TokenConditioner
    
    Input:
        B (int): Batch size
        cond: torch.Tensor [B, Cc] - Conditioning vector from TokenConditioner
              Cc = conditioning dimension (typically 4 + id_dim)
    
    Output:
        torch.Tensor [B, K, D] - Conditioned rank tokens
        B = batch size, K = num_tokens, D = d_model
        Each token represents a spatial frequency rank modulated by input conditions
    
    Token Organization:
        - Token i represents spatial rank (rank_start + i)
        - Lower ranks capture global structure, higher ranks capture details
        - FiLM modulation: token' = token * (1 + γ) + β, where γ,β depend on input
    """
    def __init__(self, num_tokens: int, d_model: int, rank_start: int, cond_dim: int):
        super().__init__()
        # Store key parameters for token bank configuration
        self.num_tokens, self.d_model, self.rank_start = num_tokens, d_model, rank_start
        
        # Initialize base token parameters with small random values
        # Each token represents a different spatial frequency rank
        self.base_tokens = nn.Parameter(torch.randn(num_tokens, d_model)*0.02)    # [K,D] base tokens
        
        # Additional rank-specific embeddings to differentiate tokens by frequency
        self.rank_emb = nn.Parameter(torch.randn(num_tokens, d_model)*0.01)       # [K,D] rank embeddings
        
        # SpectralTokenEncoder: Encodes (σ, u, v) spectral components into tokens
        # Based on the recommendation to properly encode spectral decomposition components
        # TODO: This should eventually take spectral components (σ, u, v) instead of generic conditioning
        
        d_tok = d_model  # Token dimension
        
        # Spectral component encoders as recommended:
        # u_mlp: Linear(h) → d/2, v_mlp: Linear(w) → d/2, sigma_mlp: Linear(1) → d/4
        # Use better dimensions (64-128) as recommended instead of low 16
        
        # IMPORTANT: These dimensions MUST match the actual resampling length
        # Each singular vector uk ∈ R^h, vk ∈ R^w must be resampled to length 128
        # Use adaptive average pooling 1D: F.adaptive_avg_pool1d(uk.unsqueeze(0), 128).squeeze(0)
        h_dim = 128  # u component dimension after resampling via adaptive_avg_pool1d
        w_dim = 128  # v component dimension after resampling via adaptive_avg_pool1d
        
        self.u_mlp = nn.Sequential(
            nn.Linear(h_dim, d_tok//2),
            nn.GELU(),
            nn.Linear(d_tok//2, d_tok//2)
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(w_dim, d_tok//2), 
            nn.GELU(),
            nn.Linear(d_tok//2, d_tok//2)
        )
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, d_tok//4),
            nn.GELU(),
            nn.Linear(d_tok//4, d_tok//4)
        )
        
        # Fusion layer to combine u, v, sigma encodings into final token
        # Verified: 96 (u) + 96 (v) + 48 (σ) = 240 for d_tok=192
        fusion_dim = d_tok//2 + d_tok//2 + d_tok//4  # d/2 + d/2 + d/4 = 5d/4 = 240
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_tok),
            nn.GELU(),
            nn.Linear(d_tok, d_tok)
        )
        
        # Token normalization as recommended
        self.token_norm = nn.LayerNorm(d_tok)
        
        # Proper per-token-channel FiLM conditioning (γ,β per token channel)
        # Instead of flattening to 2*d_model*num_tokens, use smaller per-channel conditioning
        self.film_gamma = nn.Sequential(
            nn.Linear(cond_dim, d_tok),                                           # Condition to token dimension
            nn.GELU(),
            nn.Linear(d_tok, d_tok)                                               # Per-channel gamma
        )
        self.film_beta = nn.Sequential(
            nn.Linear(cond_dim, d_tok),                                           # Condition to token dimension  
            nn.GELU(),
            nn.Linear(d_tok, d_tok)                                               # Per-channel beta
        )
        
        # Initialize FiLM networks for better conditioning responsiveness
        nn.init.zeros_(self.film_gamma[-1].bias)                                  # Zero bias initialization
        nn.init.normal_(self.film_gamma[-1].weight, std=5e-3)                    # Small weight initialization
        nn.init.zeros_(self.film_beta[-1].bias)                                   # Zero bias initialization
        nn.init.normal_(self.film_beta[-1].weight, std=5e-3)                     # Small weight initialization

        # Interpretability features
        self.interpretability_enabled = False
        self.token_evolution = []
        self.conditioning_effects = []
        self.spectral_analysis = []

    def forward(self, B: int, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Generate spectral tokens from (σ, u, v) components
        
        Process:
        1. Combine base tokens with rank embeddings
        2. For each token rank, encode the spectral components (σ, u, v)
        3. Apply normalization to ensure stable token representations
        
        Args:
            B (int): Batch size
            cond (torch.Tensor): Conditioning vector [B,Cc] from TokenConditioner
                                TODO: Should be replaced with (sigma, u, v) components
            
        Returns:
            torch.Tensor: Token bank [B, T, d] - properly shaped for MHA
            B = batch size, T = num_tokens, d = d_model
        """
        # Combine base tokens with rank-specific embeddings and expand for batch
        base = (self.base_tokens + self.rank_emb).unsqueeze(0).expand(B,-1,-1)    # [B,T,d] base token features
        
        # TODO: Replace this section with proper spectral encoding once (σ, u, v) input is available
        # For now, use existing FiLM approach but keep proper [B, T, d] shape
        
        # Generate placeholder spectral components from conditioning
        # This is temporary until proper SVD components are passed
        sigma_placeholder = torch.ones(B, self.num_tokens, 1, device=cond.device)   # [B, T, 1]
        u_placeholder = torch.randn(B, self.num_tokens, 128, device=cond.device)    # [B, T, h_dim]
        v_placeholder = torch.randn(B, self.num_tokens, 128, device=cond.device)    # [B, T, w_dim]
        
        # Process each token's spectral components
        tokens = []
        for t in range(self.num_tokens):
            # Encode sigma, u, v components for this token
            sigma_enc = self.sigma_mlp(sigma_placeholder[:, t:t+1, :])  # [B, 1, d/4]
            u_enc = self.u_mlp(u_placeholder[:, t:t+1, :])              # [B, 1, d/2]  
            v_enc = self.v_mlp(v_placeholder[:, t:t+1, :])              # [B, 1, d/2]
            
            # Fuse spectral encodings
            spectral_features = torch.cat([u_enc, v_enc, sigma_enc], dim=-1)  # [B, 1, 5d/4]
            token_features = self.fusion(spectral_features)                   # [B, 1, d]
            
            # Apply normalization as recommended
            token_features = self.token_norm(token_features)                  # [B, 1, d]
            
            tokens.append(token_features)
        
        # Stack all tokens: [B, T, d] - proper shape for MHA as recommended
        spectral_tokens = torch.cat(tokens, dim=1)  # [B, T, d]
        
        # Combine with base tokens (additive residual)
        final_tokens = base + spectral_tokens  # [B, T, d]
        
        # Proper per-token-channel FiLM conditioning (γ,β per token channel)
        # This avoids flattening and maintains [B, T, d] structure
        # Note: Using global statistics conditioning appropriate for luminance SVD
        # For RGB SVD tokens, consider adding channel embeddings to conditioning
        gamma = self.film_gamma(cond).unsqueeze(1)                               # [B, 1, d] - broadcast across tokens
        beta = self.film_beta(cond).unsqueeze(1)                                 # [B, 1, d] - broadcast across tokens
        
        # Apply FiLM modulation per channel, maintaining proper [B, T, d] shape
        conditioned_tokens = final_tokens * (1 + gamma) + beta                   # [B,T,d] conditioned tokens

        # Store interpretability data if enabled
        if self.interpretability_enabled:
            self._store_interpretability_data(base, spectral_tokens, final_tokens,
                                             conditioned_tokens, gamma, beta, cond)

        return conditioned_tokens                                                 # [B,T,d] - proper token bank shape

    def enable_interpretability(self, enabled: bool = True):
        """Enable/disable interpretability data collection."""
        self.interpretability_enabled = enabled
        if not enabled:
            self.clear_interpretability_data()

    def clear_interpretability_data(self):
        """Clear stored interpretability data."""
        self.token_evolution.clear()
        self.conditioning_effects.clear()
        self.spectral_analysis.clear()

    def _store_interpretability_data(self, base: torch.Tensor, spectral: torch.Tensor,
                                   final: torch.Tensor, conditioned: torch.Tensor,
                                   gamma: torch.Tensor, beta: torch.Tensor,
                                   conditioning: torch.Tensor):
        """Store data for interpretability analysis."""
        with torch.no_grad():
            B, T, D = conditioned.shape

            # Token evolution metrics
            token_norms = torch.norm(conditioned, dim=-1).mean(dim=0)  # [T] - average norm per token
            token_diversity = torch.var(conditioned, dim=-1).mean(dim=0)  # [T] - variance per token
            inter_token_similarity = torch.zeros(T, T)

            # Compute token similarity matrix
            for b in range(B):
                tokens_b = conditioned[b]  # [T, D]
                tokens_norm = F.normalize(tokens_b, p=2, dim=1)
                similarity = torch.mm(tokens_norm, tokens_norm.t())
                inter_token_similarity += similarity.cpu()
            inter_token_similarity /= B

            self.token_evolution.append({
                'token_norms': token_norms.cpu(),
                'token_diversity': token_diversity.cpu(),
                'similarity_matrix': inter_token_similarity,
                'average_norm': token_norms.mean().item(),
                'norm_variance': token_norms.var().item()
            })

            # Conditioning effects
            base_norms = torch.norm(base, dim=-1).mean(dim=0)  # [T]
            conditioning_delta = torch.norm(conditioned - final, dim=-1).mean(dim=0)  # [T]

            gamma_stats = {
                'mean': gamma.mean().item(),
                'std': gamma.std().item(),
                'min': gamma.min().item(),
                'max': gamma.max().item()
            }

            beta_stats = {
                'mean': beta.mean().item(),
                'std': beta.std().item(),
                'min': beta.min().item(),
                'max': beta.max().item()
            }

            self.conditioning_effects.append({
                'base_norms': base_norms.cpu(),
                'conditioning_delta': conditioning_delta.cpu(),
                'gamma_stats': gamma_stats,
                'beta_stats': beta_stats,
                'conditioning_magnitude': torch.norm(conditioning, dim=-1).mean().item()
            })

            # Spectral analysis (placeholder for future SVD analysis)
            spectral_norms = torch.norm(spectral, dim=-1).mean(dim=0)  # [T]
            spectral_contribution = spectral_norms / (base_norms + spectral_norms + 1e-8)

            self.spectral_analysis.append({
                'spectral_norms': spectral_norms.cpu(),
                'spectral_contribution_ratio': spectral_contribution.cpu(),
                'rank_specialization': self._analyze_rank_specialization(conditioned)
            })

    def _analyze_rank_specialization(self, tokens: torch.Tensor) -> dict:
        """Analyze how well tokens specialize for different frequency ranks."""
        B, T, D = tokens.shape

        # Compute token activation patterns
        token_activations = tokens.abs().mean(dim=(0, 2))  # [T] - mean absolute activation per token

        # Compute specialization index (how much each token differs from others)
        specialization_scores = []
        for t in range(T):
            token_t = tokens[:, t, :].flatten()  # [B*D]
            other_tokens = torch.cat([tokens[:, :t, :], tokens[:, t+1:, :]], dim=1)  # [B, T-1, D]
            other_flat = other_tokens.flatten()  # [B*(T-1)*D]

            # Compute KL divergence as specialization measure (simplified)
            token_hist = torch.histc(token_t, bins=50, min=-3, max=3)
            other_hist = torch.histc(other_flat, bins=50, min=-3, max=3)

            token_hist = token_hist / (token_hist.sum() + 1e-8)
            other_hist = other_hist / (other_hist.sum() + 1e-8)

            kl_div = (token_hist * torch.log((token_hist + 1e-8) / (other_hist + 1e-8))).sum()
            specialization_scores.append(kl_div.item())

        return {
            'token_activations': token_activations.cpu(),
            'specialization_scores': torch.tensor(specialization_scores),
            'most_specialized_token': int(torch.tensor(specialization_scores).argmax()),
            'least_specialized_token': int(torch.tensor(specialization_scores).argmin())
        }

    def analyze_token_dynamics(self):
        """Analyze token dynamics and specialization over time."""
        if not self.token_evolution:
            return {}

        import numpy as np

        # Analyze norm evolution
        norm_history = [te['average_norm'] for te in self.token_evolution]
        norm_variance_history = [te['norm_variance'] for te in self.token_evolution]

        # Analyze conditioning effects
        conditioning_magnitudes = [ce['conditioning_magnitude'] for ce in self.conditioning_effects]
        gamma_means = [ce['gamma_stats']['mean'] for ce in self.conditioning_effects]
        beta_means = [ce['beta_stats']['mean'] for ce in self.conditioning_effects]

        return {
            'norm_evolution': {
                'trend': 'increasing' if norm_history[-1] > norm_history[0] else 'decreasing',
                'stability': np.std(norm_history[-10:]) if len(norm_history) >= 10 else np.std(norm_history),
                'current_value': norm_history[-1]
            },
            'conditioning_responsiveness': {
                'gamma_trend': 'increasing' if gamma_means[-1] > gamma_means[0] else 'decreasing',
                'beta_trend': 'increasing' if beta_means[-1] > beta_means[0] else 'decreasing',
                'conditioning_strength': np.mean(conditioning_magnitudes),
                'adaptation_rate': np.std(gamma_means) + np.std(beta_means)
            },
            'specialization': {
                'diversity_trend': 'increasing' if norm_variance_history[-1] > norm_variance_history[0] else 'decreasing',
                'current_diversity': norm_variance_history[-1],
                'specialization_stability': np.std(norm_variance_history)
            }
        }

    def get_token_similarity_evolution(self):
        """Get evolution of token similarity patterns."""
        if len(self.token_evolution) < 2:
            return {}

        first_sim = self.token_evolution[0]['similarity_matrix']
        last_sim = self.token_evolution[-1]['similarity_matrix']

        similarity_change = torch.abs(last_sim - first_sim).mean().item()

        return {
            'similarity_change': similarity_change,
            'current_max_similarity': last_sim.max().item(),
            'current_min_similarity': last_sim.min().item(),
            'trend': 'more_similar' if last_sim.mean() > first_sim.mean() else 'more_diverse'
        }

    def get_rank_for_token(self, idx: int) -> int:
        """
        Get the spatial frequency rank for a given token index
        
        Args:
            idx (int): Token index (0 to num_tokens-1)
            
        Returns:
            int: Spatial frequency rank (rank_start + idx)
        """
        return self.rank_start + idx