#!/usr/bin/env python3
"""
Query Builder: Combines positional encoding with local image features to create queries.

This module provides the QueryBuilder class which takes Fourier positional encoding 
and local scalar features (luminance, gradients, Laplacian) and maps them to query 
vectors for cross-attention with rank tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryBuilder(nn.Module):
    """
    Query Builder: Combines positional encoding with local image features to create queries.
    
    🎯 ANALOGY: Like creating detailed "help requests" for each pixel:
    - Position info: "I'm at coordinates (x,y) with this GPS signature"
    - Local context: "Around me I see this brightness, these edge patterns, this texture"
    - Each pixel sends a unique help request: "Given where I am and what I see, what should my RGB values be?"
    - These requests go to the expert team (rank tokens) for answers
    
    Takes Fourier positional encoding and local scalar features (luminance, gradients, Laplacian)
    and maps them to query vectors for cross-attention with rank tokens.
    
    Args:
        pe_dim (int): Dimension of positional encoding features
        d_model (int): Output query dimension (must match token dimension)
        extra_scalars (int): Number of additional scalar features per pixel
        dropout (float): Dropout probability for regularization
    
    Input:
        pe: torch.Tensor [B, H, W, Pe] - Positional encoding features
            Pe = positional encoding dimension (typically 4*num_freqs)
        scalars: torch.Tensor [B, S, H, W] or [B, H, W, S] - Local scalar features
                S = extra_scalars, typically 4 for [luma, |grad_x|, |grad_y|, |laplacian|]
    
    Output:
        torch.Tensor [B, H*W, D] - Query vectors for each spatial location
        B = batch size, H*W = number of spatial locations, D = d_model
        Each spatial location gets a query vector combining position and local features
    """
    def __init__(self, pe_dim: int, d_model: int, extra_scalars: int, dropout: float, canvas_dim: int = 3):
        super().__init__()
        # Store canvas dimension for input validation
        self.canvas_dim = canvas_dim
        
        # Multi-layer perceptron to map combined features to query vectors
        # Input: concatenated FourierPE(x,y) + I^(1)↑(x,y) + local scalar features
        # Output: query vectors for cross-attention with rank tokens
        # Updated input size as recommended: PE_dim + canvas_dim + extra scalars
        # Breakdown: PE_dim=56 (fourier_K=14*4) + canvas_dim=3 (RGB) + extra_scalars=4 (Y,dx,dy,lap) = 63
        total_input_dim = pe_dim + canvas_dim + extra_scalars
        
        self.mlp = nn.Sequential(
            # First layer: map input features to model dimension
            nn.Linear(total_input_dim, d_model),                                  # Combined features → D
            nn.GELU(),                                                            # Non-linear activation
            nn.LayerNorm(d_model),                                                # Layer normalization for stability
            nn.Dropout(dropout),                                                  # Dropout for regularization
            # Second layer: refine query representations
            nn.Linear(d_model, d_model),                                          # Query refinement layer
        )

        # Interpretability features
        self.interpretability_enabled = False
        self.feature_statistics = []
        self.query_evolution = []
        self.component_contributions = []

    def enable_interpretability(self, enabled: bool = True):
        """Enable/disable interpretability data collection."""
        self.interpretability_enabled = enabled
        if not enabled:
            self.clear_interpretability_data()

    def clear_interpretability_data(self):
        """Clear stored interpretability data."""
        self.feature_statistics.clear()
        self.query_evolution.clear()
        self.component_contributions.clear()

    def forward(self, pe: torch.Tensor, scalars: torch.Tensor, canvas: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: Combine FourierPE(x,y) + I^(1)↑(x,y) + local features to create queries
        
        Process:
        1. Include canvas sample I^(1)↑(x,y) as recommended  
        2. Ensure all features have correct spatial dimensions
        3. Concatenate FourierPE + canvas + scalar features
        4. Apply MLP to generate query vectors for each spatial location
        
        Args:
            pe (torch.Tensor): Positional encoding [B,H,W,Pe] 
            scalars (torch.Tensor): Local features [B,S,H,W] or [B,H,W,S]
            canvas (torch.Tensor): Upsampled canvas I^(1)↑(x,y) [B,C,H,W] or [B,H,W,C]
                                  Should be the low-res input upsampled to HR size
            
        Returns:
            torch.Tensor: Query vectors [B,H*W,D] for cross-attention including canvas
        """
        # Handle different scalar tensor formats - ensure [B,H,W,S] format
        if scalars.dim()==4 and scalars.shape[1] != pe.shape[-1]:
            # If scalars are in [B,S,H,W] format, transpose to [B,H,W,S]
            scalars = scalars.permute(0,2,3,1)                                    # [B,H,W,S]
        
        # Handle canvas tensor format - ensure [B,H,W,C] format
        if canvas is not None:
            if canvas.dim()==4 and canvas.shape[1] == self.canvas_dim:
                # If canvas is in [B,C,H,W] format, transpose to [B,H,W,C]
                canvas = canvas.permute(0,2,3,1)                                  # [B,H,W,C]
            
            # Concatenate FourierPE(x,y) + I^(1)↑(x,y) + local scalars as recommended
            x = torch.cat([pe, canvas, scalars], dim=-1)                          # [B,H,W,Pe+C+S]
        else:
            # Fallback: use old behavior if canvas not provided (during transition)
            x = torch.cat([pe, scalars], dim=-1)                                  # [B,H,W,Pe+S]
            
        # Flatten spatial dimensions and apply MLP to create queries
        B,H,W,_ = x.shape
        queries = self.mlp(x.view(B, H*W, -1))                                       # [B,H*W,D] query vectors

        # Store interpretability data if enabled
        if self.interpretability_enabled:
            self._store_interpretability_data(pe, scalars, canvas, x, queries, B, H, W)

        return queries

    def _store_interpretability_data(self, pe: torch.Tensor, scalars: torch.Tensor,
                                   canvas: torch.Tensor, combined_features: torch.Tensor,
                                   queries: torch.Tensor, B: int, H: int, W: int):
        """Store data for interpretability analysis."""
        with torch.no_grad():
            # Feature component statistics
            pe_stats = {
                'mean': pe.mean().item(),
                'std': pe.std().item(),
                'min': pe.min().item(),
                'max': pe.max().item()
            }

            scalar_stats = {
                'mean': scalars.mean().item(),
                'std': scalars.std().item(),
                'min': scalars.min().item(),
                'max': scalars.max().item()
            }

            canvas_stats = None
            if canvas is not None:
                canvas_stats = {
                    'mean': canvas.mean().item(),
                    'std': canvas.std().item(),
                    'min': canvas.min().item(),
                    'max': canvas.max().item()
                }

            self.feature_statistics.append({
                'pe': pe_stats,
                'scalars': scalar_stats,
                'canvas': canvas_stats
            })

            # Query evolution
            query_norms = torch.norm(queries, dim=-1).mean().item()
            query_mean = queries.mean().item()
            query_std = queries.std().item()

            self.query_evolution.append({
                'norm_mean': query_norms,
                'activation_mean': query_mean,
                'activation_std': query_std,
                'spatial_variance': queries.var(dim=1).mean().item()
            })

            # Component contributions (approximate via gradient magnitude)
            if hasattr(self, 'training') and self.training:
                # Compute approximate contribution via feature magnitude
                pe_contribution = torch.norm(pe.view(B, -1), dim=1).mean().item()
                scalar_contribution = torch.norm(scalars.reshape(B, -1), dim=1).mean().item()
                canvas_contribution = 0.0
                if canvas is not None:
                    canvas_contribution = torch.norm(canvas.reshape(B, -1), dim=1).mean().item()

                total_contribution = pe_contribution + scalar_contribution + canvas_contribution
                if total_contribution > 0:
                    self.component_contributions.append({
                        'pe_ratio': pe_contribution / total_contribution,
                        'scalar_ratio': scalar_contribution / total_contribution,
                        'canvas_ratio': canvas_contribution / total_contribution if canvas is not None else 0.0
                    })

    def analyze_feature_importance(self):
        """Analyze the importance of different feature components."""
        if not self.component_contributions:
            return {}

        import numpy as np
        contributions = self.component_contributions

        pe_ratios = [c['pe_ratio'] for c in contributions]
        scalar_ratios = [c['scalar_ratio'] for c in contributions]
        canvas_ratios = [c['canvas_ratio'] for c in contributions]

        return {
            'positional_encoding': {
                'mean_contribution': np.mean(pe_ratios),
                'std_contribution': np.std(pe_ratios),
                'trend': 'increasing' if pe_ratios[-1] > pe_ratios[0] else 'decreasing'
            },
            'scalar_features': {
                'mean_contribution': np.mean(scalar_ratios),
                'std_contribution': np.std(scalar_ratios),
                'trend': 'increasing' if scalar_ratios[-1] > scalar_ratios[0] else 'decreasing'
            },
            'canvas_features': {
                'mean_contribution': np.mean(canvas_ratios),
                'std_contribution': np.std(canvas_ratios),
                'trend': 'increasing' if canvas_ratios[-1] > canvas_ratios[0] else 'decreasing'
            }
        }

    def get_query_quality_metrics(self):
        """Get metrics about query quality and evolution."""
        if not self.query_evolution:
            return {}

        import numpy as np
        evolution = self.query_evolution

        norms = [e['norm_mean'] for e in evolution]
        means = [e['activation_mean'] for e in evolution]
        stds = [e['activation_std'] for e in evolution]
        variances = [e['spatial_variance'] for e in evolution]

        return {
            'norm_stability': np.std(norms[-10:]) if len(norms) >= 10 else np.std(norms),
            'activation_trend': 'increasing' if means[-1] > means[0] else 'decreasing',
            'current_norm': norms[-1],
            'spatial_diversity': np.mean(variances),
            'consistency_score': 1.0 / (1.0 + np.std(norms)) if norms else 0.0
        }