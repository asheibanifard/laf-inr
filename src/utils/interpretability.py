#!/usr/bin/env python3
"""
Interpretability Utilities for LAF-INR
======================================

This module provides comprehensive interpretability tools for analyzing LAF-INR model behavior,
including attention visualization, feature analysis, and activation mapping.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os

class InterpretabilityHook:
    """Base class for model interpretability hooks."""

    def __init__(self, save_dir: str = "interpretability_outputs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.activations = {}
        self.hooks = []

    def register_hook(self, module: torch.nn.Module, name: str):
        """Register a forward hook on a module."""
        def hook_fn(module, input, output):
            self.activations[name] = output.detach().cpu() if isinstance(output, torch.Tensor) else output

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save_tensor_as_image(self, tensor: torch.Tensor, filename: str, title: str = ""):
        """Save a tensor as an image with proper normalization."""
        if tensor.dim() > 2:
            # Handle multi-channel tensors
            if tensor.shape[0] == 1:  # Single channel
                tensor = tensor.squeeze(0)
            elif tensor.shape[0] == 3:  # RGB
                tensor = tensor.permute(1, 2, 0)
            else:  # Multiple channels - take first few
                tensor = tensor[:3].mean(dim=0)

        tensor_np = tensor.numpy()

        plt.figure(figsize=(8, 6))
        if len(tensor_np.shape) == 2:
            plt.imshow(tensor_np, cmap='viridis')
            plt.colorbar()
        else:
            plt.imshow(np.clip(tensor_np, 0, 1))

        plt.title(title)
        plt.axis('off')
        plt.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()


class AttentionVisualizer(InterpretabilityHook):
    """Visualizer for attention patterns in CrossAttentionRefiner."""

    def __init__(self, save_dir: str = "attention_analysis"):
        super().__init__(save_dir)

    def visualize_attention_patterns(self, attention_weights: torch.Tensor,
                                   queries: torch.Tensor = None, tokens: torch.Tensor = None,
                                   image_size: Tuple[int, int] = None, epoch: int = 0):
        """
        Visualize attention patterns comprehensively.

        Args:
            attention_weights: [B, Nq, K] attention weights from cross-attention
            queries: [B, Nq, D] query vectors (optional)
            tokens: [B, K, D] token vectors (optional)
            image_size: (H, W) spatial dimensions (optional, inferred if not provided)
            epoch: Current training epoch
        """
        B, Nq, K = attention_weights.shape

        # Infer image size if not provided
        if image_size is None:
            # Assume square image
            H = W = int(Nq ** 0.5)
            if H * W != Nq:
                # If not square, try common aspect ratios
                import math
                H = int(math.sqrt(Nq))
                W = Nq // H
                if H * W != Nq:
                    # Fallback to closest square
                    H = W = int(Nq ** 0.5)
        else:
            H, W = image_size

        # Reshape attention to spatial format
        attn_spatial = attention_weights.view(B, H, W, K)  # [B, H, W, K]

        for b in range(min(B, 2)):  # Visualize first 2 batches
            self._visualize_attention_maps(attn_spatial[b], epoch, b)

            # Only run token specialization if tokens are available
            if tokens is not None:
                self._visualize_token_specialization(attention_weights[b], tokens[b], epoch, b)
            else:
                self._visualize_token_specialization_basic(attention_weights[b], epoch, b)

            self._analyze_attention_entropy(attention_weights[b], epoch, b)
            self._visualize_spatial_attention_patterns(attn_spatial[b], epoch, b)

    def _visualize_attention_maps(self, attn_spatial: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize attention maps for each token."""
        H, W, K = attn_spatial.shape

        # Create a grid of attention maps
        cols = min(8, K)
        rows = (K + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for k in range(K):
            row, col = k // cols, k % cols
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]

            attn_map = attn_spatial[:, :, k].cpu().numpy()
            im = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
            ax.set_title(f'Token {k}\n(Rank {k+1})', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide unused subplots
        for k in range(K, rows * cols):
            row, col = k // cols, k % cols
            if rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')

        plt.suptitle(f'Attention Maps - Epoch {epoch}, Batch {batch_idx}', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'attention_maps_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_token_specialization_basic(self, attention_weights: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze token specialization without token vectors (basic version)."""
        Nq, K = attention_weights.shape

        # Compute token utilization (how much each token is used)
        token_usage = attention_weights.mean(dim=0)  # [K] average attention per token

        # Find most attended pixels for each token
        top_attended_per_token = []
        for k in range(K):
            _, top_indices = torch.topk(attention_weights[:, k], k=min(100, Nq//4))
            top_attended_per_token.append(top_indices)

        # Plot basic token analysis
        plt.figure(figsize=(15, 8))

        # Subplot 1: Token usage bar chart
        plt.subplot(2, 3, 1)
        plt.bar(range(K), token_usage.cpu().numpy())
        plt.xlabel('Token Index')
        plt.ylabel('Average Attention Weight')
        plt.title('Token Utilization')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Attention weight distribution
        plt.subplot(2, 3, 2)
        plt.hist(attention_weights.cpu().numpy().flatten(), bins=50, alpha=0.7)
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        plt.title('Attention Weight Distribution')
        plt.grid(True, alpha=0.3)

        # Subplot 3: Token dominance analysis
        plt.subplot(2, 3, 3)
        dominant_tokens = torch.argmax(attention_weights, dim=1)  # [Nq]
        dominance_counts = torch.bincount(dominant_tokens, minlength=K).float()
        plt.bar(range(K), dominance_counts.cpu().numpy())
        plt.xlabel('Token Index')
        plt.ylabel('Times Most Dominant')
        plt.title('Token Dominance Frequency')
        plt.grid(True, alpha=0.3)

        # Subplot 4: Attention variance per token
        plt.subplot(2, 3, 4)
        token_variance = attention_weights.var(dim=0)
        plt.bar(range(K), token_variance.cpu().numpy())
        plt.xlabel('Token Index')
        plt.ylabel('Attention Variance')
        plt.title('Token Attention Variance')
        plt.grid(True, alpha=0.3)

        # Subplot 5: Top-k attention per token
        plt.subplot(2, 3, 5)
        top_k_means = []
        for k in range(K):
            top_k = min(10, Nq//10)
            top_values, _ = torch.topk(attention_weights[:, k], k=top_k)
            top_k_means.append(top_values.mean().item())

        plt.bar(range(K), top_k_means)
        plt.xlabel('Token Index')
        plt.ylabel('Mean Top-10 Attention')
        plt.title('Token Peak Attention')
        plt.grid(True, alpha=0.3)

        # Subplot 6: Token correlation matrix
        plt.subplot(2, 3, 6)
        correlation_matrix = torch.corrcoef(attention_weights.t())  # [K, K]
        import seaborn as sns
        sns.heatmap(correlation_matrix.cpu().numpy(), annot=True, fmt='.2f',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=[f'T{i}' for i in range(K)],
                   yticklabels=[f'T{i}' for i in range(K)])
        plt.title('Token Attention Correlation')

        plt.suptitle(f'Basic Token Specialization - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'token_specialization_basic_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_token_specialization(self, attention_weights: torch.Tensor,
                                      tokens: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze which spatial regions each token specializes in."""
        Nq, K = attention_weights.shape

        # Compute token utilization (how much each token is used)
        token_usage = attention_weights.mean(dim=0)  # [K] average attention per token

        # Find most attended pixels for each token
        top_attended_per_token = []
        for k in range(K):
            _, top_indices = torch.topk(attention_weights[:, k], k=min(100, Nq//4))
            top_attended_per_token.append(top_indices)

        # Plot token usage distribution
        plt.figure(figsize=(12, 8))

        # Subplot 1: Token usage bar chart
        plt.subplot(2, 2, 1)
        plt.bar(range(K), token_usage.numpy())
        plt.xlabel('Token Index')
        plt.ylabel('Average Attention Weight')
        plt.title('Token Utilization')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Attention weight distribution
        plt.subplot(2, 2, 2)
        plt.hist(attention_weights.numpy().flatten(), bins=50, alpha=0.7)
        plt.xlabel('Attention Weight')
        plt.ylabel('Frequency')
        plt.title('Attention Weight Distribution')
        plt.grid(True, alpha=0.3)

        # Subplot 3: Token specialization heatmap
        plt.subplot(2, 2, (3, 4))
        # Create specialization matrix: for each pixel, which token has highest attention
        dominant_tokens = torch.argmax(attention_weights, dim=1)  # [Nq]
        specialization_matrix = torch.zeros(K, K)

        for k1 in range(K):
            for k2 in range(K):
                # How often does token k1 have highest attention when token k2 is also active?
                mask = (dominant_tokens == k1)
                if mask.sum() > 0:
                    specialization_matrix[k1, k2] = attention_weights[mask, k2].mean()

        sns.heatmap(specialization_matrix.numpy(), annot=True, fmt='.3f',
                   xticklabels=[f'T{i}' for i in range(K)],
                   yticklabels=[f'T{i}' for i in range(K)])
        plt.title('Token Co-activation Matrix')
        plt.xlabel('Token Index')
        plt.ylabel('Dominant Token')

        plt.suptitle(f'Token Specialization Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'token_specialization_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _analyze_attention_entropy(self, attention_weights: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze attention entropy patterns."""
        # Compute entropy per pixel
        entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1)  # [Nq]

        # Statistics
        mean_entropy = entropy.mean().item()
        max_entropy = torch.log(torch.tensor(attention_weights.shape[1], dtype=torch.float32)).item()
        normalized_entropy = entropy / max_entropy

        plt.figure(figsize=(15, 5))

        # Subplot 1: Entropy histogram
        plt.subplot(1, 3, 1)
        plt.hist(entropy.cpu().numpy(), bins=30, alpha=0.7, density=True)
        plt.axvline(mean_entropy, color='red', linestyle='--', label=f'Mean: {mean_entropy:.3f}')
        plt.axvline(max_entropy, color='green', linestyle='--', label=f'Max: {max_entropy:.3f}')
        plt.xlabel('Entropy')
        plt.ylabel('Density')
        plt.title('Attention Entropy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Entropy vs position
        plt.subplot(1, 3, 2)
        positions = torch.arange(len(entropy))
        plt.scatter(positions.cpu().numpy(), entropy.cpu().numpy(), alpha=0.5, s=1)
        plt.xlabel('Pixel Index')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Spatial Position')
        plt.grid(True, alpha=0.3)

        # Subplot 3: Normalized entropy
        plt.subplot(1, 3, 3)
        plt.hist(normalized_entropy.cpu().numpy(), bins=30, alpha=0.7, density=True)
        plt.xlabel('Normalized Entropy')
        plt.ylabel('Density')
        plt.title('Normalized Entropy (0=focused, 1=uniform)')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Attention Entropy Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'attention_entropy_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_spatial_attention_patterns(self, attn_spatial: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize spatial patterns in attention."""
        H, W, K = attn_spatial.shape

        # Compute attention statistics per spatial location
        attention_mean = attn_spatial.mean(dim=2)  # [H, W] average attention across tokens
        attention_std = attn_spatial.std(dim=2)    # [H, W] attention variance
        dominant_token = torch.argmax(attn_spatial, dim=2)  # [H, W] most attended token

        plt.figure(figsize=(15, 5))

        # Subplot 1: Mean attention
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(attention_mean.cpu().numpy(), cmap='viridis')
        plt.title('Mean Attention per Pixel')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.axis('off')

        # Subplot 2: Attention variance
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(attention_std.cpu().numpy(), cmap='plasma')
        plt.title('Attention Std per Pixel')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.axis('off')

        # Subplot 3: Dominant token map
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(dominant_token.cpu().numpy(), cmap='tab20', vmin=0, vmax=K-1)
        plt.title('Dominant Token per Pixel')
        plt.colorbar(im3, fraction=0.046, pad=0.04, ticks=range(K))
        plt.axis('off')

        plt.suptitle(f'Spatial Attention Patterns - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'spatial_attention_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()


class FeatureVisualizer(InterpretabilityHook):
    """Visualizer for feature representations in QueryBuilder."""

    def __init__(self, save_dir: str = "feature_analysis"):
        super().__init__(save_dir)

    def visualize_feature_components(self, pe_features: torch.Tensor,
                                   scalar_features: torch.Tensor,
                                   canvas_features: torch.Tensor,
                                   queries: torch.Tensor,
                                   image_size: Tuple[int, int],
                                   epoch: int = 0):
        """
        Visualize different feature components and their contributions.

        Args:
            pe_features: [B, H, W, Pe] Positional encoding features
            scalar_features: [B, 4, H, W] Scalar features [Y, |dx|, |dy|, |lap|]
            canvas_features: [B, 3, H, W] Canvas features
            queries: [B, H*W, D] Final query vectors
            image_size: (H, W) spatial dimensions
            epoch: Current epoch
        """
        B, H, W = image_size[0], image_size[1], image_size[1]  # Assuming square

        for b in range(min(B, 2)):
            self._visualize_scalar_features(scalar_features[b], epoch, b)
            self._visualize_positional_encoding(pe_features[b], epoch, b)
            self._visualize_canvas_features(canvas_features[b], epoch, b)
            self._analyze_query_statistics(queries[b].view(H, W, -1), epoch, b)

    def _visualize_scalar_features(self, scalar_features: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize scalar features: Y, |dx|, |dy|, |lap|."""
        feature_names = ['Luminance (Y)', 'Gradient X', 'Gradient Y', 'Laplacian']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i in range(4):
            feature_map = scalar_features[i].numpy()
            im = axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'{feature_names[i]}\nRange: [{feature_map.min():.3f}, {feature_map.max():.3f}]')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle(f'Scalar Features - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'scalar_features_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_positional_encoding(self, pe_features: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize positional encoding patterns."""
        H, W, Pe = pe_features.shape

        # Select a few frequency components to visualize
        num_freqs = Pe // 4  # 4 features per frequency
        freq_to_show = min(8, num_freqs)

        fig, axes = plt.subplots(2, freq_to_show, figsize=(3*freq_to_show, 6))
        if freq_to_show == 1:
            axes = axes.reshape(2, 1)

        for f in range(freq_to_show):
            # Show sin(x) and cos(x) components
            sin_x = pe_features[:, :, f*4].numpy()
            cos_x = pe_features[:, :, f*4+1].numpy()

            im1 = axes[0, f].imshow(sin_x, cmap='RdBu', vmin=-1, vmax=1)
            axes[0, f].set_title(f'sin(2^{f}πx)')
            axes[0, f].axis('off')
            plt.colorbar(im1, ax=axes[0, f], fraction=0.046, pad=0.04)

            im2 = axes[1, f].imshow(cos_x, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, f].set_title(f'cos(2^{f}πx)')
            axes[1, f].axis('off')
            plt.colorbar(im2, ax=axes[1, f], fraction=0.046, pad=0.04)

        plt.suptitle(f'Positional Encoding Components - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'positional_encoding_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_canvas_features(self, canvas_features: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize canvas (upsampled LR) features."""
        # Transpose from [3, H, W] to [H, W, 3] for visualization
        canvas_rgb = canvas_features.permute(1, 2, 0).numpy()

        plt.figure(figsize=(12, 4))

        # RGB channels separately
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(1, 4, i+1)
            plt.imshow(canvas_rgb[:, :, i], cmap='gray', vmin=0, vmax=1)
            plt.title(f'{channel_name} Channel')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)

        # Combined RGB
        plt.subplot(1, 4, 4)
        plt.imshow(np.clip(canvas_rgb, 0, 1))
        plt.title('RGB Composite')
        plt.axis('off')

        plt.suptitle(f'Canvas Features - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'canvas_features_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _analyze_query_statistics(self, queries_spatial: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze statistical properties of query vectors."""
        H, W, D = queries_spatial.shape

        # Compute statistics
        query_norms = torch.norm(queries_spatial, dim=2)  # [H, W]
        query_mean = queries_spatial.mean(dim=2)          # [H, W]
        query_std = queries_spatial.std(dim=2)            # [H, W]

        plt.figure(figsize=(15, 5))

        # Subplot 1: Query norms
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(query_norms.numpy(), cmap='viridis')
        plt.title(f'Query Vector Norms\nMean: {query_norms.mean():.3f}')
        plt.colorbar(im1, fraction=0.046, pad=0.04)
        plt.axis('off')

        # Subplot 2: Query means
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(query_mean.numpy(), cmap='RdBu')
        plt.title(f'Query Vector Means\nRange: [{query_mean.min():.3f}, {query_mean.max():.3f}]')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.axis('off')

        # Subplot 3: Query standard deviations
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(query_std.numpy(), cmap='plasma')
        plt.title(f'Query Vector Std\nMean: {query_std.mean():.3f}')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.suptitle(f'Query Statistics - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'query_statistics_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()


class TokenAnalyzer(InterpretabilityHook):
    """Analyzer for token behavior in RankTokenBank."""

    def __init__(self, save_dir: str = "token_analysis"):
        super().__init__(save_dir)

    def analyze_token_dynamics(self, tokens: torch.Tensor, conditioning: torch.Tensor,
                             base_tokens: torch.Tensor, rank_embeddings: torch.Tensor,
                             film_params: Dict[str, torch.Tensor], epoch: int = 0):
        """
        Analyze token dynamics and conditioning effects.

        Args:
            tokens: [B, K, D] Final conditioned tokens
            conditioning: [B, Cc] Conditioning vector
            base_tokens: [K, D] Base token parameters
            rank_embeddings: [K, D] Rank-specific embeddings
            film_params: Dict with 'gamma' and 'beta' parameters
            epoch: Current epoch
        """
        B, K, D = tokens.shape

        for b in range(min(B, 2)):
            self._visualize_token_conditioning_effects(
                tokens[b], conditioning[b], base_tokens,
                rank_embeddings, film_params, epoch, b
            )
            self._analyze_token_similarity_patterns(tokens[b], epoch, b)
            self._visualize_token_evolution(tokens[b], epoch, b)

    def _visualize_token_conditioning_effects(self, tokens: torch.Tensor, conditioning: torch.Tensor,
                                            base_tokens: torch.Tensor, rank_embeddings: torch.Tensor,
                                            film_params: Dict[str, torch.Tensor], epoch: int, batch_idx: int):
        """Visualize how conditioning affects tokens."""
        K, D = tokens.shape

        # Compute conditioning influence
        base_combined = base_tokens + rank_embeddings  # [K, D]
        conditioning_effect = tokens - base_combined   # [K, D]

        plt.figure(figsize=(15, 10))

        # Subplot 1: Token activation heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(tokens.numpy(), cmap='RdBu_r', center=0)
        plt.title('Final Token Activations')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Token Index')

        # Subplot 2: Base tokens
        plt.subplot(2, 3, 2)
        sns.heatmap(base_combined.numpy(), cmap='RdBu_r', center=0)
        plt.title('Base + Rank Tokens')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Token Index')

        # Subplot 3: Conditioning effect
        plt.subplot(2, 3, 3)
        sns.heatmap(conditioning_effect.numpy(), cmap='RdBu_r', center=0)
        plt.title('Conditioning Effect')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Token Index')

        # Subplot 4: FiLM parameters
        plt.subplot(2, 3, 4)
        if 'gamma' in film_params and 'beta' in film_params:
            gamma = film_params['gamma'][batch_idx]  # [D]
            beta = film_params['beta'][batch_idx]    # [D]

            x_dims = np.arange(len(gamma))
            plt.plot(x_dims, gamma.numpy(), label='γ (scale)', alpha=0.7)
            plt.plot(x_dims, beta.numpy(), label='β (shift)', alpha=0.7)
            plt.xlabel('Feature Dimension')
            plt.ylabel('FiLM Parameter Value')
            plt.title('FiLM Parameters')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Subplot 5: Token norms
        plt.subplot(2, 3, 5)
        token_norms = torch.norm(tokens, dim=1)
        base_norms = torch.norm(base_combined, dim=1)

        x_tokens = np.arange(K)
        plt.bar(x_tokens - 0.2, token_norms.numpy(), width=0.4, label='Final', alpha=0.7)
        plt.bar(x_tokens + 0.2, base_norms.numpy(), width=0.4, label='Base', alpha=0.7)
        plt.xlabel('Token Index')
        plt.ylabel('L2 Norm')
        plt.title('Token Norms')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 6: Conditioning vector
        plt.subplot(2, 3, 6)
        plt.bar(range(len(conditioning)), conditioning.numpy())
        plt.xlabel('Conditioning Dimension')
        plt.ylabel('Value')
        plt.title('Conditioning Vector')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Token Conditioning Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'token_conditioning_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _analyze_token_similarity_patterns(self, tokens: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze similarity patterns between tokens."""
        K, D = tokens.shape

        # Compute pairwise cosine similarities
        tokens_normalized = F.normalize(tokens, p=2, dim=1)
        similarity_matrix = torch.mm(tokens_normalized, tokens_normalized.t())

        plt.figure(figsize=(12, 5))

        # Subplot 1: Similarity matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(similarity_matrix.numpy(), annot=True, fmt='.2f',
                   xticklabels=[f'T{i}' for i in range(K)],
                   yticklabels=[f'T{i}' for i in range(K)],
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1)
        plt.title('Token Cosine Similarity Matrix')

        # Subplot 2: Similarity distribution
        plt.subplot(1, 2, 2)
        # Extract upper triangular values (excluding diagonal)
        triu_indices = torch.triu_indices(K, K, offset=1)
        similarities = similarity_matrix[triu_indices[0], triu_indices[1]]

        plt.hist(similarities.numpy(), bins=20, alpha=0.7, density=True)
        plt.axvline(similarities.mean().item(), color='red', linestyle='--',
                   label=f'Mean: {similarities.mean():.3f}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Token Similarity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Token Similarity Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'token_similarity_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_token_evolution(self, tokens: torch.Tensor, epoch: int, batch_idx: int):
        """Visualize token feature evolution patterns."""
        K, D = tokens.shape

        # Analyze token diversity and specialization
        token_means = tokens.mean(dim=1)  # [K] mean activation per token
        token_stds = tokens.std(dim=1)    # [K] std deviation per token
        token_max_features = torch.argmax(torch.abs(tokens), dim=1)  # [K] dominant feature per token

        plt.figure(figsize=(15, 5))

        # Subplot 1: Token statistics
        plt.subplot(1, 3, 1)
        x_tokens = np.arange(K)
        plt.errorbar(x_tokens, token_means.numpy(), yerr=token_stds.numpy(),
                    fmt='o-', capsize=3, capthick=1)
        plt.xlabel('Token Index')
        plt.ylabel('Mean ± Std Activation')
        plt.title('Token Activation Statistics')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Dominant features
        plt.subplot(1, 3, 2)
        plt.scatter(range(K), token_max_features.numpy(), s=100, alpha=0.7)
        plt.xlabel('Token Index')
        plt.ylabel('Dominant Feature Dimension')
        plt.title('Dominant Feature per Token')
        plt.grid(True, alpha=0.3)

        # Subplot 3: Feature utilization
        plt.subplot(1, 3, 3)
        feature_usage = torch.bincount(token_max_features, minlength=D)
        plt.bar(range(D), feature_usage.numpy())
        plt.xlabel('Feature Dimension')
        plt.ylabel('Usage Count')
        plt.title('Feature Dimension Utilization')
        plt.grid(True, alpha=0.3)

        plt.suptitle(f'Token Evolution Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'token_evolution_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()


class ResidualCompositionVisualizer(InterpretabilityHook):
    """Visualizer for residual composition in ResidualComposer."""

    def __init__(self, save_dir: str = "residual_analysis"):
        super().__init__(save_dir)

    def visualize_composition_process(self, base_image: torch.Tensor,
                                   residual: torch.Tensor, final_output: torch.Tensor,
                                   alpha: float, epoch: int = 0):
        """
        Visualize the residual composition process.

        Args:
            base_image: [B, 3, H, W] Base upsampled image
            residual: [B, 3, H, W] Predicted residual
            final_output: [B, 3, H, W] Final composed output
            alpha: Alpha blending factor used
            epoch: Current epoch
        """
        B = base_image.shape[0]

        for b in range(min(B, 2)):
            self._visualize_composition_components(
                base_image[b], residual[b], final_output[b], alpha, epoch, b
            )
            self._analyze_residual_patterns(residual[b], epoch, b)
            self._visualize_alpha_effects(base_image[b], residual[b], alpha, epoch, b)

    def _visualize_composition_components(self, base_image: torch.Tensor,
                                        residual: torch.Tensor, final_output: torch.Tensor,
                                        alpha: float, epoch: int, batch_idx: int):
        """Visualize all components of the composition process."""
        # Convert to numpy and transpose for visualization
        base_np = base_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        residual_np = residual.permute(1, 2, 0).cpu().numpy()
        final_np = final_output.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        plt.figure(figsize=(20, 12))

        # Row 1: RGB channels of base image
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(3, 5, i+1)
            plt.imshow(base_np[:, :, i], cmap='gray', vmin=0, vmax=1)
            plt.title(f'Base {channel}')
            plt.axis('off')

        # Base RGB composite
        plt.subplot(3, 5, 4)
        plt.imshow(base_np)
        plt.title('Base RGB')
        plt.axis('off')

        # Alpha value display
        plt.subplot(3, 5, 5)
        plt.text(0.5, 0.5, f'α = {alpha:.3f}', fontsize=20, ha='center', va='center',
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
        plt.title('Blending Factor')
        plt.axis('off')

        # Row 2: Residual channels
        residual_range = max(abs(residual_np.min()), abs(residual_np.max()))
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(3, 5, 6+i)
            plt.imshow(residual_np[:, :, i], cmap='RdBu', vmin=-residual_range, vmax=residual_range)
            plt.title(f'Residual {channel}')
            plt.axis('off')
            plt.colorbar(fraction=0.046, pad=0.04)

        # Residual magnitude
        plt.subplot(3, 5, 9)
        residual_mag = np.linalg.norm(residual_np, axis=2)
        plt.imshow(residual_mag, cmap='hot')
        plt.title('Residual Magnitude')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        # Composition formula
        plt.subplot(3, 5, 10)
        plt.text(0.5, 0.5, f'Output = α × Base + (1-α) × Residual',
                fontsize=12, ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round", facecolor='lightyellow'))
        plt.title('Composition Formula')
        plt.axis('off')

        # Row 3: Final output
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            plt.subplot(3, 5, 11+i)
            plt.imshow(final_np[:, :, i], cmap='gray', vmin=0, vmax=1)
            plt.title(f'Final {channel}')
            plt.axis('off')

        # Final RGB composite
        plt.subplot(3, 5, 14)
        plt.imshow(final_np)
        plt.title('Final RGB')
        plt.axis('off')

        # Difference from base
        plt.subplot(3, 5, 15)
        diff = final_np - base_np
        diff_range = max(abs(diff.min()), abs(diff.max()))
        plt.imshow(diff, cmap='RdBu', vmin=-diff_range, vmax=diff_range)
        plt.title('Final - Base')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.suptitle(f'Residual Composition Process - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'composition_process_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _analyze_residual_patterns(self, residual: torch.Tensor, epoch: int, batch_idx: int):
        """Analyze patterns in the residual signal."""
        residual_np = residual.permute(1, 2, 0).cpu().numpy()
        H, W, C = residual_np.shape

        # Statistics per channel
        channel_means = np.mean(residual_np, axis=(0, 1))
        channel_stds = np.std(residual_np, axis=(0, 1))
        channel_max = np.max(residual_np, axis=(0, 1))
        channel_min = np.min(residual_np, axis=(0, 1))

        plt.figure(figsize=(15, 10))

        # Subplot 1: Channel statistics
        plt.subplot(2, 3, 1)
        channels = ['Red', 'Green', 'Blue']
        x = np.arange(3)
        width = 0.2

        plt.bar(x - width, channel_means, width, label='Mean', alpha=0.7)
        plt.bar(x, channel_stds, width, label='Std', alpha=0.7)
        plt.bar(x + width, (channel_max - channel_min), width, label='Range', alpha=0.7)

        plt.xlabel('Channel')
        plt.ylabel('Value')
        plt.title('Residual Statistics by Channel')
        plt.xticks(x, channels)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Residual histograms
        plt.subplot(2, 3, 2)
        for c, color, name in zip(range(3), ['red', 'green', 'blue'], channels):
            plt.hist(residual_np[:, :, c].flatten(), bins=30, alpha=0.5,
                    label=name, color=color, density=True)
        plt.xlabel('Residual Value')
        plt.ylabel('Density')
        plt.title('Residual Value Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 3: Spatial frequency analysis (rough approximation)
        plt.subplot(2, 3, 3)
        residual_gray = np.mean(residual_np, axis=2)
        fft = np.fft.fft2(residual_gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)

        plt.imshow(magnitude_spectrum, cmap='hot')
        plt.title('Frequency Spectrum (Log Scale)')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        # Subplot 4: Residual energy map
        plt.subplot(2, 3, 4)
        energy_map = np.sum(residual_np**2, axis=2)
        plt.imshow(energy_map, cmap='plasma')
        plt.title('Residual Energy Map')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        # Subplot 5: Edge detection on residual
        plt.subplot(2, 3, 5)
        from scipy import ndimage
        residual_gray = np.mean(residual_np, axis=2)
        edges = ndimage.sobel(residual_gray)
        plt.imshow(edges, cmap='gray')
        plt.title('Residual Edge Content')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

        # Subplot 6: Correlation between channels
        plt.subplot(2, 3, 6)
        correlations = np.corrcoef([residual_np[:, :, c].flatten() for c in range(3)])
        sns.heatmap(correlations, annot=True, fmt='.3f',
                   xticklabels=channels, yticklabels=channels,
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1)
        plt.title('Inter-channel Correlations')

        plt.suptitle(f'Residual Pattern Analysis - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'residual_patterns_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _visualize_alpha_effects(self, base_image: torch.Tensor, residual: torch.Tensor,
                               alpha: float, epoch: int, batch_idx: int):
        """Visualize the effects of different alpha values."""
        # Test different alpha values
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        fig, axes = plt.subplots(2, len(alpha_values), figsize=(4*len(alpha_values), 8))

        for i, test_alpha in enumerate(alpha_values):
            # Compute output with test alpha
            output = test_alpha * base_image + (1 - test_alpha) * residual
            output_clamped = torch.clamp(output, 0, 1)

            # Convert to numpy for visualization
            base_np = base_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            output_np = output_clamped.permute(1, 2, 0).cpu().numpy()

            # Show base and output
            axes[0, i].imshow(base_np)
            axes[0, i].set_title(f'Base (α={test_alpha:.2f})')
            axes[0, i].axis('off')

            axes[1, i].imshow(output_np)
            axes[1, i].set_title(f'Output (α={test_alpha:.2f})')
            axes[1, i].axis('off')

            # Highlight current alpha
            if abs(test_alpha - alpha) < 0.01:
                for ax in [axes[0, i], axes[1, i]]:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)

        plt.suptitle(f'Alpha Blending Effects - Current α={alpha:.3f} - Epoch {epoch}, Batch {batch_idx}')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'alpha_effects_epoch_{epoch}_batch_{batch_idx}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()


def create_interpretability_report(model, dataloader, device, save_dir: str = "interpretability_report"):
    """
    Generate a comprehensive interpretability report for LAF-INR model.

    Args:
        model: LAF-INR model instance
        dataloader: DataLoader for images to analyze
        device: Device to run analysis on
        save_dir: Directory to save analysis results
    """
    report_dir = Path(save_dir)
    report_dir.mkdir(exist_ok=True)

    # Initialize analyzers
    attention_viz = AttentionVisualizer(str(report_dir / "attention"))
    feature_viz = FeatureVisualizer(str(report_dir / "features"))
    token_analyzer = TokenAnalyzer(str(report_dir / "tokens"))
    residual_viz = ResidualCompositionVisualizer(str(report_dir / "residual"))

    model.eval()

    print("🔍 Generating interpretability report...")

    with torch.no_grad():
        for batch_idx, (lr_imgs, hr_imgs, img_ids) in enumerate(dataloader):
            if batch_idx >= 2:  # Limit to first 2 batches
                break

            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            img_ids = img_ids.to(device)

            # Get model outputs and intermediate activations
            outputs = model(lr_imgs, img_ids, target_size=hr_imgs.shape[-2:])

            # Extract relevant tensors for analysis
            pred = outputs["pred"]
            residual = outputs["residual"]
            attention_weights = outputs["attn"]
            canvas_base = outputs["canvas_base"]

            # Analyze attention patterns
            print(f"  📊 Analyzing attention patterns (batch {batch_idx})")
            attention_viz.visualize_attention_patterns(
                attention_weights, None, None, hr_imgs.shape[-2:], epoch=batch_idx
            )

            # Analyze residual composition
            print(f"  🎨 Analyzing residual composition (batch {batch_idx})")
            alpha = model.residual.composer.get_current_alpha()
            residual_viz.visualize_composition_process(
                canvas_base, residual, pred, alpha, epoch=batch_idx
            )

    print(f"✅ Interpretability report saved to: {report_dir}")
    return report_dir