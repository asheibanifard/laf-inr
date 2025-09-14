#!/usr/bin/env python3
import torch


def grad_l1(x):
    """
    Gradient L1 Loss: Computes L1 norm of image gradients for edge preservation.
    
    Input:
        x: torch.Tensor [B, C, H, W] - Input image tensor
    
    Output:
        torch.Tensor scalar - Average L1 norm of horizontal and vertical gradients
    """
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return 0.5*(dx.abs().mean() + dy.abs().mean())


def attn_entropy(attn, eps=1e-8):  # attn: [B,Nq,K]
    """
    Attention Entropy: Computes entropy of attention weights to encourage diversity.
    
    Input:
        attn: torch.Tensor [B, Nq, K] - Attention weights
              B = batch, Nq = number of queries, K = number of keys
        eps (float): Small constant for numerical stability
    
    Output:
        torch.Tensor scalar - Average entropy across all attention distributions
        Higher entropy encourages more diverse attention patterns
    """
    p = attn.clamp_min(eps)
    return (-(p * p.log()).sum(dim=-1)).mean()