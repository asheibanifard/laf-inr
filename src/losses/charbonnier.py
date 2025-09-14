#!/usr/bin/env python3
import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss: Robust alternative to L2 loss with reduced sensitivity to outliers.
    
    🎯 ANALOGY: Like a "forgiving ruler" for measuring errors:
    - Regular MSE: Harshly punishes big mistakes (squares them), gentle on small ones
    - Charbonnier: More forgiving on big mistakes, still cares about small ones  
    - Like a teacher who doesn't give zero for one bad answer, but still corrects it
    - Better for images because extreme pixel errors shouldn't dominate the loss
    
    Computes: mean(sqrt((x-y)² + ε²)) where ε prevents division by zero.
    More robust than MSE for handling outliers and provides smoother gradients.
    
    Args:
        eps (float): Small constant to prevent division by zero and add robustness
    
    Input:
        x: torch.Tensor [...] - Predicted values
        y: torch.Tensor [...] - Target values (same shape as x)
    
    Output:
        torch.Tensor scalar - Charbonnier loss value
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps=eps
    
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x-y)**2 + self.eps**2))