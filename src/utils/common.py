#!/usr/bin/env python3
import torch
import numpy as np


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducible experiments.
    
    Args:
        seed (int): Random seed value for all random number generators
    """
    # Set PyTorch random seed for CPU operations (affects random tensor generation)
    torch.manual_seed(seed)
    
    # Set PyTorch random seed for all CUDA devices (affects GPU random operations)
    torch.cuda.manual_seed_all(seed)
    
    # Set NumPy random seed (affects numpy.random operations)
    np.random.seed(seed)