#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    """Compute cosine similarity between tensors"""
    return F.cosine_similarity(a, b, dim=-1)