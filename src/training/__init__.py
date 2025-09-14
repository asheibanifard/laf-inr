"""
LAF-INR Training Module

This module provides comprehensive training functionality for LAF-INR models including:
- Main training functions for multi-image datasets
- Single image overfitting capabilities  
- Multi-image training workflows
- Utility functions for visualization and demonstration
- Demonstration functions showcasing unique INR capabilities

Usage:
    from src.training import train, overfit_single_image, train_multi_image
    from src.training.utils import save_comparison, demonstrate_arbitrary_resolution
"""

# Main training functions
from .trainer import (
    run_epoch,
    get_visualization_samples, 
    train,
    create_psnr_plot
)

# Single image training
from .single_image import overfit_single_image

# Multi-image training
from .multi_image import train_multi_image

# Utility and demonstration functions
from .utils import (
    save_comparison,
    demonstrate_arbitrary_resolution,
    demonstrate_partial_input_sr,
    demonstrate_continuous_scale,
    run_all_demonstrations
)

__all__ = [
    # Main training
    'run_epoch',
    'get_visualization_samples',
    'train',
    'create_psnr_plot',
    
    # Specialized training
    'overfit_single_image',
    'train_multi_image',
    
    # Utilities and demonstrations
    'save_comparison',
    'demonstrate_arbitrary_resolution', 
    'demonstrate_partial_input_sr',
    'demonstrate_continuous_scale',
    'run_all_demonstrations'
]