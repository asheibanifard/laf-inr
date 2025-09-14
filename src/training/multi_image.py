"""
Multi-image training for LAF-INR model.

This module implements training functionality for training LAF-INR on multiple images,
which leverages the model's ability to handle multiple image identities.
"""

import os

# Import required modules  
try:
    from .trainer import train
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('/home/armin/Documents/my_research_projects/Medical_AI/INR/laf-inr')
    from LAF_INR import train


def train_multi_image(dataset_path: str, hr_size: int, scale_factor: int, epochs: int,
                     lr: float, batch_size: int, batch_points: int, save_dir: str = "results", 
                     use_uncertainty: bool = False, resume_from: str = None, 
                     use_mixed_precision: bool = False, alpha_warmup_epochs: int = 15,
                     use_adaptive_sampling: bool = False, max_images: int = None, validate_every: int = 5):
    """
    Multi-image training with LAF-INR.
    
    This function trains the LAF-INR model on multiple images using the main training
    function with appropriate parameters for multi-image scenarios.
    
    Args:
        dataset_path (str): Path to directory containing training images
        hr_size (int): High resolution target size
        scale_factor (int): Downsampling factor for creating LR inputs
        epochs (int): Number of training epochs
        lr (float): Learning rate
        batch_size (int): Number of images per batch
        batch_points (int): Number of points to sample per image
        save_dir (str): Directory to save results
        use_uncertainty (bool): Whether to use uncertainty estimation
        resume_from (str, optional): Path to checkpoint to resume from
        use_mixed_precision (bool): Whether to use mixed precision training
        alpha_warmup_epochs (int): Number of epochs for alpha warmup
        use_adaptive_sampling (bool): Whether to use adaptive coordinate sampling
        max_images (int, optional): Maximum number of images to use from dataset
        validate_every (int): Validation frequency in epochs
        
    Returns:
        LAFINR: Trained model instance
    """
    print(f"🎯 Multi-Image Training: {dataset_path}")
    print(f"   HR Size: {hr_size}, Scale: {scale_factor}x, Batch: {batch_size}, Epochs: {epochs}")
    
    # Use the existing train function with updated parameters
    model = train(
        data_path=dataset_path,
        crop_size=hr_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        num_workers=4,
        amp=use_mixed_precision,
        save_path=os.path.join(save_dir, "multi_image_model.pth"),
        sr_factor=float(scale_factor) if scale_factor > 1 else None,
    )
    
    return model