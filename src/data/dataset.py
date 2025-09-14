#!/usr/bin/env python3
import os
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class CustomDataset(Dataset):
    """
    Custom Dataset: Creates clean LR-HR pairs using ONLY bicubic downsampling.
    
    🎯 CLEAN SUPER-RESOLUTION: No blur, noise, or other degradations!
    - HR: Original high-resolution image  
    - LR: Clean bicubic downsampling of HR (no additional artifacts)
    - Perfect for testing model's pure super-resolution capability
    
    Creates proper super-resolution pairs where LR is created by bicubic downsampling
    the HR image with no additional degradations. This provides a clean baseline
    for super-resolution training and evaluation.
    
    Args:
        data_path (str): Path to image directory
        crop_size (int): Size for HR crops (LR will be crop_size // sr_scale)  
        split (str): "train" or "val" dataset split
        sr_scale (float): Super-resolution scale factor (2.0 = 2x SR, 1.0 = restoration)
        
    Returns dictionary with:
        - 'lf': torch.Tensor [3, H_lr, W_lr] - Clean LR from bicubic downsampling
        - 'hf': torch.Tensor [3, H_hr, W_hr] - Original HR image
        - 'img_id': int - Image index for conditioning
        
    Degradation Process (CLEAN):
        - sr_scale > 1.0: Pure bicubic downsampling to (crop_size//sr_scale)²
        - sr_scale = 1.0: Bicubic down-up cycle (2x down, then back up) for mild degradation  
        - NO blur kernels, NO noise, NO compression artifacts
        - Perfect for evaluating model's intrinsic super-resolution ability
    """
    def __init__(self, data_path: str, crop_size: int, split: str = "train", sr_scale: float = 2.0):
        super().__init__()
        # Store dataset configuration parameters
        self.data_path = data_path    # Directory containing training images
        self.crop_size = crop_size    # Size of square HR crops (e.g., 256x256)
        self.split = split            # "train" or "val" for dataset splitting
        self.sr_scale = sr_scale      # Super-resolution scale factor (2.0 = 2x upsampling)
        
        # Calculate LR size based on scale factor
        # For sr_scale > 1: LR is smaller (e.g., 128x128 for 2x SR with 256x256 HR)
        # For sr_scale = 1: Same size (restoration mode)
        self.lr_size = int(crop_size // sr_scale) if sr_scale > 1.0 else crop_size
        
        # Get list of all valid image files from directory
        # Support common image formats: PNG, JPG, JPEG, BMP, TIF, TIFF
        self.fnames = sorted([f for f in os.listdir(data_path) 
                              if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))])
        
        # Split dataset: first 800 images for training, next 100 for validation
        if split == "train":
            # Take first 800 images (or all if fewer than 800)
            self.fnames = self.fnames[:min(800, len(self.fnames))]
        elif split == "val":
            # Take images 800-899 (or remaining if fewer images available)
            self.fnames = self.fnames[min(800, len(self.fnames)) : min(900, len(self.fnames))]
        
        # Define image preprocessing pipeline
        # Use center crops only for both train and val (no random crops for consistency)
        self.transform = T.Compose([
            T.CenterCrop(crop_size),  # Extract center square crop of specified size
            T.ToTensor()              # Convert PIL Image to torch tensor [0,1] range
        ])

    # Return the total number of images in this dataset split
    def __len__(self): 
        return len(self.fnames)

    def _create_lr_hr_pair(self, hr_image: torch.Tensor) -> tuple:
        """Create LR-HR pair using ONLY bicubic downsampling - no other degradations"""
        if self.sr_scale == 1.0:
            # Restoration mode: apply bicubic down-up cycle for consistent degradation
            # This creates a slightly degraded version at the same size
            scale_down = 0.5  # Downsample by 2x then upsample back
            
            # First downsample: HR -> smaller LR (introduces aliasing/blur)
            lr_small = F.interpolate(hr_image.unsqueeze(0),           # Add batch dim for interpolate 
                                   scale_factor=scale_down,           # 0.5 = half size
                                   mode='bicubic', align_corners=False, 
                                   antialias=True)                    # Prevent aliasing artifacts
            
            # Then upsample back to original size (introduces interpolation artifacts)
            lr_degraded = F.interpolate(lr_small,                     # The downsampled image
                                      size=hr_image.shape[-2:],       # Back to original H,W
                                      mode='bicubic', align_corners=False).squeeze(0)  # Remove batch dim
            
            # Clamp to valid pixel range and return LR-HR pair
            return lr_degraded.clamp(0, 1), hr_image
        else:
            # True super-resolution mode: LR is physically smaller - ONLY bicubic downsampling
            lr_clean = F.interpolate(hr_image.unsqueeze(0),           # Add batch dimension
                                   size=(self.lr_size, self.lr_size), # Target LR resolution 
                                   mode='bicubic', align_corners=False, # Standard bicubic interpolation
                                   antialias=True).squeeze(0)         # Remove batch dim, prevent aliasing
            
            # NO additional degradation - just pure bicubic downsampling
            # This gives us clean LR-HR pairs for evaluating pure SR capability
            return lr_clean.clamp(0, 1), hr_image

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a single image sample"""
        # Construct full path to the image file
        path = os.path.join(self.data_path, self.fnames[idx])
        
        # Load image and ensure RGB format (converts grayscale/RGBA to RGB)
        img = Image.open(path).convert("RGB")
        
        # Apply preprocessing: center crop + convert to tensor [0,1]
        hr = self.transform(img).float().clamp(0,1)         # [3,crop_size,crop_size] - HR ground truth
        
        # Generate corresponding LR image using clean bicubic downsampling
        lf, hf = self._create_lr_hr_pair(hr)                # LR: [3,lr_size,lr_size], HR: [3,crop_size,crop_size]
        
        # Return dictionary with LR input, HR target, and image ID for conditioning
        return {
            "lf": lf,                                        # Low-frequency (LR) input image
            "hf": hf,                                        # High-frequency (HR) target image  
            "img_id": torch.tensor(idx, dtype=torch.long)    # Image index for per-image conditioning
        }