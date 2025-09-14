"""
Main training functions for LAF-INR model.

This module contains the core training functionality including epoch running,
visualization sample collection, main training loop, and plotting utilities.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib.pyplot as plt

# Import required modules
try:
    from ..models.laf_inr import LAFINR, LAFINRConfig
    from ..losses import LAFINRLoss
    from ..data.dataset import CustomDataset
    from ..utils.metrics import psnr
    from ..utils.common import seed_everything
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('/home/armin/Documents/my_research_projects/Medical_AI/INR/laf-inr')
    from LAF_INR import LAFINR, LAFINRConfig, LAFINRLoss, CustomDataset, psnr, seed_everything

# Try to import optional visualization components
try:
    from epoch_visualizer import EpochVisualizer
except ImportError:
    EpochVisualizer = None


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, train=True, amp=True, clip=1.0):
    """
    Run one epoch of training or evaluation with automatic super-resolution detection.
    
    Args:
        model: LAF-INR model instance
        loader: DataLoader providing LR-HR pairs
        criterion: Loss function
        optimizer: Optimizer (ignored if train=False)
        scaler: GradScaler for mixed precision
        scheduler: Learning rate scheduler (ignored if train=False)
        device: Device to run on
        train: Whether in training mode
        amp: Whether to use automatic mixed precision
        clip: Gradient clipping threshold (ignored if train=False)
        
    Automatically detects if super-resolution is needed based on LR vs HR sizes.
    """
    model.train(train)
    mode = "train" if train else "eval"
    tot_loss=tot_psnr=tot_base=n=0
    pbar = tqdm(loader, desc=mode, leave=False)
    # Attach scheduler to pbar for access in training loop
    if train and scheduler:
        pbar.scheduler = scheduler
    for batch in pbar:
        lp = batch["lf"].to(device, non_blocking=True)  # Low-resolution input
        hr = batch["hf"].to(device, non_blocking=True)  # High-resolution target
        img_id = batch["img_id"].to(device, non_blocking=True)

        # Automatically detect if super-resolution is needed
        lr_h, lr_w = lp.shape[-2:]
        hr_h, hr_w = hr.shape[-2:]
        need_sr = (lr_h != hr_h) or (lr_w != hr_w)

        if train: optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.type=="cuda"):
            if need_sr:
                # Super-resolution: output at HR size
                out = model(lp, img_id, target_size=(hr_h, hr_w))
            else:
                # Restoration: same size input/output
                out = model(lp, img_id)
            losses = criterion(out, hr, lp); loss = losses["loss"]
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Check for inf/nan before stepping
            if torch.isfinite(loss):
                scaler.step(optimizer); scaler.update()
            else:
                print(f"Warning: inf/nan loss detected: {loss.item()}, skipping step")
                scaler.update()  # Still update scaler
            # Step scheduler once per optimizer step (per batch)
            if hasattr(pbar, 'scheduler'):
                pbar.scheduler.step()

        with torch.no_grad():
            # Clamp predictions only for PSNR calculation
            pred_clamped = out["pred"].clamp(0, 1)
            p = psnr(pred_clamped, hr)
            # For PSNR baseline, compare HR with LR upsampled to HR size
            if lp.shape[-2:] != hr.shape[-2:]:
                lp_for_baseline = F.interpolate(lp, size=hr.shape[-2:], mode='bicubic', align_corners=False)
            else:
                lp_for_baseline = lp
            b = psnr(lp_for_baseline, hr)
        tot_loss += float(loss); tot_psnr += float(p); tot_base += float(b); n += 1
        pbar.set_postfix(loss=f"{float(loss):.4f}", PSNR=f"{float(p):.2f}", Base=f"{float(b):.2f}", dB=f"{float(p-b):+.2f}",
                         alpha=f"{float(out['alpha']):.3f}")
    return {"loss": tot_loss/max(n,1), "psnr": tot_psnr/max(n,1), "base": tot_base/max(n,1),
            "improve": (tot_psnr-tot_base)/max(n,1)}


def get_visualization_samples(model, loader, device, num_samples=4, sr_factor=None):
    """
    Get sample outputs for visualization.
    
    Args:
        model: LAF-INR model
        loader: DataLoader for getting samples
        device: Device to run on
        num_samples: Number of samples to collect
        sr_factor: Super-resolution factor (None for same-size)
        
    Returns:
        Dict with 'lr_input', 'model_output', 'hr_target' tensors
    """
    model.eval()
    samples = {'lr_input': [], 'model_output': [], 'hr_target': []}
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break
                
            lp = batch["lf"].to(device, non_blocking=True)  # Low-resolution input  
            hr = batch["hf"].to(device, non_blocking=True)  # High-resolution target
            img_id = batch["img_id"].to(device, non_blocking=True)

            # Automatically detect if super-resolution is needed
            lr_h, lr_w = lp.shape[-2:]
            hr_h, hr_w = hr.shape[-2:]
            need_sr = (lr_h != hr_h) or (lr_w != hr_w)

            if need_sr:
                # Super-resolution: output at HR size
                out = model(lp, img_id, target_size=(hr_h, hr_w))
            else:
                # Restoration: same size input/output
                out = model(lp, img_id)
            
            pred = out["pred"]
            
            # Store samples (take first item from batch)
            samples['lr_input'].append(lp[0])
            samples['model_output'].append(pred[0])
            samples['hr_target'].append(hr[0])
    
    # Stack samples into tensors
    if samples['lr_input']:
        samples['lr_input'] = torch.stack(samples['lr_input'])
        samples['model_output'] = torch.stack(samples['model_output'])
        samples['hr_target'] = torch.stack(samples['hr_target'])
    
    return samples


def train(
    data_path: str,
    crop_size: int = 256,
    batch_size: int = 16,
    epochs: int = 200,
    lr: float = 2e-4,
    num_workers: int = 4,
    amp: bool = True,
    save_path: str = "lafinr_best.pth",
    sr_factor: Optional[float] = None,
    visualize: bool = True,
    viz_config: Optional[Dict] = None,
):
    """
    Training function for LAF-INR model on image super-resolution dataset.
    
    🎯 ANALOGY: Like training a team of art restoration experts:
    - Students (model) practice on training images with known "before/after" pairs
    - Teacher (loss) gives feedback: "Your restoration quality score is X/100" 
    - Gradual learning: Start eager (high learning rate), become careful (decay rate)
    - Validation: Test on unseen images to check if skills generalize
    - Save best student work (checkpoint) when validation scores are highest
    - Mixed precision: Work faster by using "sketch mode" when possible, "detail mode" when needed
    
    Args:
        data_path (str): Path to directory containing training images
        crop_size (int): Size for square image crops during training  
        batch_size (int): Training batch size
        epochs (int): Total number of training epochs
        lr (float): Peak learning rate for OneCycleLR scheduler
        num_workers (int): Number of data loading workers
        amp (bool): Whether to use automatic mixed precision training
        save_path (str): Path to save best model checkpoint
        sr_factor (float, optional): Super-resolution factor (e.g., 2.0 for 2x SR).
                                   If None, performs same-size reconstruction.
    
    Returns:
        LAFINR: Trained model instance
        
    Training Details:
        - Uses OneCycleLR scheduler with 10% warmup
        - Automatic mixed precision for memory efficiency  
        - Gradient clipping (max norm = 1.0)
        - Best model saved based on validation PSNR
        - Adaptive loss components with epoch-dependent weights
        
    Data Setup:
        - First 800 images: training with augmentations
        - Next 100 images: validation with center crops
        - Each image gets unique ID for conditioning
        
    Model Configuration:
        - Uses default LAFINRConfig parameters
        - Automatically determines num_images from training set size
        - Epoch-dependent alpha warmup in ResidualComposer
        - Adaptive residual loss decay after epoch 20
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Determine super-resolution scale factor
    dataset_sr_scale = sr_factor if sr_factor is not None else 1.0
    print(f"📊 Dataset Mode: {'Super-Resolution' if dataset_sr_scale > 1.0 else 'Restoration'}")
    if dataset_sr_scale > 1.0:
        print(f"   LR Size: {int(crop_size // dataset_sr_scale)}x{int(crop_size // dataset_sr_scale)}")
        print(f"   HR Size: {crop_size}x{crop_size}")
        print(f"   Scale Factor: {dataset_sr_scale}x")
    else:
        print(f"   Input/Output Size: {crop_size}x{crop_size} (same size restoration)")

    train_ds = CustomDataset(data_path, crop_size, split="train", sr_scale=dataset_sr_scale)
    val_ds   = CustomDataset(data_path, crop_size, split="val", sr_scale=dataset_sr_scale)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # Set cond_dim to match TokenConditioner output: 4 (if use_grad_stats) + id_dim
    # Only include id_dim if using multiple images (avoid no-op embedding)
    stats_dim = 4 if True else 2  # use_grad_stats=True
    id_dim = 8
    # For single-image training, id_embed would be no-op, so exclude it
    cond_dim = stats_dim  # Just use statistical features without id_embed for single image

    cfg = LAFINRConfig(d_model=160,
                       n_heads=4,
                       mlp_hidden=320,
                       fourier_K=14,
                       include_input=False,
                       num_rank_tokens=16,
                       rank_start=2,
                       cond_dim=cond_dim,  # Make sure this matches TokenConditioner output
                       id_dim=8,
                       use_grad_stats=True,
                       alpha_init=0.8,
                       alpha_min=0.5,
                       alpha_warmup_epochs=10,
                       warmup_epochs=10)
    model = LAFINR(cfg, num_images=len(train_ds)).to(device)
    criterion = LAFINRLoss(lambda_g=0.3, lambda_ent=0.1, lambda_hp=0.15, lambda_ortho=0.05, hp_sigma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize visualizer if requested
    visualizer = None
    if visualize and viz_config and EpochVisualizer is not None:
        visualizer = EpochVisualizer(viz_config, viz_config.get('save_dir', 'epoch_visualizations'))
    
    # Fixed scheduler - total steps should be epochs * batches_per_epoch  
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, div_factor=10.0, final_div_factor=10.0
    )
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device.type=="cuda")

    best_val = -1e9; step = 0
    for ep in range(epochs):
        # Set epoch for adaptive components
        model.residual.composer.set_epoch(ep)
        # Training/validation loops - no sr_factor needed since dataset handles LR/HR sizing
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, train=True, amp=amp, clip=1.0)
        va = run_epoch(model, val_loader,   criterion, optimizer, scaler, None, device, train=False, amp=amp, clip=None)

        print(f"Epoch {ep:3d}/{epochs} | "
              f"Train: Loss={tr['loss']:.4f} PSNR={tr['psnr']:.2f}dB Δ={tr['improve']:+.2f}dB | "
              f"Val:   Loss={va['loss']:.4f} PSNR={va['psnr']:.2f}dB Δ={va['improve']:+.2f}dB")

        # Update visualizer metrics
        if visualizer:
            visualizer.update_metrics(ep, tr['loss'], va['loss'], tr['psnr'], va['psnr'])
            # Get sample outputs for visualization
            if ep % viz_config.get('save_frequency', 5) == 0:
                samples = get_visualization_samples(model, val_loader, device, viz_config.get('num_samples', 4), sr_factor)
                visualizer.save_epoch_comparison(ep, samples, tr['psnr'], va['psnr'])

        # --- Save a rec plot for a sample val image every N epochs ---
        rec_plot_freq = 1
        if ep % rec_plot_freq == 0 or ep == epochs - 1:
            # Get a single sample from val_loader
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    lp = batch["lf"].to(device)
                    hr = batch["hf"].to(device)
                    img_id = batch["img_id"].to(device)
                    out = model(lp, img_id, target_size=hr.shape[-2:])
                    pred = out["pred"].clamp(0, 1)
                    # Plot and save
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(lp[0].permute(1, 2, 0).cpu().numpy()); axs[0].set_title('LR input')
                    axs[1].imshow(pred[0].permute(1, 2, 0).cpu().numpy()); axs[1].set_title('Model Rec')
                    axs[2].imshow(hr[0].permute(1, 2, 0).cpu().numpy()); axs[2].set_title('HR target')
                    plt.tight_layout()
                    os.makedirs('rec_plots', exist_ok=True)
                    plt.savefig(f'rec_plots/rec_epoch_{ep:03d}.png', dpi=120)
                    plt.close(fig)
                    break  # Only one sample

        if va["psnr"] > best_val:
            best_val = va["psnr"]
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": va["psnr"],
                "val_improve": va["improve"],
                "cfg": cfg.__dict__,
            }, save_path)
            print(f"  ↳ Saved best to {save_path} (Val PSNR {best_val:.2f} dB)")

    # Save final visualization summary
    if visualizer:
        visualizer.save_final_summary(epochs - 1)

    print("Done.")
    return model


def create_psnr_plot(epoch_list, rgb_psnr_history, y_psnr_history, loss_history, 
                     bicubic_rgb, bicubic_y, save_path):
    """Create and save training PSNR plot with bicubic baselines."""
    plt.style.use('default')  # Use clean default style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: PSNR curves
    ax1.plot(epoch_list, rgb_psnr_history, 'b-', linewidth=2, label='LAF-INR RGB', marker='o', markersize=3)
    ax1.plot(epoch_list, y_psnr_history, 'r-', linewidth=2, label='LAF-INR Y', marker='s', markersize=3)
    
    # Bicubic baselines (horizontal lines)
    ax1.axhline(y=bicubic_rgb, color='b', linestyle='--', alpha=0.7, linewidth=2, label='Bicubic RGB')
    ax1.axhline(y=bicubic_y, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Bicubic Y')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('PSNR (dB)', fontsize=12)
    ax1.set_title('Training PSNR vs Bicubic Baseline', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=min(min(rgb_psnr_history), min(y_psnr_history)) - 1)
    
    # Plot 2: Loss curve
    ax2.plot(epoch_list, loss_history, 'g-', linewidth=2, label='Training Loss', marker='^', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()