#!/usr/bin/env python3
"""
Single image overfitting for LAF-INR.

- Safe defaults for 2×–4× SR on 256–512px images
- Optional AMP (mixed precision)
- Robust guards (divisibility, scheduler step size, checkpoint resume)
- Config kept small for easy overfit
- Saves comparisons and (optionally) PSNR curves at a chosen frequency
"""

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Import required modules
try:
    from ..models import LAFINR, LAFINRConfig
    from ..losses import LAFINRLoss
    from ..utils.metrics import psnr
    from .trainer import create_psnr_plot
    from .utils import save_comparison
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models import LAFINR, LAFINRConfig
    from src.losses import LAFINRLoss
    from src.utils.metrics import psnr
    from src.training.trainer import create_psnr_plot
    from src.training.utils import save_comparison


def _to_y(rgb: torch.Tensor) -> torch.Tensor:
    """RGB -> luminance Y (expects [B,3,H,W])."""
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Simple SSIM implementation"""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu1 = F.avg_pool2d(pred, 3, 1, 1)
    mu2 = F.avg_pool2d(target, 3, 1, 1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, 3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _simple_lpips(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple LPIPS approximation using feature differences"""
    # Simple approximation: compute L2 distance in feature space after some convolutions
    # This is a very simplified version - real LPIPS uses pre-trained VGG features

    # Apply some basic feature extraction (edge detection)
    kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=pred.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(pred.shape[1], 1, 1, 1)

    pred_features = F.conv2d(pred, kernel, padding=1, groups=pred.shape[1])
    target_features = F.conv2d(target, kernel, padding=1, groups=target.shape[1])

    # Compute normalized L2 distance
    diff = (pred_features - target_features).pow(2).mean()
    return diff


def _rgb_to_frequency_components(rgb_tensor: torch.Tensor) -> tuple:
    """Split RGB into low and high frequency components using Gaussian blur"""
    # Apply Gaussian blur for low frequencies
    kernel_size = 15
    sigma = 3.0
    channels = rgb_tensor.shape[1]

    # Create Gaussian kernel
    kernel = torch.zeros(channels, 1, kernel_size, kernel_size, device=rgb_tensor.device)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[:, 0, i, j] = torch.exp(torch.tensor(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)))
    kernel = kernel / kernel.sum()

    # Apply Gaussian blur with padding
    padding = kernel_size // 2
    low_freq = F.conv2d(rgb_tensor, kernel, padding=padding, groups=channels)
    high_freq = rgb_tensor - low_freq

    return low_freq, high_freq


def _create_comprehensive_plot(lr_img, pred_img, hr_img, epoch_list, rgb_hist, y_hist, loss_hist, bicubic_rgb, bicubic_y, save_path, best_psnr_rgb, best_psnr_y, best_loss):
    """Create comprehensive analysis plot with all requested components"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.25, wspace=0.3)

    # Helper function to convert tensor to numpy
    def to_numpy(x):
        return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    # Calculate bicubic upsampling
    hr_size = hr_img.shape[-1]
    bicubic = F.interpolate(lr_img, size=(hr_size, hr_size), mode="bicubic", align_corners=False, antialias=True)

    # Get frequency components
    lf_hr, hf_hr = _rgb_to_frequency_components(hr_img)
    lf_pred, hf_pred = _rgb_to_frequency_components(pred_img)

    # Calculate residuals
    pred_lf_residual = torch.abs(pred_img - lr_img)
    hr_lr_residual = torch.abs(hr_img - F.interpolate(lr_img, size=(hr_size, hr_size), mode="bicubic", align_corners=False, antialias=True))

    # Row 1: Original Images with crop markers
    ax1 = fig.add_subplot(gs[0, 0])
    bicubic_np = to_numpy(bicubic[0])
    ax1.imshow(bicubic_np)
    ax1.set_title('Low Frequency (Bicubic)', fontsize=12)
    # Mark the cropped area [0:100, 0:100] with red rectangle
    ax1.add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='red', linewidth=2))

    ax2 = fig.add_subplot(gs[0, 1])
    # Properly display high frequency components: normalize to [0,1] range
    hf_display = (hf_hr[0] - hf_hr[0].min()) / (hf_hr[0].max() - hf_hr[0].min() + 1e-8)
    hf_np = to_numpy(hf_display)
    ax2.imshow(hf_np)
    ax2.set_title('Original High Frequency image', fontsize=12)
    # Mark the cropped area [0:100, 0:100] with red rectangle
    ax2.add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='red', linewidth=2))

    ax3 = fig.add_subplot(gs[0, 2])
    pred_np = to_numpy(pred_img[0])
    ax3.imshow(pred_np)
    ax3.set_title('Predicted High Frequency image', fontsize=12)
    # Mark the cropped area [0:100, 0:100] with red rectangle
    ax3.add_patch(plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='red', linewidth=2))

    # Row 2: Cropped Details [0:100, 0:100]
    ax1_crop = fig.add_subplot(gs[1, 0])
    crop_bicubic = bicubic_np[:100, :100]
    ax1_crop.imshow(crop_bicubic)
    ax1_crop.set_title('Detail: crop of LF', fontsize=11)

    ax2_crop = fig.add_subplot(gs[1, 1])
    crop_hf = hf_np[:100, :100]
    ax2_crop.imshow(crop_hf)
    ax2_crop.set_title('Detail: crop of HF', fontsize=11)

    ax3_crop = fig.add_subplot(gs[1, 2])
    crop_pred = pred_np[:100, :100]
    ax3_crop.imshow(crop_pred)
    ax3_crop.set_title('Detail: crop of Predicted', fontsize=11)

    # Row 3: Residuals
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(to_numpy(pred_lf_residual[0]))
    ax5.set_title('Residual |Predicted - LR|', fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(to_numpy(hr_lr_residual[0]))
    ax6.set_title('Residual |HR - LR|', fontsize=12)
    ax6.axis('off')

    # Calculate SSIM and LPIPS values for current best prediction
    with torch.no_grad():
        ssim_val = _ssim(pred_img, hr_img).item()
        ssim_bicubic = _ssim(bicubic, hr_img).item()
        lpips_val = _simple_lpips(pred_img, hr_img).item()
        lpips_bicubic = _simple_lpips(bicubic, hr_img).item()

    # Row 3, Col 3: Quality Metrics Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    metrics_text = f"""Quality Metrics Summary (Best):

PSNR RGB: {best_psnr_rgb:.2f} dB
PSNR Y: {best_psnr_y:.2f} dB
SSIM: {ssim_val:.4f}
LPIPS: {lpips_val:.4f}
Loss: {best_loss:.6f}

Bicubic Baseline:
RGB: {bicubic_rgb:.2f} dB
Y: {bicubic_y:.2f} dB
SSIM: {ssim_bicubic:.4f}
LPIPS: {lpips_bicubic:.4f}

Improvements:
RGB: +{best_psnr_rgb - bicubic_rgb:.2f} dB
Y: +{best_psnr_y - bicubic_y:.2f} dB
SSIM: +{ssim_val - ssim_bicubic:.4f}
LPIPS: {lpips_bicubic - lpips_val:+.4f}
"""

    ax7.text(0.0, 1.0, metrics_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.5))

    # Row 4: Training Progress (dual y-axis)
    ax9 = fig.add_subplot(gs[3, 0])  # First column
    ax9_twin = ax9.twinx()

    line1 = ax9.plot(epoch_list, loss_hist, 'b-', linewidth=2, label='Training Loss')
    line2 = ax9_twin.plot(epoch_list, rgb_hist, 'r-', linewidth=2, label='RGB PSNR')
    line3 = ax9_twin.plot(epoch_list, y_hist, 'g-', linewidth=2, label='Y PSNR')

    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Loss', color='blue')
    ax9_twin.set_ylabel('PSNR (dB)', color='red')
    ax9.set_title('Training Progress', fontsize=11)
    ax9.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax9.legend(lines, labels, loc='center right', fontsize=9)

    # Frequency Analysis in row 4
    ax10 = fig.add_subplot(gs[3, 1])

    # Calculate frequency domain metrics
    with torch.no_grad():
        lf_psnr = psnr(lf_pred, lf_hr).item()
        hf_psnr = psnr(hf_pred + 0.5, hf_hr + 0.5).item()

        lf_bicubic, hf_bicubic = _rgb_to_frequency_components(bicubic)
        lf_bicubic_psnr = psnr(lf_bicubic, lf_hr).item()
        hf_bicubic_psnr = psnr(hf_bicubic + 0.5, hf_hr + 0.5).item()

    # Frequency domain comparison
    freq_methods = ['Bic\nLF', 'Pred\nLF', 'Bic\nHF', 'Pred\nHF']
    freq_psnr = [lf_bicubic_psnr, lf_psnr, hf_bicubic_psnr, hf_psnr]
    freq_colors = ['orange', 'blue', 'red', 'green']

    bars = ax10.bar(freq_methods, freq_psnr, color=freq_colors, alpha=0.7)
    ax10.set_ylabel('PSNR (dB)')
    ax10.set_title('Frequency Analysis', fontsize=11)
    ax10.grid(True, alpha=0.5, axis='y')
    ax10.tick_params(axis='x', labelsize=8)

    # Add value labels on bars
    for bar, val in zip(bars, freq_psnr):
        ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    # SSIM vs LPIPS comparison in row 3, col 3
    ax11 = fig.add_subplot(gs[3, 2])
    # Create SSIM vs LPIPS comparison
    import numpy as np
    x = np.arange(2)
    width = 0.35

    ssim_values = [ssim_bicubic, ssim_val]
    lpips_values = [lpips_bicubic, lpips_val]

    bars1 = ax11.bar(x - width/2, ssim_values, width, label='SSIM', color='blue', alpha=0.7)
    ax11_twin = ax11.twinx()
    bars2 = ax11_twin.bar(x + width/2, lpips_values, width, label='LPIPS', color='red', alpha=0.7)

    ax11.set_ylabel('SSIM (↑ better)', color='blue', fontsize=9)
    ax11_twin.set_ylabel('LPIPS (↓ better)', color='red', fontsize=9)
    ax11.set_title('SSIM vs LPIPS', fontsize=10, pad=15)
    ax11.set_xticks(x)
    ax11.set_xticklabels(['Bicubic', 'Predicted'], fontsize=8)
    ax11.set_ylim(0, 1)
    ax11.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, ssim_values):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='blue')
    for bar, val in zip(bars2, lpips_values):
        ax11_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lpips_values)*0.05,
                      f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='red')

    # Remove row 4 - no longer needed
    # All plots now fit in first 3 rows

    plt.suptitle('Comprehensive LAF-INR Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.05, right=0.98, hspace=0.30, wspace=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def overfit_single_image(
    image_path: str,
    hr_size: int,
    scale_factor: int,
    epochs: int,
    lr: float,
    batch_points: Optional[int] = None,   # (currently unused; kept for API compatibility)
    save_dir: str = "results",
    use_uncertainty: bool = False,        # (unused in current losses; placeholder)
    resume_from: Optional[str] = None,
    use_mixed_precision: bool = False,
    alpha_warmup_epochs: int = 100,
    use_adaptive_sampling: bool = False,  # (unused; placeholder)
    viz_every: int = 25,                  # save visual/plots every N epochs (set 1 to save every epoch)
) -> LAFINR:
    """
    Overfit LAF-INR on a single image for capacity/debug checks.

    Args:
        image_path: Path to an RGB image.
        hr_size: Target HR size (square). Must be divisible by scale_factor.
        scale_factor: Downsample factor to create LR (2 or 4 recommended).
        epochs: Training epochs (e.g., 600 for 2× on 256px, 1200 for 4×).
        lr: Learning rate (1e-3 for 2×, 5e-4–1e-3 for 4×).
        viz_every: Save comparison and PSNR plot every N epochs.

    Returns:
        Trained LAFINR model.
    """
    # ---------------------------
    # I/O & basic validations
    # ---------------------------
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be a positive integer.")
    if hr_size % scale_factor != 0:
        raise ValueError("hr_size must be divisible by scale_factor to avoid fractional resize.")

    os.makedirs(save_dir, exist_ok=True)

    print(f"🎯 Single Image Overfitting: {image_path}")
    print(f"   HR Size: {hr_size}, Scale: {scale_factor}×, Epochs: {epochs}, LR: {lr:g}")
    print(f"   AMP: {use_mixed_precision}, Save dir: {save_dir}, viz_every: {viz_every}")

    # ---------------------------
    # Load & prepare data
    # ---------------------------
    IMG = []

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((hr_size, hr_size)), T.ToTensor()])
    hr_img = transform(img).unsqueeze(0)  # [1,3,H,W]

    # Create LR by downsampling then upsampling back to HR size (degraded HR)
    lr_size = hr_size // scale_factor
    lr_downsampled = F.interpolate(
        hr_img,
        size=(lr_size, lr_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    # Upsample back to HR size - this is the degraded LR input at HR resolution
    lr_img = F.interpolate(
        lr_downsampled,
        size=(hr_size, hr_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    print(f"   Degradation: {hr_size}×{hr_size} → {lr_size}×{lr_size} → {hr_size}×{hr_size} (scale {scale_factor}×)")
    print(f"   LR: {tuple(lr_img.shape[-2:])} → HR: {tuple(hr_img.shape[-2:])}")

    # ---------------------------
    # Model config (small & stable)
    # ---------------------------
    cfg = LAFINRConfig()
    cfg.alpha_warmup_epochs = alpha_warmup_epochs

    # Small architecture to overfit easily
    cfg.d_model = 64
    cfg.n_heads = 8
    cfg.mlp_hidden = cfg.d_model * 2
    cfg.num_rank_tokens = 16
    cfg.fourier_K = 12

    # Conservative residual blending
    cfg.alpha_init = 0.04
    cfg.alpha_min = 0.02

    # ID embedding effectively disabled by num_images=1
    num_images = 1

    model = LAFINR(cfg, num_images=num_images)

    # ---------------------------
    # Device, optimizer, scheduler
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    hr_img = hr_img.to(device)
    lr_img = lr_img.to(device)
    img_id = torch.tensor([0], dtype=torch.long, device=device)

    # Adam is generally smoother for single-image overfit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.0)

    # Safe StepLR: ensure step_size >= 1
    step_size = max(1, epochs // 4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    # Loss: pure reconstruction for overfit
    loss_fn = LAFINRLoss(lambda_g=0.0, lambda_ent=0.0, lambda_hp=0.0, lambda_ortho=0.0)

    # AMP scaler (optional)
    amp_enabled = use_mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda:0', enabled=amp_enabled)

    # ---------------------------
    # Baselines & (optional) resume
    # ---------------------------
    with torch.no_grad():
        bicubic = F.interpolate(lr_img, size=(hr_size, hr_size), mode="bicubic", align_corners=False, antialias=True)
        bicubic_rgb = psnr(bicubic, hr_img).item()
        bicubic_y = psnr(_to_y(bicubic), _to_y(hr_img)).item()
    print(f"📏 Bicubic baseline: RGB={bicubic_rgb:.2f} dB, Y={bicubic_y:.2f} dB")

    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"↺ Resumed weights from: {resume_from}")

    # Quick untrained sanity pass
    with torch.no_grad():
        model.eval()
        test_out = model(lr_img, img_id, target_size=(hr_size, hr_size))["pred"]
        test_psnr = psnr(test_out, hr_img).item()
        print(f"🧪 Initial PSNR (untrained): {test_psnr:.2f} dB")
    model.train()

    # ---------------------------
    # Training loop
    # ---------------------------
    epoch_list, rgb_hist, y_hist, loss_hist = [], [], [], []
    best_loss = float('inf')
    best_psnr = 0.0
    best_y_psnr = 0.0
    best_pred = None
    best_epoch = 0

    # Track running best values for plotting
    running_best_psnr = 0.0
    running_best_y_psnr = 0.0
    running_best_loss = float('inf')
    prog = tqdm(range(epochs), desc=f"Training (Bicubic RGB={bicubic_rgb:.1f}dB, Y={bicubic_y:.1f}dB)")

    for epoch in prog:
        # Alpha warmup awareness
        model.residual.composer.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp_enabled):
            outputs = model(lr_img, img_id, target_size=(hr_size, hr_size))
            pred = outputs["pred"]
            total_loss = loss_fn(outputs, hr_img, lr_img)["loss"]

        # Backward
        if amp_enabled:
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            with torch.no_grad():
                best_psnr = psnr(pred, hr_img).item()
                best_y_psnr = psnr(_to_y(pred), _to_y(hr_img)).item()
                best_pred = pred.detach().clone()
                best_epoch = epoch + 1

                # Save best model immediately
                best_ckpt_path = os.path.join(save_dir, "best_model.pth")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "best_loss": best_loss,
                        "best_rgb_psnr": best_psnr,
                        "best_y_psnr": best_y_psnr,
                        "best_epoch": best_epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    best_ckpt_path,
                )
        # Metrics - track current values for this epoch
        current_psnr_rgb = psnr(pred, hr_img).item()
        current_psnr_y = psnr(_to_y(pred), _to_y(hr_img)).item()
        current_loss = float(total_loss)

        # Update running best values for plotting
        running_best_psnr = max(running_best_psnr, current_psnr_rgb)
        running_best_y_psnr = max(running_best_y_psnr, current_psnr_y)
        running_best_loss = min(running_best_loss, current_loss)

        epoch_list.append(epoch + 1)
        rgb_hist.append(running_best_psnr)
        y_hist.append(running_best_y_psnr)
        loss_hist.append(running_best_loss)

        # Progress bar text
        rgb_diff = best_psnr - bicubic_rgb
        y_diff = best_y_psnr - bicubic_y
        lr_now = scheduler.get_last_lr()[0]
        prog.set_description(
            f"Training (RGB={best_psnr:.1f}dB[{rgb_diff:+.1f}], Y={best_y_psnr:.1f}dB[{y_diff:+.1f}], lr={lr_now:.2e})"
        )

        # Save visuals/plot periodically
        if viz_every and (epoch % viz_every == 0 or epoch == epochs - 1):
            save_path = os.path.join(save_dir, f"comprehensive_analysis_scale_{scale_factor}.png")
            _create_comprehensive_plot(
                lr_img, best_pred, hr_img,
                epoch_list, rgb_hist, y_hist, loss_hist,
                bicubic_rgb, bicubic_y, save_path,
                best_psnr, best_y_psnr, best_loss
            )

    # ---------------------------
    # Wrap-up & save
    # ---------------------------
    with torch.no_grad():
        final_rgb = psnr(best_pred, hr_img).item()
        final_y = psnr(_to_y(best_pred), _to_y(hr_img)).item()

    # Save final model (last epoch)
    final_ckpt_path = os.path.join(save_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "final_loss": float(total_loss),
            "final_rgb_psnr": final_rgb,
            "final_y_psnr": final_y,
            "total_epochs": epochs,
            "best_epoch": best_epoch,
        },
        final_ckpt_path,
    )

    best_ckpt_path = os.path.join(save_dir, "best_model.pth")
    print(f"✅ Done. Saved final model to {final_ckpt_path}")
    print(f"📈 Best model saved to {best_ckpt_path} (epoch {best_epoch})")
    print(f"📊 Final: LAF-INR RGB={final_rgb:.2f} dB (+{final_rgb - bicubic_rgb:+.2f}), "
          f"Y={final_y:.2f} dB (+{final_y - bicubic_y:+.2f})")

    # One last plot (in case viz_every=0)
    if viz_every == 0:
        plot_path = os.path.join(save_dir, "training_psnr.png")
        create_psnr_plot(epoch_list, rgb_hist, y_hist, loss_hist, bicubic_rgb, bicubic_y, plot_path)

    return model



if __name__ == "__main__":
    overfit_single_image(
        image_path='dataset/DIV2K/0001.png',
        hr_size=512,
        scale_factor=2,
        epochs=3000,
        lr=1e-3,
        save_dir="results_single",
        use_mixed_precision=True,
        viz_every=200,
        alpha_warmup_epochs=100,
    )
