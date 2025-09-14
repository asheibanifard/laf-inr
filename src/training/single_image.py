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
    sys.path.append('/home/armin/Documents/my_research_projects/Medical_AI/INR/laf-inr')
    from LAF_INR import LAFINR, LAFINRConfig, LAFINRLoss, psnr, create_psnr_plot, save_comparison


def _to_y(rgb: torch.Tensor) -> torch.Tensor:
    """RGB -> luminance Y (expects [B,3,H,W])."""
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


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

    lr_size = hr_size // scale_factor
    lr_img = F.interpolate(
        hr_img,
        size=(lr_size, lr_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

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
    prog = tqdm(range(epochs), desc=f"Training (Bicubic RGB={bicubic_rgb:.1f}dB, Y={bicubic_y:.1f}dB)")

    for epoch in prog:
        # Alpha warmup awareness
        model.residual.composer.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
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
        # Metrics

        epoch_list.append(epoch + 1)
        rgb_hist.append(best_psnr)
        y_hist.append(best_y_psnr)
        loss_hist.append(float(total_loss))

        # Progress bar text
        rgb_diff = best_psnr - bicubic_rgb
        y_diff = best_y_psnr - bicubic_y
        lr_now = scheduler.get_last_lr()[0]
        prog.set_description(
            f"Training (RGB={best_psnr:.1f}dB[{rgb_diff:+.1f}], Y={best_y_psnr:.1f}dB[{y_diff:+.1f}], lr={lr_now:.2e})"
        )

        # Save visuals/plot periodically
        if viz_every and (epoch % viz_every == 0 or epoch == epochs - 1):

            save_path = os.path.join(save_dir, f"best_image.png")
            # Calculate residuals
            # Upscale LR to HR size for comparison
            lr_upscaled = F.interpolate(lr_img, size=(hr_size, hr_size), mode="bicubic", align_corners=False, antialias=True)
            pred_lr_residual = torch.abs(best_pred[0] - lr_upscaled[0])
            hr_lr_residual = torch.abs(hr_img[0] - lr_upscaled[0])

            # Create comparison with residuals in 2x3 grid
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            def to_numpy(x):
                return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

            lr_np = to_numpy(lr_img[0])
            pred_np = to_numpy(best_pred[0].clamp(0, 1))
            hr_np = to_numpy(hr_img[0])
            pred_lr_residual_np = to_numpy(pred_lr_residual * 5)  # Brighten residual 5x
            lr_upscaled_np = to_numpy(lr_upscaled[0])
            hr_lr_residual_np = to_numpy(hr_lr_residual * 5)  # Brighten residual 5x

            axes[0, 0].imshow(lr_np); axes[0, 0].set_title('LR Input')
            axes[0, 1].imshow(pred_np); axes[0, 1].set_title('Predicted')
            axes[0, 2].imshow(hr_np); axes[0, 2].set_title('HR Target')
            axes[1, 0].imshow(pred_lr_residual_np); axes[1, 0].set_title('Residual (|Pred-LR|) ×5')
            axes[1, 1].imshow(lr_upscaled_np); axes[1, 1].set_title('LR Upscaled (Bicubic)')
            axes[1, 2].imshow(hr_lr_residual_np); axes[1, 2].set_title('Residual (|HR-LR|) ×5')

            # Show axis ticks
            imgs = [lr_np, pred_np, hr_np, pred_lr_residual_np, lr_upscaled_np, hr_lr_residual_np]
            for i, ax in enumerate(axes.flat):
                img_np = imgs[i]
                h, w = img_np.shape[:2]
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.set_xticks(range(0, w, max(1, w//5)))
                ax.set_yticks(range(0, h, max(1, h//5)))
                ax.tick_params(which='both', length=4, width=0.5, direction='out', labelsize=8)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            plot_path = os.path.join(save_dir, "best_procedure.png")
            create_psnr_plot(epoch_list, rgb_hist, y_hist, loss_hist, bicubic_rgb, bicubic_y, plot_path)

    # ---------------------------
    # Wrap-up & save
    # ---------------------------
    with torch.no_grad():
        final_rgb = psnr(best_pred, hr_img).item()
        final_y = psnr(_to_y(best_pred), _to_y(hr_img)).item()

    # Save final model
    ckpt_path = os.path.join(save_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg.__dict__,
            "final_loss": float(total_loss),
            "final_rgb_psnr": final_rgb,
            "final_y_psnr": final_y,
            "epochs": best_epoch,
        },
        ckpt_path,
    )

    print(f"✅ Done. Saved model to {ckpt_path}")
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
