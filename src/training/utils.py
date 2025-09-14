"""
Training utility functions for LAF-INR model.

This module contains utility functions for visualization, comparison, and demonstration
of LAF-INR capabilities including arbitrary resolution generation and partial input processing.
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Import required modules
try:
    from ..models.laf_inr import LAFINR, LAFINRConfig
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('/home/armin/Documents/my_research_projects/Medical_AI/INR/laf-inr')
    from LAF_INR import LAFINR, LAFINRConfig


def save_comparison(lr_imgs: torch.Tensor, preds: torch.Tensor, hr_imgs: torch.Tensor, save_path: str):
    """
    Save validation samples visualization with up to 4 samples.

    Args:
        lr_imgs (torch.Tensor): Low-resolution input images [B,C,H,W] or [C,H,W]
        preds (torch.Tensor): Model predictions [B,C,H,W] or [C,H,W]
        hr_imgs (torch.Tensor): High-resolution targets [B,C,H,W] or [C,H,W]
        save_path (str): Path to save the comparison image
    """
    import matplotlib.pyplot as plt

    # Handle single image case - add batch dimension
    if lr_imgs.dim() == 3:
        lr_imgs = lr_imgs.unsqueeze(0)
    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    if hr_imgs.dim() == 3:
        hr_imgs = hr_imgs.unsqueeze(0)

    # Convert to numpy and denormalize
    def to_numpy(x):
        return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    # Take first 4 samples
    num_samples = min(4, lr_imgs.size(0))

    # Create 4x3 grid: 4 samples, 3 columns (LR, Pred, HR)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    # Handle case where we have only one sample
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    column_titles = ['LR Input', 'Predicted', 'HR Target']

    for i in range(num_samples):
        lr_np = to_numpy(lr_imgs[i])
        pred_np = to_numpy(preds[i])
        hr_np = to_numpy(hr_imgs[i])

        # Plot LR, Pred, HR for this sample
        axes[i, 0].imshow(lr_np)
        axes[i, 1].imshow(pred_np)
        axes[i, 2].imshow(hr_np)

        # Set titles only for first row
        if i == 0:
            for j, title in enumerate(column_titles):
                axes[i, j].set_title(title)

        # Add sample number on left
        axes[i, 0].set_ylabel(f'Sample {i+1}', rotation=90, labelpad=20)

        # Show axis ticks
        for j in range(3):
            img_np = [lr_np, pred_np, hr_np][j]
            h, w = img_np.shape[:2]
            axes[i, j].set_xlim(0, w)
            axes[i, j].set_ylim(h, 0)
            axes[i, j].set_xticks(range(0, w, max(1, w//5)))
            axes[i, j].set_yticks(range(0, h, max(1, h//5)))
            axes[i, j].tick_params(which='both', length=4, width=0.5, direction='out', labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_arbitrary_resolution(image_path: str, model_path: str, save_dir: str):
    """
    Demonstrate arbitrary resolution generation capability.
    
    This function loads a trained LAF-INR model and demonstrates its ability to generate
    the same image at multiple arbitrary resolutions, showcasing the continuous nature
    of implicit neural representations.
    
    Args:
        image_path (str): Path to test image
        model_path (str): Path to trained model checkpoint
        save_dir (str): Directory to save demonstration results
    """
    print(f"🎨 Arbitrary Resolution Demo: {image_path}")
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg_dict = checkpoint['cfg']
    cfg = LAFINRConfig(**cfg_dict)
    model = LAFINR(cfg, num_images=1000)  # Support many images
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # Load test image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    lr_img = transform(img).unsqueeze(0).to(device)
    img_id = torch.tensor([0], dtype=torch.long, device=device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate at multiple resolutions
    resolutions = [(384, 384), (512, 512), (768, 768), (1024, 1024)]
    
    with torch.no_grad():
        for i, (h, w) in enumerate(resolutions):
            print(f"  Generating {h}×{w}...")
            outputs = model(lr_img, img_id, target_size=(h, w))
            pred = outputs["pred"][0]  # [3,H,W]
            
            # Save result
            save_path = os.path.join(save_dir, f"arbitrary_res_{h}x{w}.png")
            T.ToPILImage()(pred.cpu().clamp(0, 1)).save(save_path)
            
    print(f"✅ Arbitrary resolution demo complete! Results in {save_dir}")


def demonstrate_partial_input_sr(image_path: str, model_path: str, save_dir: str):
    """
    Demonstrate super-resolution from partial input.
    
    This function demonstrates the model's ability to perform super-resolution
    even when given incomplete or masked input, showcasing the inpainting
    capabilities of implicit neural representations.
    
    Args:
        image_path (str): Path to test image
        model_path (str): Path to trained model checkpoint
        save_dir (str): Directory to save demonstration results
    """
    print(f"🧩 Partial Input Demo: {image_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg_dict = checkpoint['cfg']
    cfg = LAFINRConfig(**cfg_dict)
    model = LAFINR(cfg, num_images=1000)  # Support many images
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # Load and prepare test image
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    full_img = transform(img).unsqueeze(0).to(device)
    img_id = torch.tensor([0], dtype=torch.long, device=device)
    
    # Create different partial inputs (masks)
    masks = {
        'half_vertical': torch.ones_like(full_img),
        'checkerboard': torch.ones_like(full_img),
        'center_crop': torch.ones_like(full_img),
        'random_holes': torch.ones_like(full_img)
    }
    
    # Half vertical mask
    masks['half_vertical'][:, :, :, full_img.shape[-1]//2:] = 0
    
    # Checkerboard mask
    h, w = full_img.shape[-2:]
    for i in range(0, h, 16):
        for j in range(0, w, 16):
            if (i//16 + j//16) % 2 == 1:
                masks['checkerboard'][:, :, i:i+16, j:j+16] = 0
    
    # Center crop mask (keep only center 128x128)
    center_h, center_w = h//2, w//2
    masks['center_crop'][:, :, :, :] = 0
    masks['center_crop'][:, :, center_h-64:center_h+64, center_w-64:center_w+64] = 1
    
    # Random holes mask
    import torch.nn.functional as F
    random_mask = torch.rand_like(full_img[:, 0:1, :, :]) > 0.3  # 70% visible
    masks['random_holes'] = random_mask.float().expand_as(full_img)
    
    with torch.no_grad():
        for mask_name, mask in masks.items():
            print(f"  Processing {mask_name} mask...")
            partial_input = full_img * mask
            
            # Generate super-resolution from partial input
            outputs = model(partial_input, img_id, target_size=(512, 512))
            pred = outputs["pred"][0]  # [3,H,W]
            
            # Save comparison: partial input | prediction | full reference
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Resize partial input for comparison
            partial_display = F.interpolate(partial_input, size=(512, 512), mode='bicubic', align_corners=False)[0]
            full_display = F.interpolate(full_img, size=(512, 512), mode='bicubic', align_corners=False)[0]
            
            def to_numpy(x):
                return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            
            axes[0].imshow(to_numpy(partial_display))
            axes[0].set_title(f'Partial Input ({mask_name})')
            axes[0].axis('off')
            
            axes[1].imshow(to_numpy(pred))
            axes[1].set_title('Model Prediction')
            axes[1].axis('off')
            
            axes[2].imshow(to_numpy(full_display))
            axes[2].set_title('Full Reference')
            axes[2].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"partial_input_{mask_name}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
    print(f"✅ Partial input demo complete! Results in {save_dir}")


def demonstrate_continuous_scale(image_path: str, model_path: str, save_dir: str):
    """
    Demonstrate continuous scale factor interpolation.
    
    This function demonstrates the model's ability to generate outputs at
    continuously varying scale factors, showcasing the smooth scaling
    capabilities of implicit neural representations.
    
    Args:
        image_path (str): Path to test image
        model_path (str): Path to trained model checkpoint
        save_dir (str): Directory to save demonstration results
    """
    print(f"📐 Continuous Scale Demo: {image_path}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg_dict = checkpoint['cfg']
    cfg = LAFINRConfig(**cfg_dict)
    model = LAFINR(cfg, num_images=1000)  # Support many images
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    # Load test image at low resolution
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
    lr_img = transform(img).unsqueeze(0).to(device)
    img_id = torch.tensor([0], dtype=torch.long, device=device)
    
    # Generate at continuous scale factors
    scale_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    base_size = 128
    
    results = []
    
    with torch.no_grad():
        for scale in scale_factors:
            target_size = int(base_size * scale)
            print(f"  Generating at scale {scale}x ({target_size}x{target_size})...")
            
            outputs = model(lr_img, img_id, target_size=(target_size, target_size))
            pred = outputs["pred"][0]  # [3,H,W]
            results.append((scale, pred))
    
    # Create a grid showing all scales
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    def to_numpy(x):
        return x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    
    for i, (scale, pred) in enumerate(results):
        axes[i].imshow(to_numpy(pred))
        axes[i].set_title(f'Scale {scale}x ({pred.shape[-1]}×{pred.shape[-2]})')
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "continuous_scale_demo.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual results
    for scale, pred in results:
        individual_path = os.path.join(save_dir, f"scale_{scale}x.png")
        T.ToPILImage()(pred.cpu().clamp(0, 1)).save(individual_path)
    
    print(f"✅ Continuous scale demo complete! Results in {save_dir}")


# Convenience function to run all demonstrations
def run_all_demonstrations(image_path: str, model_path: str, save_dir: str):
    """
    Run all available demonstrations.
    
    Args:
        image_path (str): Path to test image
        model_path (str): Path to trained model checkpoint
        save_dir (str): Base directory to save all demonstration results
    """
    print("🚀 Running all LAF-INR demonstrations...")
    
    # Create subdirectories for each demo
    demos = [
        ("arbitrary_resolution", demonstrate_arbitrary_resolution),
        ("partial_input", demonstrate_partial_input_sr),
        ("continuous_scale", demonstrate_continuous_scale)
    ]
    
    for demo_name, demo_func in demos:
        demo_dir = os.path.join(save_dir, demo_name)
        try:
            demo_func(image_path, model_path, demo_dir)
        except Exception as e:
            print(f"❌ Error in {demo_name} demo: {e}")
            continue
    
    print(f"🎯 All demonstrations complete! Results saved in {save_dir}")