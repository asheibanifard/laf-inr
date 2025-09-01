import torch
import torch.cuda
import tqdm
import yaml
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from dataset import LAFINRDataset
from torch.utils.data import DataLoader
from model import LAFINRModel
from loss import LAFINRCriterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Clear GPU cache at start
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def psnr(pred, target):
    """
    Calculate PSNR between two tensors
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        torch.Tensor: PSNR value in dB (always returns tensor for consistent .item() usage)
    """
    mse = torch.mean((pred - target) ** 2)
    # Fix: Use tensor comparison and return consistent tensor type
    if mse < 1e-10:  # Use small threshold instead of exact zero comparison
        return torch.tensor(100.0, device=mse.device, dtype=mse.dtype)  # High PSNR for near-perfect match
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_reconstruction_results(lr_image, hr_image, pred_image, epoch, batch_idx, save_dir="src/reconstruction_results"):
    """
    Save reconstruction results as PNG images
    
    Args:
        lr_image: Low resolution input tensor [C, H, W]
        hr_image: High resolution ground truth tensor [C, H, W]  
        pred_image: Predicted high resolution tensor [C, H, W]
        epoch: Current epoch number
        batch_idx: Current batch index
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and denormalize
    def tensor_to_image(tensor):
        # Move to CPU and convert to numpy
        img = tensor.detach().cpu().numpy()
        # Clamp values to [0, 1] range
        img = np.clip(img, 0, 1)
        # Convert from CHW to HWC format
        if img.shape[0] == 3:  # RGB image
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:  # Grayscale image
            img = img.squeeze(0)
        # Convert to 0-255 range
        img = (img * 255).astype(np.uint8)
        return img
    
    # Convert tensors to images
    lr_img = tensor_to_image(lr_image)
    hr_img = tensor_to_image(hr_image)
    pred_img = tensor_to_image(pred_image)
    
    # Create combined image showing LR, HR, and Predicted side by side
    if len(lr_img.shape) == 3:  # RGB
        h, w, c = hr_img.shape
        combined = np.zeros((h, w * 3, c), dtype=np.uint8)
        
        # Resize LR to same height as HR for comparison
        lr_pil = Image.fromarray(lr_img)
        lr_resized = lr_pil.resize((w, h), Image.LANCZOS)
        lr_resized = np.array(lr_resized)
        
        combined[:, :w] = lr_resized      # LR (upscaled for visualization)
        combined[:, w:2*w] = hr_img       # HR (ground truth)
        combined[:, 2*w:3*w] = pred_img   # Predicted
        
    else:  # Grayscale
        h, w = hr_img.shape
        combined = np.zeros((h, w * 3), dtype=np.uint8)
        
        # Resize LR to same height as HR for comparison
        lr_pil = Image.fromarray(lr_img)
        lr_resized = lr_pil.resize((w, h), Image.LANCZOS)
        lr_resized = np.array(lr_resized)
        
        combined[:, :w] = lr_resized      # LR (upscaled for visualization)
        combined[:, w:2*w] = hr_img       # HR (ground truth)
        combined[:, 2*w:3*w] = pred_img   # Predicted
    
    # Save combined image
    combined_pil = Image.fromarray(combined)
    filename = f"epoch_{epoch+1:03d}_batch_{batch_idx:03d}_reconstruction.png"
    save_path = os.path.join(save_dir, filename)
    combined_pil.save(save_path)
    
    return save_path

def train(model: LAFINRModel, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, criterion: LAFINRCriterion, 
          num_epochs: int, device: torch.device) -> None:
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    best_psnr = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        last_residual_scale = 0.0  # Track residual scale from training loop
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_pbar):
            lr_batch = batch['lr'].to(device)
            hr_batch = batch['hr'].to(device)

            optimizer.zero_grad()
            output = model(lr_batch)
            losses = criterion(output['pred'], hr_batch, output['attn'], 
                              output.get('canvas_base'), output.get('residual'))
            loss = losses['loss']  
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            with torch.no_grad():
                batch_psnr = psnr(output['pred'], hr_batch)
                epoch_psnr += batch_psnr.item()
            num_batches += 1
            
            # Update progress bar
            residual_scale = output.get('residual_scale', torch.tensor(0.0)).item()
            last_residual_scale = residual_scale  # Cache for epoch summary
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{batch_psnr.item():.2f}dB",
                'Scale': f"{residual_scale:.4f}"
            })
            
            # Less frequent memory cleanup - only when really needed
            # PyTorch's allocator benefits from caching, so avoid frequent empty_cache()
            if batch_idx > 0 and batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate epoch averages
        avg_train_loss = epoch_loss / num_batches
        avg_train_psnr = epoch_psnr / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                lr_batch = batch['lr'].to(device)
                hr_batch = batch['hr'].to(device)
                
                output = model(lr_batch)
                losses = criterion(output['pred'], hr_batch, output['attn'], 
                                  output.get('canvas_base'), output.get('residual'))
                
                val_loss += losses['loss'].item()
                val_psnr_total += psnr(output['pred'], hr_batch).item()
                val_batches += 1
                
                # Save reconstruction results for first batch of each epoch
                if batch_idx == 0:
                    # Save first image from the batch
                    lr_img = lr_batch[0]    # Shape: [C, H, W]
                    hr_img = hr_batch[0]    # Shape: [C, H, W] 
                    pred_img = output['pred'][0]  # Shape: [C, H, W]
                    
                    save_path = save_reconstruction_results(
                        lr_img, hr_img, pred_img, epoch, batch_idx
                    )
                    print(f"  📸 Saved reconstruction: {save_path}")
                
                # Only validate on a subset for speed
                if val_batches >= 10:
                    break
        
        # Safety check for validation division
        if val_batches == 0:
            print("Warning: No validation batches processed!")
            avg_val_loss = float('inf')
            avg_val_psnr = 0.0
        else:
            avg_val_loss = val_loss / val_batches
            avg_val_psnr = val_psnr_total / val_batches
        
        # Print epoch summary using cached residual scale from training
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}dB")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f}dB")
        print(f"  Residual Scale: {last_residual_scale:.4f}")
        
        # Save best model
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'src/best_model.pth')
            print(f"  New best model saved! PSNR: {best_psnr:.2f}dB")
            print(f"  🎉 New best PSNR: {best_psnr:.2f}dB - Model saved!")
        
        print("-" * 60)

if __name__ == "__main__":
    # Clear all GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def conf(path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    config = conf('src/config.yml')
    
    dataset_train = LAFINRDataset(
        root=config['dataset_train']['root'],
        # scale=config['dataset_train']['scale'],
        patch_size=config['dataset_train']['patch_size'],
        crop_type=config['dataset_train']['crop_type'],
        color=config['dataset_train']['color'],
        augment=config['dataset_train']['augment'],
        split=config['dataset_train']['split'],
        recflag=config['dataset_train']['recflag']  # Add recflag parameter
    )
    train_dataloader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=1)  # Minimal batch size for 96x96 patches

    dataset_val = LAFINRDataset(
        root=config['dataset_val']['root'],
        # scale=config['dataset_val']['scale'],
        patch_size=config['dataset_val']['patch_size'],
        crop_type=config['dataset_val']['crop_type'],
        color=config['dataset_val']['color'],
        augment=config['dataset_val']['augment'],
        split= config['dataset_val']['split'],
        recflag=config['dataset_val']['recflag']  # Add recflag parameter
    )
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)  # Minimal workers

    model = LAFINRModel(
        num_rank_tokens=config['model']['num_rank_tokens'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        mlp_hidden=config['model']['mlp_hidden'],
        fourier_K=config['model']['fourier_K'],
        use_luma_canvas=config['model']['use_luma_canvas'],
        out_ch=config['model']['out_ch'],
        rank_start=config['model'].get('rank_start', 2),  # Default to 2 if not specified
        preserve_color=config['model'].get('preserve_color', False),  # Default to False
        residual_clamp=config['model'].get('residual_clamp', 0.05)   # Default to 0.05
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Smaller LR for stability
    criterion = LAFINRCriterion(
        w_rec=1.0,              # Reconstruction loss weight
        w_grad=0.02,            # Reduced gradient loss (was over-smoothing)
        w_ent=0.001,            # Attention entropy for diversity
        w_canvas=0.0,           # Removed canvas bias (doesn't backprop anyway)
        w_residual=0.0001       # Much reduced residual penalty (was discouraging improvements)
    )
    
    # Start training
    train(model, train_dataloader, val_dataloader, optimizer, criterion, 10, device)