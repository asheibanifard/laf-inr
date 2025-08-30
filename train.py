import torch
import tqdm
import yaml

from dataset import LAFINRDataset
from torch.utils.data import DataLoader
from model import LAFINRModel
from loss import LAFINRCriterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def psnr(pred, target):
    """Calculate PSNR between two tensors"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

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
        
        train_pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in train_pbar:
            lr_batch = batch['lr'].to(device)
            hr_batch = batch['hr'].to(device)

            optimizer.zero_grad()
            output = model(lr_batch)
            losses = criterion(output['pred'], hr_batch, output['attn'])
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
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{batch_psnr.item():.2f}dB"
            })
        
        # Calculate epoch averages
        avg_train_loss = epoch_loss / num_batches
        avg_train_psnr = epoch_psnr / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr_total = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lr_batch = batch['lr'].to(device)
                hr_batch = batch['hr'].to(device)
                
                output = model(lr_batch)
                losses = criterion(output['pred'], hr_batch, output['attn'])
                
                val_loss += losses['loss'].item()
                val_psnr_total += psnr(output['pred'], hr_batch).item()
                val_batches += 1
                
                # Only validate on a subset for speed
                if val_batches >= 10:
                    break
        
        avg_val_loss = val_loss / val_batches
        avg_val_psnr = val_psnr_total / val_batches
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.2f}dB")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f}dB")
        
        # Save best model
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  🎉 New best PSNR: {best_psnr:.2f}dB - Model saved!")
        
        print("-" * 60)

if __name__ == "__main__":
    def conf(path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    config = conf('src/config.yml')
    
    dataset_train = LAFINRDataset(
        root=config['dataset_train']['root'],
        scale=config['dataset_train']['scale'],
        patch_size=config['dataset_train']['patch_size'],
        crop_type=config['dataset_train']['crop_type'],
        color=config['dataset_train']['color'],
        augment=config['dataset_train']['augment'],
        split=config['dataset_train']['split']
    )
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)

    dataset_val = LAFINRDataset(
        root=config['dataset_val']['root'],
        scale=config['dataset_val']['scale'],
        patch_size=config['dataset_val']['patch_size'],
        crop_type=config['dataset_val']['crop_type'],
        color=config['dataset_val']['color'],
        augment=config['dataset_val']['augment'],
        split= config['dataset_val']['split']
    )
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    model = LAFINRModel(
        scale=config['model']['scale'],
        num_rank_tokens=config['model']['num_rank_tokens'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        mlp_hidden=config['model']['mlp_hidden'],
        fourier_K=config['model']['fourier_K'],
        use_luma_canvas=config['model']['use_luma_canvas'],
        out_ch=config['model']['out_ch']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LAFINRCriterion(
        w_rec=1.0,              # Reconstruction loss weight
        w_grad=0.0,             # No gradient loss
        w_ent=0.000001          # Ultra-low sparsity (1000x reduction)
    )
    
    # Start training
    train(model, train_dataloader, val_dataloader, optimizer, criterion, 10, device)