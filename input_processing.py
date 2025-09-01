import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

class FourierCoordEncoder(nn.Module):
    """
    Enhanced FourierCoordEncoder for atomic layer decomposition.
    
    Encodes 2D coordinates (x, y) from the unit square [0, 1]^2 into a higher-dimensional 
    feature space using multi-band Fourier (sin/cos) features. Enhanced to support 
    rank-aware coordinate encoding for atomic layer modeling.
    
    Args:
        num_freqs (int): Number of frequency bands to use for encoding.
        include_input (bool): If True, the original (x, y) coordinates are concatenated.
        rank_aware (bool): If True, enable rank-specific coordinate encodings.
        max_rank (int): Maximum rank for rank-aware encoding.
    """
    def __init__(self, num_freqs: int = 14, include_input: bool = False, 
                 rank_aware: bool = False, max_rank: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.rank_aware = rank_aware
        self.max_rank = max_rank
        
        # Base output dimension
        self.base_out_dim = 4 * num_freqs + (2 if include_input else 0)
        
        # Enhanced output dimension for rank-aware encoding
        if rank_aware:
            self.rank_embedding_dim = 16
            self.out_dim = self.base_out_dim + self.rank_embedding_dim
            # Learnable rank embeddings
            self.rank_embeddings = nn.Embedding(max_rank, self.rank_embedding_dim)
        else:
            self.out_dim = self.base_out_dim
        
        # Frequency bands for Fourier encoding
        self.register_buffer(
            "omegas",
            torch.tensor([(2.0 ** k) * math.pi for k in range(num_freqs)], dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def grid(self, B: int, Hs: int, Ws: int, device: torch.device) -> torch.Tensor:
        """Generate coordinate grid normalized to [0, 1]."""
        try:
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, Hs, device=device),
                torch.linspace(0, 1, Ws, device=device),
                indexing="ij",
            )
            coords = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            return coords  # [B,Hs,Ws,2]
        except Exception as e:
            print(f"Error in FourierCoordEncoder.grid: {e}")
            print(f"B={B}, Hs={Hs}, Ws={Ws}, device={device}")
            raise

    def forward(self, coords_xy: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
        """
        Encode coordinates with optional rank-awareness.
        
        Args:
            coords_xy: Input coordinates [..., 2] in [0, 1]^2
            rank: Optional rank index for rank-aware encoding
        Returns:
            Encoded features [..., out_dim]
        """
        try:
            if coords_xy is None:
                raise ValueError("coords_xy cannot be None")
                
            x, y = coords_xy[..., 0], coords_xy[..., 1]
            
            # Multi-band Fourier encoding
            xw = x.unsqueeze(-1) * self.omegas  # [..., num_freqs]
            yw = y.unsqueeze(-1) * self.omegas  # [..., num_freqs]
            
            feats = [torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)]
            out = torch.cat(feats, dim=-1)  # [..., 4*num_freqs]
            
            # Include original coordinates if requested
            if self.include_input:
                out = torch.cat([coords_xy, out], dim=-1)
            
            # Add rank-specific encoding if enabled
            if self.rank_aware and rank is not None:
                rank_tensor = torch.full(
                    coords_xy.shape[:-1] + (1,), 
                    rank, 
                    device=coords_xy.device, 
                    dtype=torch.long
                )
                rank_emb = self.rank_embeddings(rank_tensor.squeeze(-1))  # [..., rank_embedding_dim]
                out = torch.cat([out, rank_emb], dim=-1)
            elif self.rank_aware:
                # Pad with zeros if rank-aware but no rank provided
                zero_pad = torch.zeros(
                    coords_xy.shape[:-1] + (self.rank_embedding_dim,),
                    device=coords_xy.device,
                    dtype=out.dtype
                )
                out = torch.cat([out, zero_pad], dim=-1)
            
            return out
            
        except Exception as e:
            print(f"Error in FourierCoordEncoder.forward: {e}")
            print(f"coords_xy shape: {coords_xy.shape if coords_xy is not None else None}")
            print(f"rank: {rank}")
            raise
        xw = x.unsqueeze(-1) * self.omegas
        yw = y.unsqueeze(-1) * self.omegas
        feats = [torch.sin(xw), torch.cos(xw), torch.sin(yw), torch.cos(yw)]
        out = torch.cat(feats, dim=-1)
        if self.include_input:
            out = torch.cat([coords_xy, out], dim=-1)
            print(f"Error in FourierCoordEncoder.forward: {e}")
            print(f"coords_xy shape: {coords_xy.shape if coords_xy is not None else None}")
            print(f"rank: {rank}")
            raise


class AtomicLayerPatchSampler(nn.Module):
    """
    Enhanced patch sampler for atomic layer analysis.
    
    Samples patches from rank-1 canvas or original image for local spectral analysis.
    Useful for analyzing local singular spectrum and adaptive rank selection.
    """
    def __init__(self, patch_size: int = 16, stride: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, image: torch.Tensor, coords: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Sample patches from image for local analysis.
        
        Args:
            image: [B, C, H, W] input image (can be rank-1 canvas or original)
            coords: Optional [B, N, 2] specific coordinates to sample around
        Returns:
            Dict containing patches and metadata
        """
        B, C, H, W = image.shape
        
        if coords is not None:
            # Sample patches around specific coordinates
            patches = self._sample_at_coords(image, coords)
        else:
            # Regular grid sampling
            patches = self._sample_grid(image)
        
        return {
            'patches': patches,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'source_shape': (H, W)
        }
    
    def _sample_at_coords(self, image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Sample patches around specific coordinates."""
        # Implementation for coordinate-based sampling
        # This is a simplified version - full implementation would handle edge cases
        B, C, H, W = image.shape
        N = coords.shape[1]
        half_patch = self.patch_size // 2
        
        patches = []
        for b in range(B):
            for n in range(N):
                x, y = coords[b, n]
                # Convert normalized coords to pixel coords
                px, py = int(x * W), int(y * H)
                
                # Extract patch with boundary handling
                x1, x2 = max(0, px - half_patch), min(W, px + half_patch)
                y1, y2 = max(0, py - half_patch), min(H, py + half_patch)
                
                patch = image[b, :, y1:y2, x1:x2]
                # Pad if necessary to maintain patch size
                if patch.shape[-2:] != (self.patch_size, self.patch_size):
                    patch = torch.nn.functional.pad(
                        patch, 
                        (0, self.patch_size - patch.shape[-1], 
                         0, self.patch_size - patch.shape[-2]),
                        mode='reflect'
                    )
                patches.append(patch)
        
        return torch.stack(patches).view(B, N, C, self.patch_size, self.patch_size)
    
    def _sample_grid(self, image: torch.Tensor) -> torch.Tensor:
        """Sample patches on a regular grid."""
        B, C, H, W = image.shape
        
        # Use unfold to extract patches
        patches = image.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        # Reshape to [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
        return patches
