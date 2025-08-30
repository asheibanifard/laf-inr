from glob import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch.utils.data import Dataset
from PIL import Image

class LAFINRDataset(Dataset):
    """Lightweight dataset for the cross-attn LAF-INR variant.

    It expects HR images under `root/{train,val}` (or directly under `root` if split=None),
    crops them to be divisible by `scale`, and creates LR by bicubic downsampling.

    Returns per item:
      {
        'lr':  [C,H,W] float32 in [0,1],
        'hr':  [C,Hs,Ws] float32 in [0,1],  (Hs=H*scale, Ws=W*scale)
        'path': str (file path)
      }
    """
    def __init__(
        self,
        root: str,
        split: Optional[str] = 'train',
        scale: int = 4,
        patch_size: Optional[int] = None,
        crop_type: str = 'random',   # 'random' or 'center'
        exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'),
        color: str = 'RGB',
        augment: bool = False,
    ):
        super().__init__()
        self.root = root
        self.scale = int(scale)
        self.patch_size = patch_size
        self.crop_type = crop_type
        self.color = color
        self.augment = augment
        search_dir = os.path.join(root, split) if split else root
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(search_dir, f'**/*{ext}'), recursive=True))
        self.files = sorted(files)
        # Optional index-based split across ALL images under root (ignores subfolders)
        if split in ('train', 'val'):
            all_files = []
            for ext in exts:
                all_files.extend(glob(os.path.join(root, f"**/*{ext}"), recursive=True))
            all_files = sorted(all_files)
            if split == 'train':
                self.files = all_files[:min(800, len(all_files))]
            else:  # 'val'
                self.files = all_files[min(800, len(all_files)) : min(900, len(all_files))]
        if len(self.files) == 0:
            raise FileNotFoundError(f'No images found in {search_dir} with extensions {exts}')

    # ---------- helpers ----------
    def _open(self, path: str) -> Image.Image:
        im = Image.open(path)
        if im.mode != self.color:
            im = im.convert(self.color)
        return im

    def _to_tensor01(self, im: Image.Image) -> torch.Tensor:
        arr = np.array(im)
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:
            arr = arr.astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    def _crop_to_multiple(self, im: Image.Image, size: Optional[int]) -> Image.Image:
        w, h = im.size
        s = self.scale
        if size is None:
            # full image, center-crop minimally so both dims are divisible by scale
            ww = (w // s) * s
            hh = (h // s) * s
            l = (w - ww) // 2
            t = (h - hh) // 2
            return im.crop((l, t, l + ww, t + hh))
        # patch mode: choose a square patch size, then snap down to multiple of scale
        ps = min(size, w, h)
        ps = max(s, ps - (ps % s))
        if self.crop_type == 'center':
            l = (w - ps) // 2
            t = (h - ps) // 2
        else:
            # random
            import random
            l = 0 if w == ps else random.randint(0, w - ps)
            t = 0 if h == ps else random.randint(0, h - ps)
        return im.crop((l, t, l + ps, t + ps))

    def _maybe_aug(self, hr: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return hr
        if torch.rand(1) < 0.5:
            hr = torch.flip(hr, dims=[2])  # horizontal
        if torch.rand(1) < 0.5:
            hr = torch.flip(hr, dims=[1])  # vertical
        if torch.rand(1) < 0.5:
            hr = hr.transpose(1, 2)        # rotate 90
        return hr

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        hr_pil = self._open(path)
        hr_pil = self._crop_to_multiple(hr_pil, self.patch_size)

        # HR tensor [C,Hs,Ws]
        hr = self._to_tensor01(hr_pil)
        hr = self._maybe_aug(hr)

        # Make LR by bicubic downsample to exact multiple
        C, Hs, Ws = hr.shape
        H, W = Hs // self.scale, Ws // self.scale
        lr = F.interpolate(hr.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)

        return {'lr': lr, 'hr': hr, 'path': path}

