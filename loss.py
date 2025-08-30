import torch
import torch.nn.functional as F

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3): super().__init__(); self.eps = eps
    def forward(self, x, y): return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

def grad_l1(x):
    # x: [B,C,H,W]
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return (dx.abs().mean() + dy.abs().mean()) / 2.0

def attn_entropy(attn, eps=1e-8, H=None, W=None):
    # attn: [B, Nq, Tk] probabilities
    p = attn.clamp_min(eps)
    H_attn = -(p * p.log()).sum(dim=-1).mean()  # mean over B,Nq
    return H_attn

class LAFINRCriterion(torch.nn.Module):
    def __init__(self, w_rec=1.0, w_grad=0.1, w_ent=0.01,
                 use_msssim=False, w_msssim=0.2,
                 w_canvas=0.0, w_tv=0.0, w_fft=0.0):
        super().__init__()
        self.rec = CharbonnierLoss()
        self.w_rec, self.w_grad, self.w_ent = w_rec, w_grad, w_ent
        self.use_msssim, self.w_msssim = use_msssim, w_msssim
        if use_msssim:
            from pytorch_msssim import MS_SSIM
            self.msssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)
        self.w_canvas, self.w_tv, self.w_fft = w_canvas, w_tv, w_fft

    def total_variation(self, a, H, W):
        # a: [B, Nq, Tk] -> [B,H,W,Tk]
        B, Nq, Tk = a.shape
        a2 = a.view(B, H, W, Tk).permute(0, 3, 1, 2)  # [B,Tk,H,W]
        tv = (a2[..., :, 1:] - a2[..., :, :-1]).abs().mean() + (a2[..., 1:, :] - a2[..., :-1, :]).abs().mean()
        return tv / 2.0

    def fft_highfreq(self, pred, tgt, frac=0.25):
        # compare log-mag at high frequencies
        Pf = torch.fft.fftshift(torch.fft.rfftn(pred, dim=(-2,-1)), dim=(-2,-1))
        Tf = torch.fft.fftshift(torch.fft.rfftn(tgt, dim=(-2,-1)), dim=(-2,-1))
        Pm, Tm = (Pf.abs() + 1e-8).log(), (Tf.abs() + 1e-8).log()
        H, W = pred.shape[-2:]
        cut_h, cut_w = int(H*frac), int((W//2+1)*frac)
        mask = torch.ones_like(Pm)
        mask[..., (H-cut_h)//2:(H+cut_h)//2, (Pm.shape[-1]-cut_w):] = 0  # keep upper-right high-freqs
        return (mask * (Pm - Tm).abs()).mean()

    def forward(self, pred, target, attn, canvas_hr=None):
        B, C, Hs, Ws = pred.shape
        losses = {}
        L = self.w_rec * self.rec(pred, target)
        losses['rec'] = L

        if self.w_grad > 0:
            lg = self.w_grad * grad_l1(pred)
            L = L + lg; losses['grad'] = lg

        if self.use_msssim and self.w_msssim > 0:
            lm = self.w_msssim * (1.0 - self.msssim(pred, target))
            L = L + lm; losses['msssim'] = lm

        if self.w_ent > 0:
            le = self.w_ent * attn_entropy(attn)
            L = L + le; losses['entropy'] = le

        if self.w_tv > 0:
            ltv = self.w_tv * self.total_variation(attn, Hs, Ws)
            L = L + ltv; losses['attn_tv'] = ltv

        if canvas_hr is not None and self.w_canvas > 0:
            # luma consistency: blur both and compare Y channel
            k = 11
            blur = torch.nn.AvgPool2d(k, stride=1, padding=k//2)
            if C == 3:
                Yp = 0.299*pred[:,0]+0.587*pred[:,1]+0.114*pred[:,2]
                Yc = canvas_hr.squeeze(1) if canvas_hr.shape[1]==1 else (0.299*canvas_hr[:,0]+0.587*canvas_hr[:,1]+0.114*canvas_hr[:,2])
            else:
                Yp, Yc = pred[:,0], canvas_hr.squeeze(1)
            lc = self.w_canvas * F.mse_loss(blur(Yp.unsqueeze(1)), blur(Yc.unsqueeze(1)))
            L = L + lc; losses['canvas'] = lc

        if self.w_fft > 0:
            lf = self.w_fft * self.fft_highfreq(pred, target)
            L = L + lf; losses['fft'] = lf

        losses['loss'] = L
        return losses
