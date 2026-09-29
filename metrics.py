import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim


def _to_numpy_hwc(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) torch tensor in [0,1] -> (H, W, C) numpy array."""
    t = t.detach().cpu().clamp(0, 1).numpy()
    return np.transpose(t, (1, 2, 0))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_numpy_hwc(pred)
    target_np = _to_numpy_hwc(target)
    return float(sk_psnr(target_np, pred_np, data_range=1.0))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = _to_numpy_hwc(pred)
    target_np = _to_numpy_hwc(target)
    channel_axis = 2 if pred_np.ndim == 3 and pred_np.shape[2] > 1 else None
    return float(sk_ssim(target_np, pred_np, data_range=1.0, channel_axis=channel_axis))


def batch_psnr_ssim(pred_batch: torch.Tensor, target_batch: torch.Tensor):
    """Average PSNR/SSIM over a batch of (B, C, H, W) tensors."""
    psnrs, ssims = [], []
    for p, t in zip(pred_batch, target_batch):
        psnrs.append(psnr(p, t))
        ssims.append(ssim(p, t))
    return float(np.mean(psnrs)), float(np.mean(ssims))
