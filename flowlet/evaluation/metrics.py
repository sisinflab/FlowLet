import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def evaluate_recon_quality(real_images, generated_images):
    real_images = real_images.detach().float().cpu()
    generated_images = generated_images.detach().float().cpu()
    # Map images from [-1, 1] to [0, 1]; metrics then use a data range of 1.0.
    real_np = (real_images.numpy() + 1.0) / 2.0
    gen_np = (generated_images.numpy() + 1.0) / 2.0
    data_range_metric = 1.0
    batch_size = real_np.shape[0]
    if real_np.ndim == 5 and real_np.shape[1] == 1: 
        real_np = real_np.squeeze(1)
    if gen_np.ndim == 5 and gen_np.shape[1] == 1: 
        gen_np = gen_np.squeeze(1)
    if real_np.ndim != 4 or gen_np.ndim != 4: 
        logger.warning(f"Unexpected shape for metrics: Real {real_images.shape}, Gen {generated_images.shape}. Expected 4D.")
        return 0.0, 0.0

    psnr_vals = []
    ssim_vals = []
    for i in range(batch_size):
        try:
            psnr_val = psnr(real_np[i], gen_np[i], data_range=data_range_metric)
            psnr_vals.append(psnr_val)
            win_size = min(7, *real_np[i].shape)
            if win_size % 2 == 0: 
                win_size -= 1
            if win_size < 3: 
                logger.warning(f"Skipping SSIM for sample {i} due to small dimensions: {real_np[i].shape}")
                ssim_vals.append(0.0)
                continue
            ssim_val = ssim(real_np[i], gen_np[i], data_range=data_range_metric, channel_axis=None, win_size=win_size, gaussian_weights=True)
            ssim_vals.append(ssim_val)
        except ValueError as ve: 
            logger.error(f"ValueError calculating PSNR/SSIM for sample {i}: {ve}", exc_info=True)
            psnr_vals.append(0.0)
            ssim_vals.append(0.0)
        except Exception as e: 
            logger.error(f"Error during PSNR/SSIM calculation for sample {i}: {e}", exc_info=True)
            psnr_vals.append(0.0)
            ssim_vals.append(0.0)
    avg_psnr = np.mean(psnr_vals) if psnr_vals else 0.0
    avg_ssim = np.mean(ssim_vals) if ssim_vals else 0.0
    return avg_psnr, avg_ssim