import torch
import numpy as np
import wandb
from ..utils.logging_utils import get_logger
from ..wavelets import idwt_3d
from .metrics import evaluate_recon_quality

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_flow_generation(wfm_model, val_loader, model_output_size, use_wandb=True, epoch_num=None):
    wfm_model.flow_net.eval()
    try:
        batch = next(iter(val_loader))
        if batch is None: 
            logger.warning("Skipping viz, received None batch from val_loader.")
            return
        real_wavelet_batch, conditions_dict = batch
        if real_wavelet_batch is None or real_wavelet_batch.nelement() == 0: 
            logger.warning("Skipping viz, empty wavelet batch.")
            return
        real_wavelet_batch = real_wavelet_batch[:1].to(device)
        conditions_dict = {k: v[:1].to(device) for k, v in conditions_dict.items()} if conditions_dict else {}
        batch_size = real_wavelet_batch.size(0)
        if batch_size == 0: 
            logger.warning("Skipping viz, batch size is 0.")
            return

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            coeffs_tuple = torch.split(real_wavelet_batch, 1, dim=1)
            real_batch = idwt_3d(coeffs_tuple[0] * 1.0, *coeffs_tuple[1:])
            real_batch = torch.clamp(real_batch, -1.0, 1.0)
            synthetic_batch = wfm_model.sample(batch_size, model_output_size, conditions_dict)

        mid_slice_idx = real_batch.shape[2] // 2; epoch_str = f"_Ep{epoch_num}" if epoch_num else ""
        real_slice_log = (real_batch[0, 0, mid_slice_idx].cpu().numpy() + 1.0) / 2.0
        synth_slice_log = (synthetic_batch[0, 0, mid_slice_idx].cpu().numpy() + 1.0) / 2.0
        psnr_synth, ssim_synth = evaluate_recon_quality(real_batch, synthetic_batch)

        if use_wandb:
             log_dict = {}
             if epoch_num == 1: 
                 log_dict[f"Gen Comparison/Real Sample{epoch_str}"] = wandb.Image(real_slice_log, caption="Real Sample (Mid Slice)")
             log_dict[f"Gen Comparison/Flow Sample{epoch_str}"] = wandb.Image(synth_slice_log, caption=f"FlowLet Gen (PSNR:{psnr_synth:.2f} SSIM:{ssim_synth:.4f})")
             wandb.log(log_dict)
        else: 
            logger.info(f"Gen Viz (Epoch {epoch_num}): Synth PSNR={psnr_synth:.2f}, SSIM={ssim_synth:.4f}")
    except StopIteration: logger.warning("Validation loader exhausted, cannot visualize generation.")
    except Exception as e: logger.error(f"Error during flow generation visualization: {e}", exc_info=True)

def visualize_multi_condition_samples(wfm_model, num_samples, model_output_size, wandb_log=True, condition_key_to_vary='Age', condition_ranges=None, epoch_num=None):
    wfm_model.flow_net.eval()
    device = next(wfm_model.flow_net.parameters()).device
    if not wfm_model.condition_dims: 
        logger.info("Model is not conditional. Skipping multi-condition viz.")
        return
    
    if condition_key_to_vary not in wfm_model.condition_dims: 
        logger.warning(f"Condition key '{condition_key_to_vary}' not found in model's condition_dims. Skipping.")
        return
    
    if condition_ranges is None or condition_key_to_vary not in condition_ranges: 
        logger.warning(f"Condition ranges not provided for '{condition_key_to_vary}'. Using normalized values [0.1, 0.9].")
        min_v, max_v = 0.0, 1.0
    else: 
        min_v, max_v = condition_ranges[condition_key_to_vary]['min'], condition_ranges[condition_key_to_vary]['max']

    norm_values = torch.linspace(0.1, 0.9, 5, device=device)
    images_to_log = {}
    epoch_str = f"_Ep{epoch_num}" if epoch_num else ""
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        for val_idx, norm_val in enumerate(norm_values):
            conditions = {condition_key_to_vary: norm_val.unsqueeze(0)}
            for k, dim in wfm_model.condition_dims.items():
                 if k != condition_key_to_vary: conditions[k] = torch.full((1, dim), 0.5, dtype=torch.float32, device=device)
            synthetic_images = wfm_model.sample(num_samples, model_output_size, conditions)
            mid_slice_idx = synthetic_images.shape[2] // 2
            img_log = (synthetic_images[0, 0, mid_slice_idx].cpu().numpy() + 1.0) / 2.0
            orig_val = min_v + norm_val.item() * (max_v - min_v) if max_v > min_v else min_v
            caption = f"{condition_key_to_vary}={orig_val:.1f} (N={norm_val.item():.2f})"
            images_to_log[f"MultiCond_{condition_key_to_vary}/Sample_{val_idx}{epoch_str}"] = wandb.Image(img_log, caption=caption)
    
    if wandb_log and images_to_log: 
        wandb.log(images_to_log)
        logger.info(f"Logged multi-condition samples for '{condition_key_to_vary}'.")