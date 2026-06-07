import os
from tqdm import tqdm
import numpy as np
import torch
import nibabel as nib
import torch.amp
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def calculate_crop_params(target_size, source_size):
    """Compute (start, end) crop indices that center-crop a `source_size` volume down to `target_size`."""
    if len(target_size) != 3 or len(source_size) != 3:
        raise ValueError("target_size and source_size must be 3D tuples (D, H, W)")
    crop_needed = [s - t for t, s in zip(target_size, source_size)]
    if any(c < 0 for c in crop_needed):
        raise ValueError(f"Target crop size {target_size} is larger than source size {source_size} in at least one dimension.")
    crop_indices = []
    for idx, dim_crop in enumerate(crop_needed):
        start = dim_crop // 2
        end = start + target_size[idx]
        crop_indices.append((start, end))
    logger.info(f"Calculated crop indices to get from {source_size} to {target_size}: D={crop_indices[0]}, H={crop_indices[1]}, W={crop_indices[2]}")
    return crop_indices

def generate_conditioned_brains(wfm_model, conditions_list, num_samples_per_condition, output_dir, save_size, model_output_size, condition_ranges):
    os.makedirs(output_dir, exist_ok=True) 
    device = next(wfm_model.flow_net.parameters()).device 
    wfm_model.flow_net.eval()
    do_crop = False
    crop_indices = None
    if save_size != model_output_size:
        try: 
            crop_indices = calculate_crop_params(save_size, model_output_size)
            do_crop = True
            logger.info(f"Will crop generated images from {model_output_size} to {save_size}.")
        except ValueError as e: 
            logger.error(f"Cannot crop: {e}. Saving full size {model_output_size}.")
            do_crop = False
    logger.info(f"Generating {num_samples_per_condition} samples for {len(conditions_list)} conditions.")

    for cond_idx, conditions_dict_orig in enumerate(tqdm(conditions_list, desc="Generating Conditions")):
        subdir_name_parts = []
        valid_conditions_for_model = True
        model_conditions = {}
        for k, v in sorted(conditions_dict_orig.items()):
             try:
                 value_str = f"{v:.2f}"
             except (TypeError, ValueError):
                 value_str = str(v)
             subdir_name_parts.append(f"{k}_{value_str}")
        subdir_name = "_".join(subdir_name_parts) if subdir_name_parts else f"condition_{cond_idx}"
        cond_dir = os.path.join(output_dir, subdir_name) 
        os.makedirs(cond_dir, exist_ok=True)

        for k_orig, v_orig in conditions_dict_orig.items():
             if k_orig in wfm_model.condition_dims:
                 if k_orig in condition_ranges:
                     min_v, max_v = condition_ranges[k_orig]['min'], condition_ranges[k_orig]['max']
                     v_norm = (v_orig - min_v) / (max_v - min_v) if max_v > min_v else 0.5
                     # Clip to [0, 1] to match training-time normalization (avoids silent extrapolation).
                     v_norm = float(np.clip(v_norm, 0.0, 1.0))
                 else:
                    logger.warning(f"Range not found for condition '{k_orig}'. Assuming value is already normalized or using raw.")
                    v_norm = v_orig
                 cond_dim = wfm_model.condition_dims[k_orig]
                 model_conditions[k_orig] = torch.tensor([[v_norm] * cond_dim], dtype=torch.float32, device='cpu')
             else: 
                logger.warning(f"Condition key '{k_orig}' provided but not in model's condition_dims. Skipping set.")
                valid_conditions_for_model = False
                break
        if not valid_conditions_for_model: 
            continue

        for k_model, dim in wfm_model.condition_dims.items():
            if k_model not in model_conditions: model_conditions[k_model] = torch.full((1, dim), 0.5, dtype=torch.float32, device='cpu')

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            model_conditions_dev = {k: v.to(device) for k, v in model_conditions.items()}
            synthetic_images = wfm_model.sample(num_samples_per_condition, model_output_size, model_conditions_dev)

        if do_crop:
            try:
                (d_start, d_end), (h_start, h_end), (w_start, w_end) = crop_indices
                synthetic_images_final = synthetic_images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                if synthetic_images_final.shape[-3:] != save_size: 
                    logger.warning(f"Cropped shape {synthetic_images_final.shape[-3:]} != target {save_size}.")
            except Exception as e: 
                logger.error(f"Error during cropping: {e}. Saving uncropped.", exc_info=True)
                synthetic_images_final = synthetic_images
        else: 
            synthetic_images_final = synthetic_images

        condition_str = ", ".join([f"{k}: {v:.2f}" for k, v in conditions_dict_orig.items()])
        logger.info(f"Saving {num_samples_per_condition} samples for {condition_str} to {cond_dir} (Shape: {synthetic_images_final.shape[-3:]})")

        synthetic_images_final = (synthetic_images_final + 1.0) / 2.0
        synthetic_images_final = torch.clamp(synthetic_images_final, 0.0, 1.0)

        for i in range(num_samples_per_condition):
            try:
                img_data = synthetic_images_final[i, 0].cpu().numpy().astype(np.float32)
                img_nifti = nib.Nifti1Image(img_data, affine=np.eye(4))
                save_path = os.path.join(cond_dir, f"FlowLet_synthetic_brain_{i}.nii.gz")
                nib.save(img_nifti, save_path)
            except Exception as e: logger.error(f"Error saving NIfTI sample {i} for {subdir_name}: {e}", exc_info=True)
    logger.info(f"Finished generating brains in {output_dir}")