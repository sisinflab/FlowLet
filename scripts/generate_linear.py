import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import nibabel as nib
import torch.amp

from flowlet.models import WaveletFlowMatching
from flowlet.generation.generator import calculate_crop_params
from flowlet.utils import setup_logging, set_seed, get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Samples with Linearly Interpolated Age using FlowLet")

    # --- Model/Checkpoint Args ---
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pth file, e.g., fmw_best.pth).")
    parser.add_argument("--config_path", type=str, default=None, help="(Optional) Path to the model configuration JSON file. If not provided, attempts to infer from checkpoint_path directory.")

    # --- Generation Args ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated NIfTI files.")
    parser.add_argument("--num_total_samples", type=int, default=3000, help="Total number of samples to generate across the Age range.")
    parser.add_argument("--min_age", type=float, required=True, help="The minimum original Age for the linear interpolation range.")
    parser.add_argument("--max_age", type=float, required=True, help="The maximum original Age for the linear interpolation range.")
    parser.add_argument("--condition_ranges_path", type=str, default=None, help="Path to JSON file containing condition ranges (min/max) used during *training* for normalization. Crucial for correct Age mapping.")
    parser.add_argument("--save_size", type=int, nargs=3, default=[91, 109, 91], metavar=('D', 'H', 'W'), help="Spatial size to crop generated images to before saving.")
    parser.add_argument("--model_input_size", type=int, nargs=3, default=None, metavar=('D', 'H', 'W'), help="Model's expected padded input size (D, H, W). Required if not found in config.")
    parser.add_argument("--filename_prefix", type=str, default="FlowLet", help="Prefix for the output filenames.")
    parser.add_argument("--num_flow_steps", type=int, default=10, help="Number of flow steps for the model. Default is 10.")
    # --- System Args ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation noise (for reproducibility).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    if args.min_age >= args.max_age:
        parser.error("--min_age must be strictly less than --max_age")

    return args

def load_config_from_checkpoint_dir(checkpoint_path):
    """Tries to find and load a config.json from the checkpoint's directory."""
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded model configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Found config.json at {config_path} but failed to load: {e}. Need manual config.")
            return None
    else:
        logger.warning(f"No config.json found in checkpoint directory ({os.path.dirname(checkpoint_path)}). Need manual config.")
        return None

def main():
    args = parse_args()

    # --- Setup ---
    setup_logging(log_dir=args.output_dir, filename_prefix="flowlet_generate_linear_ablation") # Log to main output dir
    logger.info(f"Starting linear Age generation for ablation study.")
    logger.info(f"Generating {args.num_total_samples} samples for Age range [{args.min_age:.2f}, {args.max_age:.2f}] across different modes.")
    logger.info(f"Base output directory for modes: {args.output_dir}")

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load Configuration ---
    model_config = None
    if args.config_path and os.path.exists(args.config_path):
        try:
            with open(args.config_path, 'r') as f:
                model_config = json.load(f)
            logger.info(f"Loaded model configuration from specified path: {args.config_path}")
        except Exception as e:
            logger.error(f"Failed to load specified config file {args.config_path}: {e}", exc_info=True)
            return
    elif args.checkpoint_path:
         model_config = load_config_from_checkpoint_dir(args.checkpoint_path)

    if model_config is None:
        logger.error("Model configuration could not be loaded. Provide a valid --config_path, or "
                     "ensure a config.json sits next to the checkpoint. Refusing to guess the U-Net "
                     "architecture, since a mismatch with the checkpoint would corrupt the loaded weights.")
        return


    # Extract necessary info from config
    try:
        model_input_size = tuple(model_config['model_input_size'])
        condition_vars = model_config.get('condition_vars', [])
        if 'Age' not in condition_vars:
            logger.warning(f"'Age' not found in model's condition_vars: {condition_vars}. Model might not be Age-conditional.")


        attention_res = tuple(map(int, model_config['unet_attention_res'].split(',')))
        channel_mult = tuple(map(int, model_config['unet_channel_mult'].split(',')))
        # Use num_flow_steps from args if provided, otherwise from config, otherwise default
        num_flow_steps_model_init = args.num_flow_steps if hasattr(args, 'num_flow_steps') and args.num_flow_steps is not None else model_config.get('num_flow_steps', 10)

    except KeyError as e:
        logger.error(f"Missing essential key in loaded/constructed config: {e}")
        return
    except Exception as e:
        logger.error(f"Error parsing configuration values: {e}")
        return

    # --- Load Condition Ranges (CRITICAL for Normalization) ---
    condition_ranges = None
    if args.condition_ranges_path and os.path.exists(args.condition_ranges_path):
        try:
            with open(args.condition_ranges_path, 'r') as f:
                condition_ranges = json.load(f)
            logger.info(f"Loaded condition ranges from: {args.condition_ranges_path}")
        except Exception as e:
            logger.warning(f"Failed to load condition ranges file {args.condition_ranges_path}: {e}.", exc_info=True)
    else:
        # Try finding ranges in checkpoint dir
        ranges_path_alt = os.path.join(os.path.dirname(args.checkpoint_path), "condition_ranges.json")
        if os.path.exists(ranges_path_alt):
            try:
                with open(ranges_path_alt, 'r') as f:
                    condition_ranges = json.load(f)
                logger.info(f"Loaded condition ranges from checkpoint directory: {ranges_path_alt}")
            except Exception as e:
                 logger.warning(f"Failed to load condition_ranges.json from checkpoint dir: {e}.", exc_info=True)

    if condition_ranges is None or 'Age' not in condition_ranges:
        logger.error("Condition ranges for 'Age' could not be loaded. These are REQUIRED to normalize the target Age for the model. Please provide a valid --condition_ranges_path pointing to the JSON file saved during training.")
        return
    else:
        logger.info(f"Using Age normalization range: Min={condition_ranges['Age']['min']:.2f}, Max={condition_ranges['Age']['max']:.2f}")


    # --- Load Model ---
    logger.info(f"Loading model checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return

    try:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        condition_dims_dict = {var: 1 for var in condition_vars} if condition_vars else {}
        unet_args = {
            "in_channels": 8, "model_channels": model_config.get('unet_model_channels', 128), "out_channels": 8,
            "num_res_blocks": model_config.get('unet_num_res_blocks', 2),
            "attention_resolutions": attention_res,
            "dropout": model_config.get('unet_dropout', 0.1),
            "channel_mult": channel_mult,
            "conv_resample": True, "dims": 3,
            "use_checkpoint": model_config.get('use_checkpointing', False), # Should be False for inference
            "num_heads": model_config.get('unet_num_heads', 8),
            "num_head_channels": model_config.get('unet_num_head_channels', -1),
            "use_scale_shift_norm": True, "resblock_updown": True,
            "condition_dims": condition_dims_dict,
            "condition_embedding_dim": model_config.get('condition_embedding_dim', 512),
            "use_xformers": model_config.get('use_xformers', True),
            "use_cross_attention": not model_config.get('unet_disable_cross_attn', False) and bool(condition_dims_dict),
            "norm_num_groups": model_config.get('unet_norm_num_groups', 32), "norm_eps": 1e-6,
        }
        # Pass num_flow_steps from args/config to model for its default sampling behavior
        wfm_model = WaveletFlowMatching(u_net_args=unet_args, num_flow_steps=num_flow_steps_model_init).to(device)
        wfm_model.flow_net.use_checkpoint = False

        state_dict = ckpt.get("flow_net_state_dict", ckpt.get("model_state_dict", ckpt))
        if not state_dict: raise KeyError("Could not find a model state dictionary in the checkpoint.")

        is_currently_compiled = hasattr(wfm_model.flow_net, '_orig_mod')
        is_saved_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        if not is_currently_compiled and is_saved_compiled:
            logger.info("Saved state_dict is compiled. Removing '_orig_mod.' prefix.")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        wfm_model.flow_net.load_state_dict(state_dict)
        wfm_model.eval()
        logger.info(f"Model loaded successfully from epoch {ckpt.get('epoch', -1)+1}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # --- Prepare Cropping ---
    do_crop = False
    crop_indices = None
    save_size_tuple = tuple(args.save_size)
    if save_size_tuple != model_input_size:
        try:
            crop_indices = calculate_crop_params(save_size_tuple, model_input_size)
            do_crop = True
            logger.info(f"Will crop generated images from {model_input_size} to {save_size_tuple}.")
        except ValueError as e:
            logger.error(f"Cannot crop: {e}. Saving full size {model_input_size}.")
            do_crop = False

    # --- Generate Target Ages ---
    target_ages = np.linspace(args.min_age, args.max_age, args.num_total_samples)
    logger.info(f"Generated {len(target_ages)} target ages using linspace.")

    # --- Generation Loop ---
    train_min_age = condition_ranges['Age']['min']
    train_max_age = condition_ranges['Age']['max']
    age_range_norm = train_max_age - train_min_age # Renamed to avoid conflict
    if age_range_norm <= 0:
        logger.error(f"Invalid training Age range from condition_ranges.json: [{train_min_age}, {train_max_age}]")
        return

    # Define generation modes for ablation
    generation_modes = {
        "baseline": {"disable_cross_attn": False, "disable_cond_film": False, "suffix": "Baseline"},
        "film_only": {"disable_cross_attn": True, "disable_cond_film": False, "suffix": "FiLM_Only"},
        "crossattn_only": {"disable_cross_attn": False, "disable_cond_film": True, "suffix": "CrossAttn_Only"},
        "unconditional": {"disable_cross_attn": True, "disable_cond_film": True, "suffix": "Unconditional"},
    }

    for mode_name, mode_flags in generation_modes.items():
        logger.info(f"--- Generating samples for mode: {mode_name} ---")
        # Create a unique subdirectory for this mode's outputs
        current_output_dir_name = f"{args.filename_prefix}_{mode_flags['suffix']}_AgeLin_{args.min_age:.1f}-{args.max_age:.1f}_N{args.num_total_samples}_Steps{args.num_flow_steps}"
        current_output_dir = os.path.join(args.output_dir, current_output_dir_name)
        os.makedirs(current_output_dir, exist_ok=True)
        logger.info(f"Output for this mode: {current_output_dir}")

        pbar = tqdm(enumerate(target_ages), total=args.num_total_samples, desc=f"Generating {mode_name}")
        for i, target_age in pbar:
            pbar.set_postfix({"Age": f"{target_age:.2f}"})

            # --- Prepare Condition for this sample ---
            norm_age = (target_age - train_min_age) / age_range_norm
            norm_age_clipped = np.clip(norm_age, 0.0, 1.0)
            if norm_age != norm_age_clipped:
                 logger.debug(f"Target Age {target_age:.2f} normalized to {norm_age:.3f}, clipped to {norm_age_clipped:.3f}.")
                 norm_age = norm_age_clipped

            model_conditions = {}
            if 'Age' in wfm_model.condition_dims:
                 model_conditions['Age'] = torch.tensor([[norm_age]], dtype=torch.float32, device=device)

            for k_model, dim in wfm_model.condition_dims.items():
                if k_model != 'Age': # Default other conditions to 0.5 normalized
                    model_conditions[k_model] = torch.full((1, dim), 0.5, dtype=torch.float32, device=device)

            # --- Generate Single Sample ---
            try:
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    synthetic_images = wfm_model.sample(
                        num_samples=1,
                        model_output_size=model_input_size,
                        conditions_dict=model_conditions,
                        disable_cross_attn_inference=mode_flags["disable_cross_attn"],
                        disable_cond_film_inference=mode_flags["disable_cond_film"]
                    )

                # --- Post-process and Save ---
                if do_crop:
                    try:
                        (d_start, d_end), (h_start, h_end), (w_start, w_end) = crop_indices
                        synthetic_images_final = synthetic_images[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                        if synthetic_images_final.shape[-3:] != save_size_tuple:
                             logger.warning(f"Cropped shape {synthetic_images_final.shape[-3:]} != target {save_size_tuple}. Check cropping logic.")
                    except Exception as e:
                        logger.error(f"Error during cropping sample {i} ({mode_name}): {e}. Saving uncropped.", exc_info=True)
                        synthetic_images_final = synthetic_images
                else:
                    synthetic_images_final = synthetic_images

                img_to_save = (synthetic_images_final[0, 0].cpu().float() + 1.0) / 2.0
                img_to_save = torch.clamp(img_to_save, 0.0, 1.0)

                img_data = img_to_save.numpy().astype(np.float32)
                img_nifti = nib.Nifti1Image(img_data, affine=np.eye(4))

                sample_index_str = str(i).zfill(len(str(args.num_total_samples - 1)))
                # Use original filename_prefix, add mode suffix here
                filename = f"{args.filename_prefix}_{mode_flags['suffix']}_AGE_{target_age:.2f}_sample_{sample_index_str}.nii.gz"
                save_path = os.path.join(current_output_dir, filename)

                nib.save(img_nifti, save_path)

            except Exception as e:
                logger.error(f"Failed to generate or save sample {i} (Age: {target_age:.2f}, Mode: {mode_name}): {e}", exc_info=True)
                continue
        logger.info(f"Finished generating samples for mode {mode_name}.")


    logger.info("Linear Age generation (all modes) finished.")


if __name__ == "__main__":
    main()