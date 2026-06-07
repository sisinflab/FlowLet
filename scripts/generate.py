import argparse
import os
import torch
import json

# Import from our flowlet package
from flowlet.models import WaveletFlowMatching
from flowlet.generation import generate_conditioned_brains
from flowlet.utils import setup_logging, set_seed, get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Samples using a pre-trained FlowLet Model")

    # --- Model/Checkpoint Args ---
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.pth file, e.g., fmw_best.pth).")
    parser.add_argument("--config_path", type=str, default=None, help="(Optional) Path to the model configuration JSON file. If not provided, attempts to infer from checkpoint_path directory.")

    # --- Generation Args ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated NIfTI files.")
    parser.add_argument("--num_synthetic", type=int, default=10, help="Number of synthetic samples per condition.")
    parser.add_argument("--generation_conditions", nargs='*', required=True, help="Conditions for generation ('key=value' strings, e.g., 'age=60').")
    parser.add_argument("--condition_ranges_path", type=str, default=None, help="(Optional) Path to JSON file containing condition ranges (min/max). If not provided, normalization might be inaccurate or skipped.")
    parser.add_argument("--save_size", type=int, nargs=3, default=[91, 109, 91], metavar=('D', 'H', 'W'), help="Spatial size to crop generated images to before saving.")
    parser.add_argument("--model_input_size", type=int, nargs=3, default=None, metavar=('D', 'H', 'W'), help="Model's expected padded input size (D, H, W). Required if not found in config.")
    parser.add_argument("--num_flow_steps", type=int, default=None, help="Number of ODE sampling steps. Overrides the value stored in config.json if provided.")

    # --- System Args ---
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for generation noise.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (if generating many samples). Adjust based on GPU memory.")


    args = parser.parse_args()
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
    setup_logging(log_dir=args.output_dir, filename_prefix="flowlet_generate") # Log to output dir
    logger.info(f"Starting generation process.")
    logger.info(f"Generated samples will be saved to: {args.output_dir}")

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
        attention_res = tuple(map(int, model_config['unet_attention_res'].split(',')))
        channel_mult = tuple(map(int, model_config['unet_channel_mult'].split(',')))
        num_flow_steps = args.num_flow_steps if args.num_flow_steps is not None else model_config.get('num_flow_steps', 100)
    except KeyError as e:
        logger.error(f"Missing essential key in loaded/constructed config: {e}")
        return
    except Exception as e:
        logger.error(f"Error parsing configuration values: {e}")
        return

    # --- Load Model ---
    logger.info(f"Loading model checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        return

    try:
        ckpt = torch.load(args.checkpoint_path, map_location=device)

        # Reconstruct U-Net args from loaded config
        condition_dims_dict = {var: 1 for var in condition_vars} if condition_vars else {}
        unet_args = {
            "in_channels": 8, "model_channels": model_config.get('unet_model_channels', 128), "out_channels": 8,
            "num_res_blocks": model_config.get('unet_num_res_blocks', 2),
            "attention_resolutions": attention_res,
            "dropout": model_config.get('unet_dropout', 0.1),
            "channel_mult": channel_mult,
            "conv_resample": True, "dims": 3,
            "use_checkpoint": model_config.get('use_checkpointing', False),
            "num_heads": model_config.get('unet_num_heads', 8),
            "num_head_channels": model_config.get('unet_num_head_channels', -1),
            "use_scale_shift_norm": True,
            "resblock_updown": True,
            "condition_dims": condition_dims_dict,
            "condition_embedding_dim": model_config.get('condition_embedding_dim', 512),
            "use_xformers": model_config.get('use_xformers', True),
            "use_cross_attention": not model_config.get('unet_disable_cross_attn', False) and bool(condition_dims_dict),
            "norm_num_groups": model_config.get('unet_norm_num_groups', 32),
            "norm_eps": 1e-6,
        }
        logger.info(f"Reconstructed U-Net args for loading: {unet_args}")

        wfm_model = WaveletFlowMatching(u_net_args=unet_args, num_flow_steps=num_flow_steps).to(device)

        # Load state dict - handle potential torch.compile mismatch
        state_dict = ckpt.get("flow_net_state_dict", ckpt.get("model_state_dict", ckpt))
        if not state_dict:
            raise KeyError("Could not find a model state dictionary in the checkpoint.")

        # Determine if the current model instance will be compiled (it won't be by default here)
        is_currently_compiled = hasattr(wfm_model.flow_net, '_orig_mod')
        is_saved_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())

        if not is_currently_compiled and is_saved_compiled:
            logger.info("Saved checkpoint seems to be from a compiled model. Removing '_orig_mod.' prefix.")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        elif is_currently_compiled and not is_saved_compiled:
             # This case shouldn't happen here unless you compile the model before loading
             logger.warning("Loading non-compiled state_dict into a compiled model (unexpected). Adding '_orig_mod.' prefix.")
             state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}


        wfm_model.flow_net.load_state_dict(state_dict)
        wfm_model.eval()
        logger.info(f"Model loaded successfully from epoch {ckpt.get('epoch', -1)+1}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    # --- Prepare Conditions ---
    parsed_conditions_list = []
    for cond_set_str in args.generation_conditions:
         cond_dict = {}
         try:
             items = cond_set_str.split() if ' ' in cond_set_str else [cond_set_str]
             for item in items:
                 if '=' not in item: raise ValueError(f"Condition item '{item}' missing '=' separator.")
                 key, value_str = item.split('=', 1); key = key.strip(); value_str = value_str.strip()
                 if not key: raise ValueError("Condition key cannot be empty.")
                 if key not in condition_vars:
                      logger.warning(f"Provided condition key '{key}' is not in the model's configured conditions {condition_vars}. Skipping this key for generation.")
                      continue
                 cond_dict[key] = float(value_str)
         except Exception as e: logger.error(f"Invalid format in condition string: '{cond_set_str}'. Skipping. Error: {e}"); continue
         if cond_dict: parsed_conditions_list.append(cond_dict)

    if not parsed_conditions_list:
        logger.error("No valid conditions provided or parsed. Cannot generate.")
        return

    # Load condition ranges for normalization
    condition_ranges = None
    if args.condition_ranges_path and os.path.exists(args.condition_ranges_path):
        try:
            with open(args.condition_ranges_path, 'r') as f:
                condition_ranges = json.load(f)
            logger.info(f"Loaded condition ranges from: {args.condition_ranges_path}")
        except Exception as e:
            logger.warning(f"Failed to load condition ranges file {args.condition_ranges_path}: {e}. Normalization might be inaccurate.", exc_info=True)
    else:
        # Try finding ranges in checkpoint dir
        ranges_path_alt = os.path.join(os.path.dirname(args.checkpoint_path), "condition_ranges.json")
        if os.path.exists(ranges_path_alt):
            try:
                with open(ranges_path_alt, 'r') as f:
                    condition_ranges = json.load(f)
                logger.info(f"Loaded condition ranges from checkpoint directory: {ranges_path_alt}")
            except Exception as e:
                 logger.warning(f"Failed to load condition_ranges.json from checkpoint dir: {e}. Normalization might be inaccurate.", exc_info=True)
        else:
            logger.warning("Condition ranges file not provided or found. Condition normalization during generation might be inaccurate or skipped if raw values are not in [0,1].")


    # --- Generate ---
    logger.info(f"Generating {args.num_synthetic} samples for {len(parsed_conditions_list)} condition sets...")
    try:
        generate_conditioned_brains(
            wfm_model=wfm_model,
            conditions_list=parsed_conditions_list,
            num_samples_per_condition=args.num_synthetic,
            output_dir=args.output_dir,
            save_size=tuple(args.save_size),
            model_output_size=model_input_size,
            condition_ranges=condition_ranges
        )
        logger.info(f"Finished generating samples in {args.output_dir}")
    except Exception as e:
        logger.error(f"An error occurred during generation: {e}", exc_info=True)

    logger.info("Generation script finished.")


if __name__ == "__main__":
    main()