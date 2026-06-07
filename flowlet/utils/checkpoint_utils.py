import torch
import argparse
import os
import json
from .logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Configuration keys potentially useful to keep for inference setup
# Adjust this list based on what your inference scripts actually need
CONFIG_KEYS_TO_KEEP = [
    'condition_vars',
    'model_input_size',
    'num_flow_steps',
    'unet_model_channels',
    'unet_num_res_blocks',
    'unet_channel_mult',
    'unet_attention_res',
    'unet_dropout',
    'condition_embedding_dim',
    'unet_num_heads',
    'unet_num_head_channels',
    'unet_norm_num_groups',
    'use_checkpointing',
    'use_xformers',
    'unet_disable_cross_attn',
]

# Function to slim a checkpoint by removing unnecessary parts. To be used for final inference only
def slim_checkpoint(input_path: str, output_path: str, config_path: str | None = None):
    """
    Loads a PyTorch checkpoint, extracts only the model state dictionary
    and optionally key configuration parameters, and saves it to a new file.

    Args:
        input_path: Path to the original (potentially large) checkpoint file.
        output_path: Path where the slimmed checkpoint will be saved.
        config_path: Optional path to the training config.json file. If provided,
                     relevant configuration keys will be added to the slimmed checkpoint.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input checkpoint file not found: {input_path}")
        return

    logger.info(f"Loading original checkpoint from: {input_path}")
    try:
        # Load to CPU to avoid unnecessary GPU memory usage
        original_ckpt = torch.load(input_path, map_location='cpu')
        logger.info(f"Original checkpoint keys: {list(original_ckpt.keys())}")

    except Exception as e:
        logger.error(f"Failed to load checkpoint file: {e}", exc_info=True)
        return

    # --- Extract State Dict ---
    state_dict = None
    if "flow_net_state_dict" in original_ckpt:
        state_dict = original_ckpt["flow_net_state_dict"]
        logger.info("Found 'flow_net_state_dict'.")
    elif "model_state_dict" in original_ckpt: # Fallback for different naming
        state_dict = original_ckpt["model_state_dict"]
        logger.warning("Found 'model_state_dict' instead of 'flow_net_state_dict'. Using it.")
    else:
        logger.error("Could not find 'flow_net_state_dict' or 'model_state_dict' in the checkpoint.")
        return

    # --- Clean State Dict Keys ---
    cleaned_state_dict = {}
    cleaned_count = 0
    prefix_to_remove = None

    # Detect prefix (only one type expected usually)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        prefix_to_remove = '_orig_mod.'
        logger.info("Detected '_orig_mod.' prefix (torch.compile).")
    elif any(k.startswith('module.') for k in state_dict.keys()):
         prefix_to_remove = 'module.'
         logger.info("Detected 'module.' prefix (DDP/DP).")

    for k, v in state_dict.items():
        new_k = k
        if prefix_to_remove and k.startswith(prefix_to_remove):
            new_k = k[len(prefix_to_remove):]
            cleaned_count += 1
        cleaned_state_dict[new_k] = v

    if cleaned_count > 0:
        logger.info(f"Removed prefix '{prefix_to_remove}' from {cleaned_count} keys.")

    # --- Prepare Slim Checkpoint ---
    slim_ckpt = {
        # Use a consistent key for the slimmed state dict
        "model_state_dict": cleaned_state_dict
    }

    # --- Add Configuration (Optional) ---
    if config_path:
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)

                config_to_save = {}
                for key in CONFIG_KEYS_TO_KEEP:
                    if key in config_data:
                        config_to_save[key] = config_data[key]
                    else:
                         logger.warning(f"Configuration key '{key}' not found in {config_path}.")

                if config_to_save:
                    slim_ckpt["config"] = config_to_save
                    logger.info(f"Added configuration keys to slim checkpoint: {list(config_to_save.keys())}")

            except Exception as e:
                logger.error(f"Failed to load or process config file {config_path}: {e}", exc_info=True)
        else:
            logger.warning(f"Specified config path not found: {config_path}. Configuration not added.")

    # --- Save Slim Checkpoint ---
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure directory exists if specified
            os.makedirs(output_dir, exist_ok=True)

        torch.save(slim_ckpt, output_path)
        logger.info(f"Slim checkpoint saved successfully to: {output_path}")

        # Compare file sizes
        original_size = os.path.getsize(input_path)
        slim_size = os.path.getsize(output_path)
        logger.info(f"Original size: {original_size / (1024*1024):.2f} MB")
        logger.info(f"Slim size:     {slim_size / (1024*1024):.2f} MB")
        reduction = (original_size - slim_size) / original_size * 100 if original_size > 0 else 0
        logger.info(f"Size reduction: {reduction:.1f}%")

    except Exception as e:
        logger.error(f"Failed to save slim checkpoint to {output_path}: {e}", exc_info=True)


def main_cli():
    """Command-line interface for the slim_checkpoint function."""
    parser = argparse.ArgumentParser(
        description="Slim a FlowLet PyTorch checkpoint by removing optimizer states, etc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the original (large) checkpoint file (.pth)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the slimmed checkpoint file will be saved (.pth)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="(Optional) Path to the training config.json file associated with the input checkpoint. "
             "If provided, key configuration parameters will be stored in the slim checkpoint."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=".",
        help="Directory to save the log file for this script run."
    )

    args = parser.parse_args()

    # Setup logging specifically for this utility script run
    setup_logging(log_dir=args.log_dir, filename_prefix="checkpoint_slimmer")

    slim_checkpoint(args.input_path, args.output_path, args.config_path)
    
    logger.info("Checkpoint slimming process completed.")

if __name__ == "__main__":
    # This allows running the script directly from the command line
    # Example: python -m flowlet.utils.checkpoint_utils --input_path path/to/fmw_best.pth --output_path path/to/fmw_best_slim.pth --config_path path/to/config.json
    main_cli()