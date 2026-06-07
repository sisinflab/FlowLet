import argparse
import os
import torch
import wandb
import json
from torch.utils.data import DataLoader, random_split

# Import both dataset types
from flowlet.data import (
    create_brain_dataset_and_split,
    collate_fn,
    train_transform,
    val_transform,
    TransformedSubset
)
from flowlet.data.dataset_csv import BrainMRIDatasetCSV

from flowlet.models import WaveletFlowMatching
from flowlet.training import train_wavelet_flow_matching
from flowlet.evaluation import visualize_flow_generation, visualize_multi_condition_samples
from flowlet.generation import generate_conditioned_brains
from flowlet.utils import setup_logging, set_seed, get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Wavelet Flow Matching (FlowLet) Model")

    # --- Data Args ---
    parser.add_argument("--data_folder", type=str, required=False,
                        help="Path to folder containing .nii.gz files (used *only* if --metadata_csv is NOT provided).")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Path to the metadata CSV file. If provided, this overrides --data_folder and uses the CSV for file paths and conditions.")
    parser.add_argument("--condition_vars", nargs="+", default=["Age"],
                        help="List of conditions to use. If using --metadata_csv, these must be column names in the CSV. If not, they are parsed from filenames.")
    parser.add_argument("--require_conditions", action=argparse.BooleanOptionalAction, default=True,
                        help="If using filename parsing (no --metadata_csv), only use images where ALL specified condition_vars are found.")
    parser.add_argument("--model_input_size", type=int, nargs=3, default=[112, 112, 112], metavar=('D', 'H', 'W'), help="Spatial size images are padded to before DWT.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation (0.0 to 1.0).")
    # --- CSV Specific Data Args (Optional Filtering) ---
    parser.add_argument("--csv_filter_col", type=str, default=None,
                        help="[CSV Mode Only] Column name in the CSV to filter by (e.g., 'Condition').")
    parser.add_argument("--csv_filter_value", type=str, default=None,
                        help="[CSV Mode Only] Value to keep in the --csv_filter_col.")


    # --- Flow Matching Training Args ---
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for Flow Matching training.")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate for Flow Matching AdamW optimizer.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for Flow Matching training.")
    parser.add_argument("--early_stop_patience", type=int, default=50, help="Epochs with no val loss improvement before stopping. <= 0 disables.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Maximum norm for gradient clipping.")
    parser.add_argument("--num_flow_steps", type=int, default=100, help="Number of integration steps for sampling.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint (.pth file) to resume training from.")
    parser.add_argument("--flow_type", type=str, default="rectified",
                        choices=["rectified", "cfm", "trigonometric", "vp_diffusion"],
                        help="The type of flow matching to use for the training loss.")
    
    ### Additional VP Diffusion parameters ###
    parser.add_argument("--vp_beta_min", type=float, default=0.1, help="[VP-Diffusion Only] Minimum beta value for the schedule.")
    parser.add_argument("--vp_beta_max", type=float, default=20.0, help="[VP-Diffusion Only] Maximum beta value for the schedule.")

    # --- U-Net Architecture Args ---
    parser.add_argument("--unet_model_channels", type=int, default=128, help="Base number of channels in the U-Net.")
    parser.add_argument("--unet_num_res_blocks", type=int, default=2, help="Number of residual blocks per U-Net level.")
    parser.add_argument("--unet_channel_mult", type=str, default="1,2,3,4", help="Channel multipliers (comma-sep string, e.g., '1,2,3,4').")
    parser.add_argument("--unet_attention_res", type=str, default="16,8", help="Resolutions (relative to initial feature map size) for attention (comma-sep string, e.g., '16,8').")
    parser.add_argument("--unet_dropout", type=float, default=0.1, help="Dropout rate in U-Net ResBlocks/Attention.")
    parser.add_argument("--condition_embedding_dim", type=int, default=512, help="Dimension of the projected condition embeddings.")
    parser.add_argument("--unet_num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--unet_num_head_channels", type=int, default=-1, help="Number of channels per head (-1 means calculate based on num_heads).")
    parser.add_argument("--unet_norm_num_groups", type=int, default=32, help="Number of groups for GroupNorm.")
    parser.add_argument("--use_checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Enable gradient checkpointing in U-Net.")
    parser.add_argument("--use_xformers", action=argparse.BooleanOptionalAction, default=True, help="Enable xformers memory-efficient attention if available.")
    parser.add_argument("--unet_disable_cross_attn", action="store_true", help="Disable cross-attention in SpatialTransformer (model becomes unconditional to context).")
    parser.add_argument("--lll_loss_weight", type=float, default=1, help="Weight for the LLL (approximation) subband loss. Default: 1")
    parser.add_argument("--detail_loss_weight", type=float, default=1, help="Weight for the combined detail subbands (LH, HL, HH) loss. Default: 1")

    # --- Generation Args (Optional post-training generation) ---
    parser.add_argument("--generate_after_train", action=argparse.BooleanOptionalAction, default=False, help="Generate samples after training finishes.")
    parser.add_argument("--num_synthetic", type=int, default=10, help="Number of synthetic samples per condition if generating.")
    parser.add_argument("--generation_conditions", nargs='*', default=['age=45', 'age=75'], help="Conditions for generation ('key=value' strings).")
    parser.add_argument("--save_size", type=int, nargs=3, default=[91, 109, 91], metavar=('D', 'H', 'W'), help="Spatial size to crop generated images to before saving.")
    parser.add_argument("--generation_output_dir", type=str, default="generated_samples", help="Subdirectory within checkpoint_dir for generated samples.")


    # --- System Args ---
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_flowlet", help="Directory for saving checkpoints and logs.")
    parser.add_argument("--run_name", type=str, default="flowlet_run", help="A name for this training run (used for logging/checkpoints).")
    parser.add_argument("--viz_every", type=int, default=1, help="Log validation/sample visualizations every N epochs (W&B only). Default: 1 (every epoch).")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="FlowLet_training", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team).")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile for the U-Net (experimental).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")


    args = parser.parse_args()

    # Validate argument dependencies
    if args.metadata_csv is None and args.data_folder is None:
        parser.error("Either --metadata_csv or --data_folder must be provided.")
    if args.metadata_csv and args.data_folder:
        logger.warning("Both --metadata_csv and --data_folder provided. --metadata_csv will be used.")
    if args.csv_filter_col and args.csv_filter_value is None:
        parser.error("--csv_filter_value must be provided if --csv_filter_col is set.")
    if args.lll_loss_weight < 0 or args.detail_loss_weight < 0:
        logger.warning("Loss weights should be non-negative. The model will use their absolute values.")
    if args.lll_loss_weight == 0 and args.detail_loss_weight == 0:
        logger.warning("Both lll_loss_weight and detail_loss_weight are 0. This will result in zero loss and no training. The model will internally default to 1 each if both are zero during loss calculation to prevent collapse.")
    return args

def main():
    args = parse_args()

    # --- Setup ---
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    setup_logging(log_dir=run_checkpoint_dir, filename_prefix=args.run_name)
    logger.info(f"Starting training run: {args.run_name}")
    logger.info(f"Checkpoints and logs will be saved to: {run_checkpoint_dir}")

    # Save configuration as JSON
    config_save_path = os.path.join(run_checkpoint_dir, "config.json")
    try:
        config_to_save = vars(args)
        # Convert tuples to lists for JSON compatibility
        for key, value in config_to_save.items():
            if isinstance(value, tuple):
                config_to_save[key] = list(value)

        with open(config_save_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, sort_keys=True)
        logger.info(f"Configuration saved to {config_save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration to {config_save_path}: {e}", exc_info=True)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    if args.wandb:
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args)
            )
            logger.info("Weights & Biases initialized.")
        except ImportError:
            logger.warning("wandb not installed, disabling logging.")
            args.wandb = False
        except Exception as e:
            logger.error(f"Wandb initialization failed: {e}. Disabling wandb.", exc_info=True)
            args.wandb = False

    # --- Dataset & Dataloaders ---
    logger.info("Creating and splitting dataset...")
    train_dataset = None
    val_dataset = None
    condition_ranges = None

    try:
        # Validate input size vs U-Net structure
        channel_mult_list = tuple(map(int, args.unet_channel_mult.split(',')))
        num_downsamples = len(channel_mult_list) - 1
        required_divisor = 2**num_downsamples
        unet_input_size = tuple(s // 2 for s in args.model_input_size) # DWT halves spatial dims
        if any(s % required_divisor != 0 for s in unet_input_size):
             logger.warning(f"U-Net input size {unet_input_size} (from model_input_size {args.model_input_size}) "
                            f"is not divisible by {required_divisor} (required by channel_mult {args.unet_channel_mult}). "
                            f"Ensure padding handles this correctly.")
        else:
            logger.info(f"U-Net input size {unet_input_size} compatible with downsampling.")

        # --- Choose Dataset Loading Method ---
        if args.metadata_csv:
            logger.info(f"Using CSV metadata from: {args.metadata_csv}")
            if not os.path.exists(args.metadata_csv):
                raise FileNotFoundError(f"Metadata CSV file not found: {args.metadata_csv}")

            full_dataset = BrainMRIDatasetCSV(
                metadata_path=args.metadata_csv,
                transform=None, 
                model_input_size=tuple(args.model_input_size),
                filepath_col="FilePath",       
                subject_id_col="SubjectID",    
                condition_cols=args.condition_vars, 
                filter_col=args.csv_filter_col,        
                filter_value=args.csv_filter_value     
            )
            condition_ranges = full_dataset.condition_ranges

            if len(full_dataset) == 0:
                raise RuntimeError("Dataset is empty after loading from CSV (and filtering).")

            val_split = max(0.0, min(1.0, args.val_split))
            if val_split == 0.0 or val_split == 1.0:
                 logger.warning(f"Validation split is {val_split}, dataset will not be split.")
                 if val_split == 0.0:
                     train_dataset = TransformedSubset(full_dataset, train_transform)
                     val_dataset = torch.utils.data.TensorDataset(torch.empty(0))
                 else: 
                     val_dataset = TransformedSubset(full_dataset, val_transform)
                     train_dataset = torch.utils.data.TensorDataset(torch.empty(0))
            else:
                train_size = int((1.0 - val_split) * len(full_dataset))
                val_size = len(full_dataset) - train_size
                logger.info(f"Splitting CSV dataset: {train_size} train, {val_size} validation samples.")
                if train_size == 0 or val_size == 0:
                    raise ValueError("Dataset split resulted in zero samples for train or validation.")

                generator = torch.Generator().manual_seed(args.seed)
                train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

                train_dataset = TransformedSubset(train_subset, train_transform)
                val_dataset = TransformedSubset(val_subset, val_transform)

        else:
            # --- Use Filename Parsing Method ---
            logger.info(f"Using filename parsing from data folder: {args.data_folder}")
            if args.data_folder is None or not os.path.isdir(args.data_folder):
                 raise FileNotFoundError(f"Data folder not found or not specified: {args.data_folder}. Required when not using --metadata_csv.")

            train_dataset, val_dataset, condition_ranges = create_brain_dataset_and_split(
                data_folder=args.data_folder,
                metadata_path=None, 
                transform_train=train_transform,
                transform_val=val_transform,
                model_input_size=tuple(args.model_input_size),
                filter_cognitive_status=None, 
                condition_vars=args.condition_vars,
                require_conditions=args.require_conditions,
                val_split=args.val_split,
                seed=args.seed
            )

        if train_dataset is None or val_dataset is None:
             raise RuntimeError("Dataset creation failed, train or validation dataset is None.")

        logger.info(f"Dataset created. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        logger.info(f"Condition ranges found: {condition_ranges}")

        ranges_save_path = os.path.join(run_checkpoint_dir, "condition_ranges.json")
        try:
            with open(ranges_save_path, 'w') as f:
                json.dump(condition_ranges, f, indent=4)
            logger.info(f"Condition ranges saved to {ranges_save_path}")
        except Exception as e:
            logger.error(f"Failed to save condition ranges to {ranges_save_path}: {e}", exc_info=True)


        if len(train_dataset) == 0:
            logger.warning("Training dataset is empty after processing.") 
            if args.val_split != 1.0:
                 raise RuntimeError("Training dataset is empty.")
        if len(val_dataset) == 0:
             logger.warning("Validation dataset is empty after processing.")
             if args.val_split != 0.0:
                  raise RuntimeError("Validation dataset is empty.")


    except Exception as e:
        logger.error(f"Failed during dataset creation or splitting: {e}", exc_info=True)
        if args.wandb: wandb.finish(exit_code=1)
        return

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn, persistent_workers=args.num_workers > 0 and not isinstance(train_dataset, torch.utils.data.TensorDataset)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=args.num_workers > 0 and not isinstance(val_dataset, torch.utils.data.TensorDataset)
    )
    logger.info("DataLoaders created.")

    # --- Flow Matching Model ---
    condition_dims_dict = {var: 1 for var in args.condition_vars} if args.condition_vars else {}
    try:
        attention_res = tuple(map(int, args.unet_attention_res.split(',')))
        # channel_mult_list is already defined above
    except ValueError as e:
        logger.error(f"Invalid format for U-Net channel mult or attention res: {e}")
        if args.wandb: wandb.finish(exit_code=1)
        return

    unet_args = {
        "in_channels": 8, "model_channels": args.unet_model_channels, "out_channels": 8,
        "num_res_blocks": args.unet_num_res_blocks,
        "attention_resolutions": attention_res,
        "dropout": args.unet_dropout,
        "channel_mult": channel_mult_list,
        "conv_resample": True, "dims": 3,
        "use_checkpoint": args.use_checkpointing,
        "num_heads": args.unet_num_heads,
        "num_head_channels": args.unet_num_head_channels,
        "use_scale_shift_norm": True, 
        "resblock_updown": True,
        "condition_dims": condition_dims_dict,
        "condition_embedding_dim": args.condition_embedding_dim,
        "use_xformers": args.use_xformers,
        "use_cross_attention": not args.unet_disable_cross_attn and bool(condition_dims_dict),
        "norm_num_groups": args.unet_norm_num_groups,
        "norm_eps": 1e-6,
    }
    wfm_model = WaveletFlowMatching(
        u_net_args=unet_args,
        num_flow_steps=args.num_flow_steps,
        lll_loss_weight=args.lll_loss_weight,
        detail_loss_weight=args.detail_loss_weight,
        flow_type=args.flow_type,
        vp_beta_min=args.vp_beta_min,
        vp_beta_max=args.vp_beta_max
    ).to(device)
    logger.info("FlowLet model initialized.")
    logger.info(f"U-Net args: {unet_args}")


    # --- Compile Flow Model (Optional) ---
    if args.compile:
        try:
            logger.info("Attempting to compile FlowLet U-Net model...")
            wfm_model.flow_net = torch.compile(wfm_model.flow_net, mode="reduce-overhead")
            logger.info("FlowLet U-Net compiled successfully.")
        except Exception as e:
            logger.warning(f"Torch compile failed for Flowlet U-Net: {e}. Continuing without compilation.", exc_info=True)

    # --- Flow Matching Training ---
    logger.info("Starting training...")
    try:
        train_wavelet_flow_matching(
            wfm_model=wfm_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            use_wandb=args.wandb,
            checkpoint_dir=run_checkpoint_dir,
            resume_from_path_arg=args.resume_from_checkpoint,
            early_stop_patience=args.early_stop_patience,
            model_output_size=tuple(args.model_input_size),
            condition_ranges=condition_ranges,
            grad_clip_norm=args.grad_clip_norm,
            device=device,
            viz_every=args.viz_every
        )
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        if args.wandb: wandb.finish(exit_code=1)
        return

    logger.info("Training finished.")
    torch.cuda.empty_cache()

    # --- Final Visualization ---
    logger.info("--- Generating Final Visualizations ---")
    wfm_model.eval()
    try:
        if len(val_dataset) > 0:
            visualize_flow_generation(wfm_model, val_loader, tuple(args.model_input_size), use_wandb=args.wandb, epoch_num=None)
            visualize_multi_condition_samples(wfm_model, num_samples=1, model_output_size=tuple(args.model_input_size), wandb_log=args.wandb, condition_ranges=condition_ranges, epoch_num=None)
        else:
            logger.info("Skipping final visualization as validation dataset is empty.")
    except Exception as e:
        logger.error(f"Error during final visualization: {e}", exc_info=True)


    # --- Optional Generation After Training ---
    if args.generate_after_train and args.generation_conditions:
        logger.info("--- Generating Samples Post-Training ---")
        parsed_conditions_list = []
        for cond_set_str in args.generation_conditions:
             cond_dict = {}
             try:
                 items = cond_set_str.split() if ' ' in cond_set_str else [cond_set_str]
                 for item in items:
                     if '=' not in item: raise ValueError(f"Condition item '{item}' missing '=' separator.")
                     key, value_str = item.split('=', 1); key = key.strip(); value_str = value_str.strip()
                     if not key: raise ValueError("Condition key cannot be empty.")
                     if key in args.condition_vars:
                          cond_dict[key] = float(value_str)
                     else:
                          logger.warning(f"Generation condition '{key}' provided but not in trained condition_vars ({args.condition_vars}). Ignoring.")

             except Exception as e: logger.error(f"Invalid format in condition string: '{cond_set_str}'. Skipping. Error: {e}"); continue
             if cond_dict: parsed_conditions_list.append(cond_dict)

        if parsed_conditions_list:
            gen_output_path = os.path.join(run_checkpoint_dir, args.generation_output_dir)
            try:
                generate_conditioned_brains(
                    wfm_model=wfm_model,
                    conditions_list=parsed_conditions_list,
                    num_samples_per_condition=args.num_synthetic,
                    output_dir=gen_output_path,
                    save_size=tuple(args.save_size),
                    model_output_size=tuple(args.model_input_size),
                    condition_ranges=condition_ranges
                )
                logger.info(f"Generated samples saved to: {gen_output_path}")
            except Exception as e:
                logger.error(f"Error during post-training generation: {e}", exc_info=True)
        else:
            logger.warning("No valid conditions parsed from --generation_conditions or conditions provided were not used during training, skipping post-training generation.")

    if args.wandb:
        wandb.finish()
    logger.info("Script finished successfully!")


if __name__ == "__main__":
    main()