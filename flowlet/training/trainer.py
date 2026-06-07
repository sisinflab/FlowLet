import os
import time
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ..utils.logging_utils import get_logger

from ..models import WaveletFlowMatching
from ..evaluation import visualize_flow_generation, visualize_multi_condition_samples

logger = get_logger(__name__)

def train_wavelet_flow_matching(wfm_model, train_loader, val_loader, num_epochs, lr,
                                use_wandb, checkpoint_dir, resume_from_path_arg, early_stop_patience,
                                model_output_size, condition_ranges, grad_clip_norm=1.0,
                                device=None, viz_every=1):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    wfm_model.to(device)
    optimizer = torch.optim.AdamW(wfm_model.flow_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    ckpt_path = os.path.join(checkpoint_dir, "fmw_best.pth")
    last_ckpt_path = os.path.join(checkpoint_dir, "fmw_last.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume training if requested
    resume_path_to_load = None # Variable to hold the path we actually load from
    if resume_from_path_arg:  # Check if the user specified a path
        if os.path.isfile(resume_from_path_arg):
            resume_path_to_load = resume_from_path_arg
            logger.info(f"Resume training explicitly requested from: {resume_path_to_load}")
        else:
            # If the specified path doesn't exist, log an error and do NOT resume.
            logger.error(f"Specified resume checkpoint not found: {resume_from_path_arg}. Starting training from scratch.")
            resume_path_to_load = None

    if resume_path_to_load:
        try:
            logger.info(f"Resuming FlowLet training from {resume_path_to_load}")
            ckpt = torch.load(resume_path_to_load, map_location=device)

            # --- Handle potential torch.compile state dict differences ---
            flow_net_state_dict = ckpt["flow_net_state_dict"]
            # If current model is NOT compiled, remove '_orig_mod.' prefix if present in ckpt
            if not isinstance(wfm_model.flow_net, torch.nn.modules.module.Module):
                 flow_net_state_dict = {k.replace('_orig_mod.', ''): v for k, v in flow_net_state_dict.items()}
            wfm_model.flow_net.load_state_dict(flow_net_state_dict)
            # --- End compile handling ---

            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
            else: logger.warning("Scaler state not found in checkpoint, reinitializing.")
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if scheduler.T_max != num_epochs:
                     logger.warning(f"Scheduler T_max ({scheduler.T_max}) differs from current num_epochs ({num_epochs}). Adjusting.")
                     scheduler.T_max = num_epochs
            except Exception as e: 
                logger.warning(f"Could not load scheduler state: {e}. Reinitializing scheduler.")
                scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
            
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            start_epoch = ckpt.get("epoch", -1) + 1
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            logger.info(f"Resumed from epoch {start_epoch}. Best Val Loss: {best_val_loss:.6f}. Patience: {epochs_without_improvement}")
        
        except Exception as e:
            logger.error(f"Failed to load FlowLet ckpt {resume_path_to_load}: {e}. Starting fresh.", exc_info=True)
            start_epoch = 0
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            optimizer = torch.optim.AdamW(wfm_model.flow_net.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
            scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    total_params = sum(p.numel() for p in wfm_model.flow_net.parameters() if p.requires_grad)
    logger.info(f"Starting Wavelet Flow Matching Training ({num_epochs} epochs from epoch {start_epoch}).")
    logger.info(f"Flowlet (U-Net) Trainable Params: {total_params:,}. LR={lr}.")
    if use_wandb: wandb.watch(wfm_model.flow_net, log_freq=500)

    global_step = start_epoch * len(train_loader)

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        wfm_model.flow_net.train()
        train_loss_epoch = 0
        num_samples_epoch = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [FlowLet Train]", leave=False)

        for step, batch in enumerate(train_pbar):
            global_step += 1
            if batch is None: logger.warning(f"Skipping step {step} due to data loading error."); continue
            batch_wavelet, conditions_dict = batch
            batch_wavelet = batch_wavelet.to(device)
            conditions_dict = {k: v.to(device) for k, v in conditions_dict.items()} if conditions_dict else {}
            batch_size = batch_wavelet.size(0)
            num_samples_epoch += batch_size
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                 loss = wfm_model.loss(batch_wavelet, conditions_dict, global_step, use_wandb)

            if torch.isinf(loss) or torch.isnan(loss):
                 logger.error(f"Epoch {epoch+1}, Step {step}: Encountered {loss.item()} loss. Skipping step.")
                 torch.cuda.empty_cache(); continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(wfm_model.flow_net.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += loss.item() * batch_size
            train_pbar.set_postfix({'loss': f"{loss.item():.6f}", 'grad': f"{grad_norm.item():.4f}", 'scale': scaler.get_scale()})
            if use_wandb and global_step % 100 == 0: wandb.log({"train/grad_norm": grad_norm.item()}, step=global_step)

        if num_samples_epoch == 0: logger.warning(f"FlowLet Epoch {epoch+1}: No training samples processed."); continue
        train_loss_epoch /= num_samples_epoch

        # Validation Phase
        wfm_model.flow_net.eval()
        val_loss_epoch = 0
        val_samples_epoch = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [FlowLet Valid]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                if batch is None: continue
                batch_wavelet, conditions_dict = batch
                batch_wavelet = batch_wavelet.to(device)
                conditions_dict = {k: v.to(device) for k, v in conditions_dict.items()} if conditions_dict else {}
                batch_size = batch_wavelet.size(0); val_samples_epoch += batch_size
                
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    loss = wfm_model.loss(batch_wavelet, conditions_dict, global_step, use_wandb=False)

                if torch.isinf(loss) or torch.isnan(loss):
                    logger.warning(f"Epoch {epoch+1}: Encountered {loss.item()} loss during validation. Skipping batch.")
                    val_samples_epoch -= batch_size
                    continue
                
                val_loss_epoch += loss.item() * batch_size; val_pbar.set_postfix({'val_loss': f"{loss.item():.6f}"})

        if val_samples_epoch == 0: 
            logger.warning(f"FlowLet Epoch {epoch+1}: No validation samples processed.")
            val_loss_epoch = float('inf')
        else: val_loss_epoch /= val_samples_epoch

        # Epoch Summary & Logging
        epoch_time = time.time() - epoch_start_time
        samples_per_sec = num_samples_epoch / epoch_time if epoch_time > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"FlowLet E{epoch+1}/{num_epochs} [{epoch_time:.1f}s, {samples_per_sec:.1f} spl/s] | Train Loss: {train_loss_epoch:.6f}, Val Loss: {val_loss_epoch:.6f}, LR: {current_lr:.6e}")

        # WandB Logging
        if use_wandb:
            log_dict = { "epoch": epoch+1, "flow_train_loss": train_loss_epoch, "flow_val_loss": val_loss_epoch, "flow_learning_rate": current_lr, "flow_epoch_time": epoch_time, "flow_samples_per_sec": samples_per_sec, "scaler_scale": scaler.get_scale() }
            if (epoch + 1) % viz_every == 0 or epoch == num_epochs - 1:
                try:
                    visualize_flow_generation(wfm_model, val_loader, model_output_size, use_wandb=True, epoch_num=epoch+1)
                    visualize_multi_condition_samples(wfm_model, num_samples=1, model_output_size=model_output_size, wandb_log=True, condition_ranges=condition_ranges, epoch_num=epoch+1)
                except Exception as e: 
                    logger.error(f"Error during visualization logging: {e}", exc_info=True)
            wandb.log(log_dict, step=global_step); torch.cuda.empty_cache()

        scheduler.step()

        # Checkpointing & Early Stopping
        # Save last checkpoint
        try:
            # Handle potential compile state
            model_state_dict = wfm_model.flow_net.state_dict()
            if hasattr(wfm_model.flow_net, '_orig_mod'): # If compiled
                 model_state_dict = wfm_model.flow_net._orig_mod.state_dict()

            torch.save({
                "epoch": epoch, "flow_net_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "best_val_loss": best_val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }, last_ckpt_path)
        except Exception as e: logger.error(f"Failed to save last checkpoint at epoch {epoch+1}: {e}", exc_info=True)

        # Save best checkpoint
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch; epochs_without_improvement = 0
            try:
                 model_state_dict = wfm_model.flow_net.state_dict()
                 if hasattr(wfm_model.flow_net, '_orig_mod'): model_state_dict = wfm_model.flow_net._orig_mod.state_dict()
                 torch.save({
                    "epoch": epoch, "flow_net_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "best_val_loss": best_val_loss,
                    "epochs_without_improvement": epochs_without_improvement,
                 }, ckpt_path)
                 logger.info(f"Saved best FlowLet model checkpoint (Epoch {epoch+1}, Val Loss: {best_val_loss:.6f})")
            except Exception as e: logger.error(f"Failed to save best checkpoint at epoch {epoch+1}: {e}", exc_info=True)
        else:
            epochs_without_improvement += 1
            logger.info(f"No FlowLet improvement for {epochs_without_improvement} epochs.")
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                logger.info(f"Early stopping FlowLet training at epoch {epoch+1} due to lack of improvement for {early_stop_patience} epochs.")
                break

    logger.info("FlowLet Training finished.")
    
    # Load best model
    if os.path.exists(ckpt_path):
        logger.info(f"Loading best FlowLet model from {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["flow_net_state_dict"]
            # Handle potential compile state mismatch when loading
            is_currently_compiled = hasattr(wfm_model.flow_net, '_orig_mod')
            is_saved_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())

            if is_currently_compiled and not is_saved_compiled:
                state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
            elif not is_currently_compiled and is_saved_compiled:
                 state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            wfm_model.flow_net.load_state_dict(state_dict)
            logger.info(f"Loaded best FlowLet model (Epoch {ckpt.get('epoch', -1)+1}, Val Loss: {ckpt.get('best_val_loss', '?'):.6f})")
        except Exception as e:
            logger.error(f"Failed to load best model state dict from {ckpt_path}: {e}. Using last state.", exc_info=True)
    else:
        logger.warning("Could not find best FlowLet checkpoint after training. Using last state.")

    return best_val_loss