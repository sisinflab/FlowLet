import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.amp
import math
from ..utils.logging_utils import get_logger

from .unet import ConditionalUNet
from ..wavelets import dwt_3d, idwt_3d

logger = get_logger(__name__)


class WaveletFlowMatching(nn.Module):
    def __init__(self, u_net_args, num_flow_steps=100, lll_loss_weight: float = 0.5, detail_loss_weight: float = 0.5, flow_type: str = "rectified",
                 vp_beta_min: float = 0.1, vp_beta_max: float = 20.0):
        super().__init__()
        self.num_flow_steps = num_flow_steps
        self.condition_dims = u_net_args.get('condition_dims', {})
        self.flow_net = ConditionalUNet(**u_net_args)
        self.dwt = dwt_3d
        self.idwt = idwt_3d
        self.lll_loss_weight = lll_loss_weight
        self.detail_loss_weight = detail_loss_weight
        
        ### VP Diffusion parameters ###
        self.vp_beta_min = vp_beta_min
        self.vp_beta_max = vp_beta_max
        
        self.flow_type = flow_type.lower()
        if self.flow_type not in ["rectified", "cfm", "trigonometric", "vp_diffusion"]:
            raise ValueError(f"Unknown flow_type: {self.flow_type}. Must be one of 'rectified', 'cfm', 'trigonometric', 'vp_diffusion'.")
        logger.info(f"WaveletFlowMatching initialized with flow_type: '{self.flow_type}'")

        if not (0 <= self.lll_loss_weight <= 1 and 0 <= self.detail_loss_weight <= 1):
            logger.warning(f"Loss weights (LLL: {self.lll_loss_weight}, Detail: {self.detail_loss_weight}) are outside [0,1]. This is allowed, but ensure they reflect desired relative importance.")
        if self.lll_loss_weight == 0 and self.detail_loss_weight == 0:
             logger.warning("Both LLL and Detail loss weights are zero! This will result in zero loss and no training. Consider setting at least one to a positive value.")
        else:
            logger.info(f"WaveletFlowMatching initialized with LLL loss weight: {self.lll_loss_weight}, Detail loss weight: {self.detail_loss_weight}")


    # VP Diffusion helper functions
    def _vp_T(self, s: torch.Tensor) -> torch.Tensor:
        return self.vp_beta_min * s + 0.5 * (s ** 2) * (self.vp_beta_max - self.vp_beta_min)

    def _vp_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.vp_beta_min + t * (self.vp_beta_max - self.vp_beta_min)

    def _vp_alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self._vp_T(t))

    def _vp_mu_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return self._vp_alpha(1. - t) * x_1

    def _vp_sigma_t(self, t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1. - self._vp_alpha(1. - t) ** 2)

    def _vp_u_t(self, t: torch.Tensor, x: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        num = torch.exp(-self._vp_T(1. - t)) * x - torch.exp(-0.5 * self._vp_T(1. - t)) * x_1
        denum = 1. - torch.exp(-self._vp_T(1. - t))
        # Add epsilon for numerical stability
        return -0.5 * self._vp_beta(1. - t) * (num / (denum + 1e-8))


    def _calculate_weighted_loss(self, v_pred, v_target, global_step, use_wandb):
        """Helper to compute the weighted LLL/detail loss. All Paper results use 0.5 for both weights, resulting in the simple MSE loss."""
        if not torch.isfinite(v_pred).all():
            logger.error(f"Step {global_step}: NaN/Inf detected in v_pred (model output)!")
            return torch.tensor(float('inf'), device=v_pred.device)

        # v_target and v_pred have shape: (B, 8, D_coeff, H_coeff, W_coeff)
        lll_v_target = v_target[:, 0:1, ...] # (B, 1, D', H', W')
        lll_v_pred = v_pred[:, 0:1, ...]
        detail_v_target = v_target[:, 1:, ...] # (B, 7, D', H', W')
        detail_v_pred = v_pred[:, 1:, ...]

        loss_lll = 0.0
        if self.lll_loss_weight > 0:
            loss_lll = F.mse_loss(lll_v_pred.float(), lll_v_target.float())
        
        loss_detail = 0.0
        if self.detail_loss_weight > 0:
            loss_detail = F.mse_loss(detail_v_pred.float(), detail_v_target.float())
        
        weighted_loss = (self.lll_loss_weight * loss_lll) + (self.detail_loss_weight * loss_detail)
        
        if not torch.isfinite(weighted_loss):
             unweighted_lll = F.mse_loss(lll_v_pred.float(), lll_v_target.float()).item() if self.lll_loss_weight > 0 else "skipped"
             unweighted_detail = F.mse_loss(detail_v_pred.float(), detail_v_target.float()).item() if self.detail_loss_weight > 0 else "skipped"
             logger.error(f"Step {global_step}: NaN/Inf detected in final loss value ({weighted_loss.item()})! "
                          f"Weighted LLL: {self.lll_loss_weight * (loss_lll.item() if isinstance(loss_lll, torch.Tensor) else loss_lll):.4e}, "
                          f"Weighted Detail: {self.detail_loss_weight * (loss_detail.item() if isinstance(loss_detail, torch.Tensor) else loss_detail):.4e}. "
                          f"Unweighted LLL: {unweighted_lll}, Unweighted Detail: {unweighted_detail}")
             if use_wandb:
                 wandb.log({"debug/loss_inf_nan": 1.0}, step=global_step)
             return torch.tensor(float('inf'), device=v_pred.device)
        
        if global_step % 100 == 0 and use_wandb:
            log_data_loss = {
                "debug/loss_lll_unweighted": loss_lll.item() if isinstance(loss_lll, torch.Tensor) and self.lll_loss_weight > 0 else 0.0,
                "debug/loss_detail_unweighted": loss_detail.item() if isinstance(loss_detail, torch.Tensor) and self.detail_loss_weight > 0 else 0.0,
            }
            wandb.log(log_data_loss, step=global_step)
            
        return weighted_loss

######### COMPUTE LOSS FUNCTIONS #########

    def compute_vp_diffusion_loss(self, x1_wavelet, conditions_dict, global_step=0, use_wandb=False):
        batch_size, device = x1_wavelet.shape[0], x1_wavelet.device
        
        # Time sampling t ~ Unif([0, 1])
        # Note: Added a small epsilon to prevent t=1, which can cause division by zero.
        t_scalar = (torch.rand(1, device=device) + torch.arange(batch_size, device=device)) / batch_size
        t_scalar = torch.fmod(t_scalar, 1.0 - 1e-5)

        # The U-Net expects a 1D tensor of shape [B] for time embedding.
        t_for_unet = t_scalar

        # The VP formulas expect t to be broadcastable to the wavelet tensor shape.
        t_broadcast = t_scalar.view(batch_size, *([1] * (x1_wavelet.dim() - 1)))
        
        # Sample x_t from the conditional path p_t(x|x_1), where t=0 is data and t=1 is noise.
        x_t = self._vp_mu_t(t_broadcast, x1_wavelet) + self._vp_sigma_t(t_broadcast, x1_wavelet) * torch.randn_like(x1_wavelet)
        
        # Calculate the target velocity field u_t(x_t | x_1) using the same t.
        v_target = self._vp_u_t(t_broadcast, x_t, x1_wavelet)
        
        # Get the predicted velocity from the U-Net.
        # The U-Net receives the exact same time `t_for_unet` that was used to generate the path.
        v_pred = self.flow_net(x_t, t_for_unet, conditions_dict)
        
        # loss calculation
        return self._calculate_weighted_loss(v_pred, v_target, global_step, use_wandb)

    def compute_rectified_flow_loss(self, x1_wavelet, conditions_dict, global_step=0, use_wandb=False):
        batch_size, device = x1_wavelet.shape[0], x1_wavelet.device
        t = torch.rand(batch_size, device=device)
        x0_wavelet = torch.randn_like(x1_wavelet)
        t_broadcast = t.view(batch_size, *([1] * (x1_wavelet.dim() - 1)))
        
        xt = (1 - t_broadcast) * x0_wavelet + t_broadcast * x1_wavelet
        v_target = x1_wavelet - x0_wavelet
        
        v_pred = self.flow_net(xt, t, conditions_dict)
        return self._calculate_weighted_loss(v_pred, v_target, global_step, use_wandb)

    def compute_cfm_loss(self, x1_wavelet, conditions_dict, global_step=0, use_wandb=False):
        batch_size, device = x1_wavelet.shape[0], x1_wavelet.device
        t = torch.rand(batch_size, device=device)
        z = torch.randn_like(x1_wavelet)
        t_broadcast = t.view(batch_size, *([1] * (x1_wavelet.dim() - 1)))
        
        xt = t_broadcast * x1_wavelet + (1 - t_broadcast) * z
        v_target = (x1_wavelet - xt) / (1 - t_broadcast + 1e-8)
        
        v_pred = self.flow_net(xt, t, conditions_dict)
        return self._calculate_weighted_loss(v_pred, v_target, global_step, use_wandb)

    def compute_trigonometric_flow_loss(self, x1_wavelet, conditions_dict, global_step=0, use_wandb=False):
        batch_size, device = x1_wavelet.shape[0], x1_wavelet.device
        t = torch.rand(batch_size, device=device)
        x0_wavelet = torch.randn_like(x1_wavelet)
        t_broadcast = t.view(batch_size, *([1] * (x1_wavelet.dim() - 1)))
        
        angle = (math.pi / 2.0) * t_broadcast
        xt = torch.cos(angle) * x0_wavelet + torch.sin(angle) * x1_wavelet
        v_target = -torch.sin(angle) * (math.pi / 2.0) * x0_wavelet + \
                   torch.cos(angle) * (math.pi / 2.0) * x1_wavelet
                   
        v_pred = self.flow_net(xt, t, conditions_dict)
        return self._calculate_weighted_loss(v_pred, v_target, global_step, use_wandb)

    @torch.no_grad()
    def sample(self, num_samples, model_output_size, conditions_dict, 
               return_trajectory=False, 
               disable_cross_attn_inference=False,
               disable_cond_film_inference=False):
        device = next(self.flow_net.parameters()).device
        self.flow_net.eval()

        d_mod, h_mod, w_mod = model_output_size
        if d_mod % 2 != 0 or h_mod % 2 != 0 or w_mod % 2 != 0:
            raise ValueError(f"Model output size {model_output_size} must be divisible by 2 for DWT/IDWT.")
        wavelet_shape = (num_samples, 8, d_mod // 2, h_mod // 2, w_mod // 2)
        x_wavelet_t = torch.randn(wavelet_shape, device=device)

        processed_conditions = {}
        if conditions_dict and self.condition_dims:
            for key, value in conditions_dict.items():
                if key not in self.condition_dims:
                    logger.warning(f"Condition '{key}' provided for sampling but not defined in model. Skipping.")
                    continue
                value = value.to(device)
                if value.shape[0] == 1 and num_samples > 1:
                    processed_conditions[key] = value.repeat(num_samples, *([1]*(value.ndim-1)))
                elif value.shape[0] != num_samples:
                    raise ValueError(f"Condition '{key}' batch size {value.shape[0]} != num_samples {num_samples}")
                else:
                    processed_conditions[key] = value
        elif self.condition_dims:
            logger.warning("Model expects conditions, but none provided for sampling. Using zeros.")
            for k, dim_spec in self.condition_dims.items():
                 dim_val = 1
                 processed_conditions[k] = torch.zeros(num_samples, dim_val, device=device)


        dt = 1.0 / self.num_flow_steps
        trajectory = [x_wavelet_t.clone()] if return_trajectory else None
        pbar_sample = tqdm(range(self.num_flow_steps), desc="Flow Euler Sampling", leave=False, disable=False)

        for i in pbar_sample:
            t_i = torch.tensor([i * dt] * num_samples, device=device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                 velocity = self.flow_net(x_wavelet_t, t_i, processed_conditions,
                                          disable_cross_attn_inference=disable_cross_attn_inference,
                                          disable_cond_film_inference=disable_cond_film_inference)
            x_wavelet_t = x_wavelet_t + dt * velocity
            if return_trajectory: trajectory.append(x_wavelet_t.clone())

        x_wavelet_1 = x_wavelet_t
        coeffs_tuple = torch.split(x_wavelet_1, 1, dim=1)
        
        lll_rescaled = coeffs_tuple[0] * 1.0 
        
        image_recon = self.idwt(lll_rescaled, *coeffs_tuple[1:])
        image_recon = torch.clamp(image_recon, -1.0, 1.0)

        if return_trajectory:
             return torch.stack(trajectory, dim=1)
        return image_recon

    def loss(self, x_wavelet, conditions_dict, global_step=0, use_wandb=False):
        if self.flow_type == "rectified":
            return self.compute_rectified_flow_loss(x_wavelet, conditions_dict, global_step, use_wandb)
        elif self.flow_type == "vp_diffusion":
            return self.compute_vp_diffusion_loss(x_wavelet, conditions_dict, global_step, use_wandb)
        elif self.flow_type == "cfm":
            return self.compute_cfm_loss(x_wavelet, conditions_dict, global_step, use_wandb)
        elif self.flow_type == "trigonometric":
            return self.compute_trigonometric_flow_loss(x_wavelet, conditions_dict, global_step, use_wandb)
        else:
            raise NotImplementedError(f"Loss for flow_type '{self.flow_type}' is not implemented.")