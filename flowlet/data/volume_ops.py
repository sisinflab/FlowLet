"""Shared volume preprocessing used by both the folder- and CSV-based datasets."""

import numpy as np
import torch
import torch.nn.functional as F
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def robust_normalize(data):
    """Percentile-clip intensities to [0.5, 99.5] and rescale to [-1, 1]. Returns float32."""
    p_low, p_high = np.percentile(data, [0.5, 99.5])
    data = np.clip(data, p_low, p_high)
    denom = p_high - p_low
    if denom < 1e-8:
        logger.warning("Normalization range near zero, returning zeros.")
        return np.zeros_like(data, dtype=np.float32)
    data = (data - p_low) / denom
    data = np.nan_to_num(data)
    data = data * 2.0 - 1.0
    if not np.isfinite(data).all():
        logger.error("Non-finite values found after robust normalization!")
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    # np.percentile returns float64 scalars, which promote the array to float64.
    # Cast back to float32 to match the float32 wavelet (DWT) filters.
    return data.astype(np.float32)


def pad_to_size(data_tensor, target_size):
    """Replication-pad the trailing 3 spatial dims of a tensor up to target_size (D, H, W)."""
    current_spatial_shape = data_tensor.shape[-3:]
    pad_needed = [max(0, o - i) for o, i in zip(target_size, current_spatial_shape)]
    if all(p == 0 for p in pad_needed):
        return data_tensor
    padding = []
    for dim_pad in reversed(pad_needed):
        pad1 = dim_pad // 2
        pad2 = dim_pad - pad1
        padding.extend([pad1, pad2])

    needs_unsqueeze_b = data_tensor.ndim < 5
    needs_unsqueeze_c = data_tensor.ndim < 4
    temp_tensor = data_tensor
    if needs_unsqueeze_c: temp_tensor = temp_tensor.unsqueeze(0)
    if needs_unsqueeze_b: temp_tensor = temp_tensor.unsqueeze(0)
    while temp_tensor.ndim < 5: temp_tensor = temp_tensor.unsqueeze(0)

    padded_tensor = F.pad(temp_tensor, tuple(padding), mode="replicate")

    if needs_unsqueeze_b: padded_tensor = padded_tensor.squeeze(0)
    if needs_unsqueeze_c: padded_tensor = padded_tensor.squeeze(0)

    if tuple(padded_tensor.shape[-3:]) != tuple(target_size):
        logger.warning(f"Padding resulted in unexpected spatial shape: {padded_tensor.shape[-3:]} vs target {target_size}")
    return padded_tensor
