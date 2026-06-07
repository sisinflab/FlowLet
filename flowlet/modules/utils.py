import torch
import torch.nn as nn
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_norm_layer(channels, num_groups=32, eps=1e-5):
    if channels == 0: return nn.Identity()
    if num_groups <= 0: num_groups = 1
    if channels < num_groups:
        potential_num_groups = [i for i in range(1, channels // 2 + 1) if channels % i == 0]
        num_groups = max(potential_num_groups) if potential_num_groups else 1
    while channels % num_groups != 0 and num_groups > 1:
        found_divisor = False
        for i in range(num_groups - 1, 0, -1):
            if channels % i == 0: num_groups = i; found_divisor = True; break
        if not found_divisor: num_groups = 1
    return nn.GroupNorm(num_groups, channels, eps=eps)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters(): p.detach().zero_()
    return module