import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..utils.logging_utils import get_logger
from .utils import get_norm_layer, zero_module

logger = get_logger(__name__)

class Upsample(nn.Module):
    """Upsampling layer, potentially using convolution."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # Use nearest neighbor interpolation for upsampling
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """Downsampling layer, potentially using convolution."""
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            # Use Conv3d for learned downsampling
            self.op = nn.Conv3d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            # Use AvgPool3d for simple downsampling
            self.op = nn.AvgPool3d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(nn.Module):
    """Residual block with timestep and optional condition embedding using FiLM."""
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=True,
                 use_scale_shift_norm=True, use_checkpoint=False, up=False, down=False,
                 condition_dim=None, norm_num_groups=32, norm_eps=1e-5):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.condition_dim = condition_dim
        self.up = up
        self.down = down

        self.in_layers = nn.Sequential(get_norm_layer(channels, norm_num_groups, norm_eps), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))

        if condition_dim is not None and use_scale_shift_norm:
             self.cond_emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(condition_dim, 2 * self.out_channels))
        else: self.cond_emb_layers = None

        self.out_layers = nn.Sequential(get_norm_layer(self.out_channels, norm_num_groups, norm_eps), nn.SiLU(), nn.Dropout(p=dropout), zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)))

        if self.out_channels == channels: self.skip_connection = nn.Identity()
        elif use_conv: 
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        else: 
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 1)

    def _forward(self, x, emb, cond_emb=None, disable_cond_film_inference=False):
        if self.up or self.down:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x); h = self.h_upd(h); x = self.x_upd(x); h = in_conv(h)
        else: h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape): 
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale_time, shift_time = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale_time) + shift_time
            
            if not disable_cond_film_inference and self.cond_emb_layers is not None and cond_emb is not None:
                 cond_emb_out = self.cond_emb_layers(cond_emb).type(h.dtype)

                 while len(cond_emb_out.shape) < len(h.shape): 
                     cond_emb_out = cond_emb_out[..., None]

                 scale_cond, shift_cond = torch.chunk(cond_emb_out, 2, dim=1)
                 h = h * (1 + scale_cond) + shift_cond
            h = out_rest(h)
        else:
            h = h + emb_out; h = self.out_layers(h)
        return self.skip_connection(x) + h

    def forward(self, x, emb, cond_emb=None, disable_cond_film_inference=False):
        if self.use_checkpoint and self.training:
            if disable_cond_film_inference:
                 return self._forward(x, emb, cond_emb, disable_cond_film_inference=disable_cond_film_inference)
            return checkpoint(self._forward, x, emb, cond_emb, use_reentrant=False)
        else: return self._forward(x, emb, cond_emb, disable_cond_film_inference=disable_cond_film_inference)