import torch
import torch.nn as nn
from ..utils.logging_utils import get_logger
from .blocks import ResBlock, Upsample, Downsample
from .attention import SpatialTransformerConditional

logger = get_logger(__name__)

class TimestepEmbedSequential(nn.Sequential):
    """
    Sequential module passing timestep and condition embeddings correctly
    to ResBlock and SpatialTransformerConditional layers.
    """
    def forward(self, x, emb, cond_emb=None, disable_cond_film_inference=False):
        for layer in self:
            if isinstance(layer, ResBlock):
                # ResBlock uses time and condition embedding
                x = layer(x, emb, cond_emb, disable_cond_film_inference=disable_cond_film_inference)
            elif isinstance(layer, SpatialTransformerConditional):
                # SpatialTransformerConditional uses the feature map x and the condition embedding (as context)
                x = layer(x, context=cond_emb)
            elif isinstance(layer, (Upsample, Downsample)):
                # Upsample/Downsample don't use embeddings
                x = layer(x)
            else:
                # Default behavior for other layers (like initial Conv3d)
                x = layer(x)
        return x
