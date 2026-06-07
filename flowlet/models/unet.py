import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.logging_utils import get_logger
from ..modules.utils import get_norm_layer, zero_module
from ..modules.embeddings import MultiLabelEmbedding, timestep_embedding
from ..modules.layers import TimestepEmbedSequential
from ..modules.blocks import ResBlock, Upsample, Downsample
from ..modules.attention import SpatialTransformerConditional, XFORMERS_AVAILABLE

logger = get_logger(__name__)

class ConditionalUNet(nn.Module):
    """
    Conditional 3D U-Net using ResBlocks with FiLM and Spatial Conditioning (SpatialTransformerConditional) for attention.

    Note: `dims` is accepted for backward compatibility with saved `unet_args` configs but is
    unused — the network is 3D only (nn.Conv3d everywhere).
    """
    def __init__(
        self,
        in_channels=8, model_channels=64, out_channels=8, num_res_blocks=2,
        attention_resolutions=(8, 4), dropout=0.1, channel_mult=(1, 2, 4, 8),
        conv_resample=True, dims=3, use_checkpoint=False, num_heads=8,
        num_head_channels=-1, use_scale_shift_norm=True, resblock_updown=True,
        condition_dims=None, condition_embedding_dim=128,
        use_cross_attention=True,
        use_xformers=False,
        norm_num_groups=32,
        norm_eps=1e-6
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.condition_dims = condition_dims if condition_dims else {}
        self.condition_embedding_dim = condition_embedding_dim
        self.use_cross_attention = use_cross_attention and self.condition_dims
        self.use_xformers = use_xformers
        self.norm_num_groups=norm_num_groups
        self.norm_eps=norm_eps

        if self.use_xformers and not XFORMERS_AVAILABLE:
            logger.warning("xformers requested for U-Net, but not installed. Attention blocks will use native PyTorch implementation.")
            self.use_xformers = False # Force disable if not available

        # Condition Embedding
        self.condition_embedder = MultiLabelEmbedding(self.condition_dims, self.condition_embedding_dim)
        self.eff_condition_dim = self.condition_embedding_dim if self.use_cross_attention else None

        # Timestep Embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- Downsampling ---
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv3d(in_channels, model_channels, 3, padding=1))])
        input_block_chans = [model_channels]; ch = model_channels; ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch,
                                   time_embed_dim,
                                   dropout,
                                   out_channels=mult * model_channels,
                                   use_checkpoint=use_checkpoint,
                                   use_scale_shift_norm=use_scale_shift_norm,
                                   condition_dim=self.condition_embedding_dim,
                                   norm_num_groups=norm_num_groups,
                                   norm_eps=norm_eps)]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    curr_num_heads = self.num_heads
                    curr_head_channels = self.num_head_channels
                    if curr_head_channels == -1:
                        assert ch % curr_num_heads == 0, f"ch={ch} not divisible by num_heads={curr_num_heads}"
                        curr_head_channels = ch // curr_num_heads
                    else:
                        assert ch % curr_head_channels == 0, f"ch={ch} not divisible by head_channels={curr_head_channels}"
                        curr_num_heads = ch // curr_head_channels

                    # Add Spatial Conditioning with SpatialTransformerConditional
                    layers.append(SpatialTransformerConditional(
                        in_channels=ch,
                        num_attention_heads=curr_num_heads,
                        num_head_channels=curr_head_channels,
                        context_dim=self.eff_condition_dim,
                        dropout=dropout,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        use_checkpoint=use_checkpoint,
                        use_xformers=self.use_xformers,
                    ))
                    
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                down_layer = ResBlock(ch,
                                      time_embed_dim,
                                      dropout,
                                      out_channels=out_ch,
                                      use_checkpoint=use_checkpoint,
                                      use_scale_shift_norm=use_scale_shift_norm,
                                      down=True,
                                      condition_dim=self.condition_embedding_dim,
                                      norm_num_groups=norm_num_groups,
                                      norm_eps=norm_eps) if resblock_updown else Downsample(ch,
                                                                                            conv_resample,
                                                                                            out_channels=out_ch)
                self.input_blocks.append(TimestepEmbedSequential(down_layer))
                ch = out_ch; input_block_chans.append(ch); ds *= 2

        # --- Bottleneck ---
        curr_num_heads = self.num_heads
        curr_head_channels = self.num_head_channels
        if curr_head_channels == -1:
            assert ch % curr_num_heads == 0, f"Bottleneck ch={ch} not divisible by num_heads={curr_num_heads}"
            curr_head_channels = ch // curr_num_heads
        else:
            assert ch % curr_head_channels == 0, f"Bottleneck ch={ch} not divisible by head_channels={curr_head_channels}"
            curr_num_heads = ch // curr_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     condition_dim=self.condition_embedding_dim,
                     norm_num_groups=norm_num_groups,
                     norm_eps=norm_eps),
            SpatialTransformerConditional(
                in_channels=ch,
                num_attention_heads=curr_num_heads,
                num_head_channels=curr_head_channels,
                context_dim=self.eff_condition_dim,
                dropout=dropout,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                use_checkpoint=use_checkpoint,
                use_xformers=self.use_xformers,
            ),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm,
                     condition_dim=self.condition_embedding_dim,
                     norm_num_groups=norm_num_groups,
                     norm_eps=norm_eps)
        )

        # --- Upsampling ---
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,
                                   time_embed_dim,
                                   dropout,
                                   out_channels=model_channels * mult,
                                   use_checkpoint=use_checkpoint,
                                   use_scale_shift_norm=use_scale_shift_norm,
                                   condition_dim=self.condition_embedding_dim,
                                   norm_num_groups=norm_num_groups,
                                   norm_eps=norm_eps)]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    curr_num_heads = self.num_heads
                    curr_head_channels = self.num_head_channels
                    if curr_head_channels == -1:
                        assert ch % curr_num_heads == 0, f"Up ch={ch} not divisible by num_heads={curr_num_heads}"
                        curr_head_channels = ch // curr_num_heads
                    else:
                        assert ch % curr_head_channels == 0, f"Up ch={ch} not divisible by head_channels={curr_head_channels}"
                        curr_num_heads = ch // curr_head_channels

                    layers.append(SpatialTransformerConditional(
                        in_channels=ch,
                        num_attention_heads=curr_num_heads,
                        num_head_channels=curr_head_channels,
                        context_dim=self.eff_condition_dim,
                        dropout=dropout,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        use_checkpoint=use_checkpoint,
                        use_xformers=self.use_xformers,
                    ))
                if level != 0 and i == num_res_blocks:
                    out_ch = ch
                    up_layer = ResBlock(ch,
                                        time_embed_dim,
                                        dropout,
                                        out_channels=out_ch,
                                        use_checkpoint=use_checkpoint,
                                        use_scale_shift_norm=use_scale_shift_norm,
                                        up=True,
                                        condition_dim=self.condition_embedding_dim,
                                        norm_num_groups=norm_num_groups,
                                        norm_eps=norm_eps) if resblock_updown else Upsample(ch,
                                                                                            conv_resample,
                                                                                            out_channels=out_ch)
                    layers.append(up_layer)
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output Layer. The final feature map has `ch` channels (model_channels * channel_mult[0]),
        # so both the norm and the output conv must use `ch` as their input width.
        self.out = nn.Sequential(
            get_norm_layer(ch, norm_num_groups, norm_eps), nn.SiLU(),
            zero_module(nn.Conv3d(ch, out_channels, 3, padding=1))
        )

    def forward(self, x, timesteps, conditions_dict=None, 
                disable_cross_attn_inference=False, 
                disable_cond_film_inference=False):
        hs = []; emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Get the single combined condition embedding (or zeros if no conditions)
        _, combined_cond_emb = self.condition_embedder(conditions_dict if conditions_dict else {}, batch_size=x.size(0))

        # Prepare context for Spatial Conditioning: it should be None if cross-attention is disabled
        if disable_cross_attn_inference:
            context_for_transformer = None
        elif self.use_cross_attention:
            context_for_transformer = combined_cond_emb
        else:
            context_for_transformer = None
        
        # Ensure context is not None if cross-attention is used and it's needed
        if self.use_cross_attention and not disable_cross_attn_inference and context_for_transformer is None:
             logger.warning("Cross attention requested but combined_cond_emb is None (and not disabled for inference). Using zeros for context.")
             context_for_transformer = torch.zeros(x.size(0), self.eff_condition_dim if self.eff_condition_dim is not None else self.condition_embedding_dim, device=x.device, dtype=x.dtype)


        h = x.type(self.dtype)
        # Pass combined_cond_emb to all blocks (ResBlock uses it for FiLM, SpatialTransformer uses it as context)
        for module in self.input_blocks:
            h = module(h, emb, context_for_transformer, disable_cond_film_inference=disable_cond_film_inference)
            hs.append(h)
        h = self.middle_block(h, emb, context_for_transformer, disable_cond_film_inference=disable_cond_film_inference)
        for module in self.output_blocks:
            if not hs: raise RuntimeError("Mismatch in U-Net skip connections - hs stack is empty.")
            skip_h = hs.pop()
            if h.shape[0] != skip_h.shape[0]: raise RuntimeError(f"Batch size mismatch in skip connection: h={h.shape[0]}, skip={skip_h.shape[0]}")
            if h.shape[2:] != skip_h.shape[2:]:
                 logger.warning(f"Spatial dimension mismatch in skip connection: h={h.shape[2:]}, skip={skip_h.shape[2:]}. Attempting to resize skip connection.")
                 skip_h = F.interpolate(skip_h, size=h.shape[2:], mode='trilinear', align_corners=False)

            h = torch.cat([h, skip_h], dim=1)
            h = module(h, emb, context_for_transformer, disable_cond_film_inference=disable_cond_film_inference)
        h = h.type(x.dtype)
        return self.out(h)