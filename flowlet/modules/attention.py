import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from ..utils.logging_utils import get_logger
from .utils import get_norm_layer, zero_module


logger = get_logger(__name__)

# Optional: Attempt to import xformers
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    logger.info("xformers found and imported successfully in attention module.")
except ImportError:
    XFORMERS_AVAILABLE = False
    xops = None # Define xops as None if not available
    logger.info("xformers not found. Using native PyTorch attention.")


class SpatialTransformerConditional(nn.Module):
    """
    This is the Spatial Conditioning Block for the UNet.
    Transformer block incorporating self-attention and optional cross-attention.
    Inspired by Diffusers SpatialTransformer, adapted for conditional input.
    Operates on 3D spatial data.
    """
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        num_head_channels: int,
        context_dim: int | None = None, # Dimension of the conditional embedding
        num_layers: int = 1,            # Number of transformer blocks (kept at 1 for now)
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        use_checkpoint: bool = False,
        use_xformers: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_attention_heads
        self.head_dim = num_head_channels
        self.inner_dim = num_attention_heads * num_head_channels
        self.context_dim = context_dim
        self.use_checkpoint = use_checkpoint
        self.use_xformers = use_xformers and XFORMERS_AVAILABLE

        # Group norm
        self.norm = get_norm_layer(in_channels, norm_num_groups)

        # Projection before attention
        self.proj_in = nn.Conv3d(in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        # --- Attention Mechanisms ---
        # combine self and cross attention logic inspired by BasicTransformerBlock
        # Layer Normalization for sequence inputs
        self.norm_seq = nn.LayerNorm(self.inner_dim, eps=norm_eps)
        if context_dim is not None:
            self.norm_context = nn.LayerNorm(context_dim, eps=norm_eps)

        # --- Self-Attention layers ---
        self.to_q = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_self_attn_out = nn.Linear(self.inner_dim, self.inner_dim)

        # --- Cross-Attention specific layers (if context_dim is provided) ---
        if context_dim is not None:
            self.to_q_cross = nn.Linear(self.inner_dim, self.inner_dim, bias=False)
            self.to_kv_cross = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
            self.proj_cross_attn_out = nn.Linear(self.inner_dim, self.inner_dim) # Projection after cross-attention

        # Final feed-forward projection
        self.proj_out = zero_module(nn.Conv3d(self.inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        # --- End Attention Mechanisms ---


        if self.use_xformers:
            logger.debug(f"SpatialTransformerConditional (channels={in_channels}, context={context_dim}) using xformers.")
        else:
            logger.debug(f"SpatialTransformerConditional (channels={in_channels}, context={context_dim}) using PyTorch SDPA.")


    def _forward_attention(self, q_in, k_in, v_in):
        """ Helper function for applying attention (xformers or native) """
        b, n_head, n_seq, d_head = q_in.shape

        if self.use_xformers:
            # xformers expects (B, N_seq, N_head, D_head) for memory_efficient_attention
            # It requires (batch_size, seq_len, num_heads, head_dim)
            q_xf = q_in.permute(0, 2, 1, 3).reshape(b, n_seq, n_head, d_head).contiguous()
            k_xf = k_in.permute(0, 2, 1, 3).reshape(b, -1, n_head, d_head).contiguous()
            v_xf = v_in.permute(0, 2, 1, 3).reshape(b, -1, n_head, d_head).contiguous()

            attn_output = xops.memory_efficient_attention(
                q_xf, k_xf, v_xf, p=self.attn_dropout.p if self.training else 0.0
            )   # Output: (B, N_seq, N_head, D_head)
                # Reshape back to (B, N_head, N_seq, D_head) standard internal format
            attn_output = attn_output.reshape(b, n_seq, n_head, d_head).permute(0, 2, 1, 3)

        else:
            # Use PyTorch's scaled_dot_product_attention
            # Input shape: (B, N_head, N_seq_q, D_head) and (B, N_head, N_seq_kv, D_head)
            attn_output = F.scaled_dot_product_attention(
                q_in, k_in, v_in,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            ) # Output: (B, N_head, N_seq_q, D_head)

        return attn_output

    def _forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        b, c, d, h, w = x.shape
        residual = x

        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm) # Shape: B, inner_dim, D, H, W

        # Reshape for attention
        x_seq = x_proj.view(b, self.inner_dim, -1).transpose(1, 2) # Shape: B, N, inner_dim (N=D*H*W)
        x_seq = self.norm_seq(x_seq) # LayerNorm on sequence

        # --- Self-Attention ---
        q_self = self.to_q(x_seq)
        k_self = self.to_k(x_seq)
        v_self = self.to_v(x_seq)

        # Reshape for multi-head attention: B, N, inner_dim -> B, N_head, N, D_head
        q_self = q_self.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_self = k_self.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_self = v_self.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Attention
        self_attn_output_h = self._forward_attention(q_self, k_self, v_self)

        # Reshape back: B, N_head, N, D_head -> B, N, inner_dim
        self_attn_output = self_attn_output_h.transpose(1, 2).reshape(b, -1, self.inner_dim)
        self_attn_output = self.proj_self_attn_out(self_attn_output)

        # Add residual after self-attention
        x_seq = x_seq + self_attn_output
        # --- End Self-Attention ---

        # --- Cross-Attention ---
        if self.context_dim is not None and context is not None:
            # Prepare context
            context_norm = self.norm_context(context) # Context shape in this version (B, 1, context_dim). Expandable to (B, N_ctx, context_dim)
            q_cross = self.to_q_cross(x_seq) # Q from self-attended sequence
            kv_cross = self.to_kv_cross(context_norm) # K, V from context
            k_cross, v_cross = kv_cross.chunk(2, dim=-1) # Split K, V

            # Reshape for multi-head attention
            q_cross = q_cross.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # B, N_head, N, D_head
            k_cross = k_cross.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # B, N_head, N_ctx, D_head
            v_cross = v_cross.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2) # B, N_head, N_ctx, D_head

            # Apply Attention
            cross_attn_output_h = self._forward_attention(q_cross, k_cross, v_cross)

            # Reshape back
            cross_attn_output = cross_attn_output_h.transpose(1, 2).reshape(b, -1, self.inner_dim)
            cross_attn_output = self.proj_cross_attn_out(cross_attn_output)

            # Add residual after cross-attention
            x_seq = x_seq + cross_attn_output
        # --- End Cross-Attention ---

        # Reshape back to spatial dimensions
        x_out = x_seq.transpose(1, 2).reshape(b, self.inner_dim, d, h, w)

        # Final projection and residual
        x_out = self.proj_out(x_out)
        return x_out + residual


    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)