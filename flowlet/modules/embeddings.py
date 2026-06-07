import torch
import torch.nn as nn
import math
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class MultiLabelEmbedding(nn.Module):
    def __init__(self, condition_dims, embedding_dim):
        super().__init__()
        self.condition_dims = condition_dims if condition_dims else {}
        self.embedding_dim = embedding_dim
        self.embedders = nn.ModuleDict()
        if not self.condition_dims: logger.warning("MultiLabelEmbedding init with no condition_dims.")
        else:
            for name, dim in self.condition_dims.items():
                 if dim <= 0: raise ValueError(f"Dim for '{name}' must be > 0")
                 self.embedders[name] = nn.Sequential(nn.Linear(dim, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim))
            logger.info(f"Initialized MultiLabelEmbedding for: {list(self.condition_dims.keys())}")

    def forward(self, conditions_dict, batch_size):
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        combined_embedding = torch.zeros(batch_size, self.embedding_dim, device=device)
        if not self.condition_dims or not conditions_dict: return {}, combined_embedding

        processed_conditions = {}
        for name, embedder in self.embedders.items():
            if name in conditions_dict:
                tensor = conditions_dict[name].to(device=device, dtype=torch.float32)
                if tensor.ndim == 1: tensor = tensor.unsqueeze(1)
                if tensor.shape[0] != batch_size: raise ValueError(f"Batch size mismatch for condition '{name}'. Expected {batch_size}, got {tensor.shape[0]}.")
                if tensor.shape[1] != self.condition_dims[name]: raise ValueError(f"Dimension mismatch for condition '{name}'. Expected {self.condition_dims[name]}, got {tensor.shape[1]}.")
                combined_embedding = combined_embedding + embedder(tensor)
                processed_conditions[name] = tensor
        return processed_conditions, combined_embedding
    

def timestep_embedding(timesteps, dim, max_period=10000):
    # (Standard sinusoidal embedding)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding