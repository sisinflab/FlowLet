import torch
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN deterministic mode is intentionally left disabled: it can slow down
        # 3D convolutions and is not required to reproduce the reported results.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False