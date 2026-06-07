import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from monai.transforms import RandRotate, Compose, RandGaussianNoise, RandScaleIntensity
import numpy as np
from ..utils.logging_utils import get_logger

# Import dataset classes
from .dataset import EnhancedBrainMRIDataset, TransformedSubset

logger = get_logger(__name__)


train_transform = Compose([
        RandRotate(range_x=np.pi/36, range_y=np.pi/36, range_z=np.pi/36, prob=0.4, keep_size=True, mode='trilinear'),
        RandScaleIntensity(factors=0.2, prob=0.2),
        RandGaussianNoise(std=0.02, prob=0.2)
    ])
val_transform = None


def collate_fn(batch):
    num_dropped = sum(1 for x in batch if x is None)
    batch = list(filter(lambda x: x is not None, batch))
    if num_dropped:
        logger.warning(f"collate_fn dropped {num_dropped} sample(s) in this batch due to load/processing errors.")
    if not batch: return None
    try:
        wavelets = torch.stack([item[0] for item in batch], dim=0)
        all_conditions = [item[1] for item in batch]
        batched_conditions = {}
        if all_conditions and isinstance(all_conditions[0], dict):
            condition_keys = all_conditions[0].keys()
            for key in condition_keys:
                if all(key in cond_dict for cond_dict in all_conditions):
                    tensors_to_stack = [cond_dict[key] for cond_dict in all_conditions]
                    if tensors_to_stack and tensors_to_stack[0].ndim == 0:
                        tensors_to_stack = [t.unsqueeze(0) for t in tensors_to_stack]
                    if tensors_to_stack:
                         batched_conditions[key] = torch.stack(tensors_to_stack, dim=0)
                else:
                     logger.warning(f"Condition key '{key}' missing in some samples within the batch. Skipping this key.")
        return wavelets, batched_conditions
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}", exc_info=True)
        return None


def create_brain_dataset_and_split(data_folder, metadata_path=None, transform_train=None, transform_val=None,
                                   model_input_size=(112, 112, 112),
                                   filter_cognitive_status="Cognitively normal",
                                   condition_vars=None, require_conditions=True,
                                   val_split=0.2, seed=42):
    logger.info(f"Creating dataset: model_in_size={model_input_size}, conditions={condition_vars}, require={require_conditions}, filter='{filter_cognitive_status}'")
    full_dataset = EnhancedBrainMRIDataset(
        data_folder=data_folder, metadata_path=metadata_path, transform=None,
        model_input_size=model_input_size,
        filter_cognitive_status=filter_cognitive_status if filter_cognitive_status else None,
        condition_vars=condition_vars, require_conditions=require_conditions
    )
    if len(full_dataset) == 0: raise RuntimeError("Dataset creation resulted in empty dataset.")
    val_split = max(0.0, min(1.0, val_split))
    if val_split == 0.0 or val_split == 1.0:
        logger.warning(f"Validation split is {val_split}, dataset will not be split.")
        if val_split == 0.0:
            train_dataset_transformed = TransformedSubset(full_dataset, transform_train)
            val_dataset_transformed = TensorDataset(torch.empty(0))
            val_size = 0; train_size = len(full_dataset)
        else:
            val_dataset_transformed = TransformedSubset(full_dataset, transform_val)
            train_dataset_transformed = TensorDataset(torch.empty(0))
            train_size = 0; val_size = len(full_dataset)
    else:
        train_size = int((1.0 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        logger.info(f"Splitting dataset: {train_size} train, {val_size} validation samples.")
        if train_size == 0 or val_size == 0: raise ValueError("Dataset split resulted in zero samples for train or validation.")
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
        train_dataset_transformed = TransformedSubset(train_subset, transform_train)
        val_dataset_transformed = TransformedSubset(val_subset, transform_val)

    condition_ranges = full_dataset.condition_ranges
    return train_dataset_transformed, val_dataset_transformed, condition_ranges