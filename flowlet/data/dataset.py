import os
import glob
import re
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, Subset
from ..utils.logging_utils import get_logger

from ..wavelets import dwt_3d
from .volume_ops import robust_normalize, pad_to_size

logger = get_logger(__name__)


class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        dataset = subset
        while isinstance(dataset, torch.utils.data.Subset): dataset = dataset.dataset
        self.original_dataset = dataset
        self.transform = transform
    def __getitem__(self, index):
        original_index = self.subset.indices[index]
        original_transform = self.original_dataset.transform
        self.original_dataset.transform = self.transform
        try: item = self.original_dataset[original_index]
        finally: self.original_dataset.transform = original_transform
        return item
    def __len__(self): return len(self.subset)

# This dataset class is designed to load 3D MRI data from NIfTI files, extract conditions from filenames,
# normalize the data, and apply padding and wavelet transforms. 
# It is tailored to ease reproduciblity task for Paper Submission, providing a uniform datalaoder interface beween FlowLet and other baselines.
# The dataset can be easily adapted to take a metadata CSV file. An implementation is provided in the dataset_csv.py file.
# The dataset can be filtered based on cognitive status and other conditions.

class EnhancedBrainMRIDataset(Dataset):
    """
    Loads NIfTI files, extracts conditions (Age in this case) directly from filenames,
    normalizes, pads, optionally augments, and applies DWT.
    Ignores CSV metadata and cognitive status filters.
    """
    def __init__(self, data_folder, metadata_path=None, transform=None,
                model_input_size=(112, 112, 112),
                 filter_cognitive_status=None, condition_vars=None,
                 require_conditions=True):

        self.data_folder = data_folder
        self.transform = transform
        self.model_input_size = model_input_size
        self.condition_vars = condition_vars or []
        if self.condition_vars and 'age' not in self.condition_vars:
             logger.warning(f"'age' not in requested condition_vars {self.condition_vars}, but filename parsing focuses on AGE.")
             if 'age' not in self.condition_vars:
                 self.condition_vars.append('age')
                 logger.warning(f"Automatically added 'age' to condition_vars.")

        self.require_conditions = require_conditions

        self.file_list = []
        self.extracted_conditions = []
        self.condition_ranges = {}

        logger.info(f"Scanning data folder: {data_folder} for .nii.gz files and extracting conditions from filenames.")
        all_files = sorted(glob.glob(os.path.join(data_folder, "*.nii.gz")))
        logger.info(f"Found {len(all_files)} total .nii.gz files initially.")

        temp_extracted_conditions = defaultdict(list)

        for file_path in tqdm(all_files, desc="Parsing Filenames"):
            filename = os.path.basename(file_path)
            conditions_found = {}
            parse_success = True

            age_match = re.search(r'[_-]AGE[_-]([0-9.]+)', filename, re.IGNORECASE)

            if 'age' in self.condition_vars:
                if age_match:
                    try:
                        age_val = float(age_match.group(1))
                        conditions_found['age'] = age_val
                    except ValueError:
                        logger.warning(f"Could not convert extracted AGE '{age_match.group(1)}' to float in filename: {filename}")
                        parse_success = False
                else:
                    parse_success = False
                    logger.debug(f"Could not find '_AGE_value' pattern in filename: {filename}")

            all_required_found = True
            if self.require_conditions:
                for req_cond in self.condition_vars:
                    if req_cond not in conditions_found:
                        all_required_found = False
                        logger.warning(f"Required condition '{req_cond}' not found or parsed in filename: {filename}. Skipping file.")
                        break

            if parse_success and all_required_found:
                self.file_list.append(file_path)
                self.extracted_conditions.append(conditions_found)
                for cond_name, value in conditions_found.items():
                    if isinstance(value, (int, float)):
                        temp_extracted_conditions[cond_name].append(value)

        logger.info(f"Found {len(self.file_list)} files with successfully parsed required conditions.")

        for cond_name, values_list in temp_extracted_conditions.items():
            if values_list:
                min_val = float(np.min(values_list))
                max_val = float(np.max(values_list))
                self.condition_ranges[cond_name] = {'min': min_val, 'max': max_val}
                logger.info(f"Calculated range for '{cond_name}': {min_val:.2f} to {max_val:.2f}")
            else:
                 logger.warning(f"No valid numeric values found for condition '{cond_name}' extracted from filenames.")

        if 'age' in self.condition_vars and 'age' not in self.condition_ranges:
             logger.error(f"Failed to calculate age range, though 'age' was requested. Check filename parsing.")

        self.final_indices = list(range(len(self.file_list)))
        logger.info(f"Final dataset size: {len(self.final_indices)}")
        if len(self.final_indices) == 0:
             raise RuntimeError("Dataset empty after parsing filenames. Check paths, filenames, and required conditions.")

    def __len__(self):
        return len(self.final_indices)

    def _robust_normalize(self, data):
        return robust_normalize(data)

    def _pad_data(self, data_tensor, target_size):
        return pad_to_size(data_tensor, target_size)

    def __getitem__(self, idx):
        actual_idx = self.final_indices[idx]
        file_path = self.file_list[actual_idx]
        try:
            img = nib.load(file_path)
            mri_data = img.get_fdata(dtype=np.float32)
            mri_data = self._robust_normalize(mri_data)
            if not np.isfinite(mri_data).all():
                logger.error(f"Non-finite values detected after normalization for {file_path}. Replacing with 0.")
                mri_data = np.nan_to_num(mri_data, nan=0.0, posinf=1.0, neginf=-1.0)

            tensor_data = torch.from_numpy(mri_data)
            if tensor_data.shape != self.model_input_size:
                tensor_data = self._pad_data(tensor_data, self.model_input_size)

            if tensor_data.ndim == 3: tensor_data = tensor_data.unsqueeze(0)

            if self.transform:
                tensor_data = self.transform(tensor_data)
                tensor_data = tensor_data.float()
                if not torch.isfinite(tensor_data).all():
                    logger.error(f"Non-finite values detected after augmentation for {file_path}. Replacing with 0.")
                    tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)

            if tensor_data.ndim == 4: tensor_data = tensor_data.unsqueeze(0)
            elif tensor_data.ndim != 5: 
                raise ValueError(f"Unexpected tensor dimension before DWT: {tensor_data.ndim} for {file_path}")
            if tensor_data.shape[-3:] != self.model_input_size: 
                raise ValueError(f"Tensor shape {tensor_data.shape} mismatch before DWT, expected spatial {self.model_input_size} for {file_path}")
            if not torch.isfinite(tensor_data).all():
                logger.error(f"Non-finite values detected before DWT for {file_path}. Replacing with 0.")
                tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)

            wavelet_coeffs_tuple = dwt_3d(tensor_data)
            wavelet_coeffs_list = [wavelet_coeffs_tuple[0] / 1.0] + list(wavelet_coeffs_tuple[1:])
            for i, coeff in enumerate(wavelet_coeffs_list):
                if not torch.isfinite(coeff).all():
                     logger.error(f"Non-finite values detected in wavelet coeff {i} for {file_path}. Replacing with 0.")
                     wavelet_coeffs_list[i] = torch.nan_to_num(coeff, nan=0.0, posinf=1.0, neginf=-1.0)
            wavelet_coeffs_concat = torch.cat(wavelet_coeffs_list, dim=1) # Shape: (1, 8, D', H', W')

            conditions_out = {}
            raw_conditions = self.extracted_conditions[actual_idx]
            for cond_name in self.condition_vars:
                if cond_name in raw_conditions:
                    raw_value = raw_conditions[cond_name]
                    if cond_name in self.condition_ranges:
                        min_v, max_v = self.condition_ranges[cond_name]['min'], self.condition_ranges[cond_name]['max']
                        norm_value = (raw_value - min_v) / (max_v - min_v) if max_v > min_v else 0.5
                        conditions_out[cond_name] = torch.tensor(norm_value, dtype=torch.float32)
                    elif cond_name == 'sex': # Example categorical handling Not used in this submission
                         sex_map = {'F': 0.0, 'M': 1.0}
                         encoded_value = sex_map.get(str(raw_value).upper(), 0.0)
                         conditions_out[cond_name] = torch.tensor(encoded_value, dtype=torch.float32)
                         logger.warning(f"Used hardcoded sex encoding for {file_path}. Consider making this more robust if using other categorical vars.")
                    else:
                         logger.warning(f"Condition '{cond_name}' found in filename but has no normalization range and isn't handled specifically. Using default 0.0.")
                         conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)
                elif self.require_conditions:
                     raise ValueError(f"Required condition '{cond_name}' missing for file {file_path} (idx {actual_idx}) despite init checks.")
                else:
                    conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)

            return wavelet_coeffs_concat.squeeze(0), conditions_out

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            return None