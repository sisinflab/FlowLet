import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from ..utils.logging_utils import get_logger
from ..wavelets import dwt_3d
from .volume_ops import robust_normalize, pad_to_size

logger = get_logger(__name__)


# Note: the canonical TransformedSubset lives in flowlet/data/dataset.py and is the one
# re-exported by the package and used by the dataloader and training script.


class BrainMRIDatasetCSV(Dataset):
    """
    Loads 3D NIfTI files based on metadata from a CSV file.
    Reads absolute file paths directly from the CSV.
    Filters data based on CSV columns (e.g., condition).
    Extracts specified conditions (e.g., Age) from the CSV.
    Normalizes image data, pads to model input size, applies DWT.
    Normalizes condition values based on ranges calculated from the filtered dataset.
    """
    def __init__(self,
                 metadata_path: str,
                 transform=None,
                 model_input_size: tuple = (112, 112, 112),
                 # --- Columns to read from CSV ---
                 filepath_col: str = "FilePath",
                 subject_id_col: str = "SubjectID",
                 # --- Conditions ---
                 condition_cols: list[str] | None = ["Age"],    # Columns to treat as conditions
                 # --- Filtering ---
                 filter_col: str | None = "Condition",          # Column to use for filtering
                 filter_value: str | None = "CN"                # Value to keep in filter_col (if filtering)
                ):
        """
        Initializes the dataset by reading the metadata CSV and preparing file lists and conditions.

        Args:
            metadata_path: Path to the metadata CSV file.
            transform: Optional MONAI/Torchvision transforms to apply to the image tensor before DWT.
            model_input_size: The spatial size (D, H, W) images will be padded/cropped to before DWT.
            filepath_col: Name of the column in the CSV containing the absolute path to the NIfTI file.
            subject_id_col: Name of the column containing the subject identifier.
            condition_cols: List of column names in the CSV to be used as conditions for the model.
                            Values will be normalized based on their range in the filtered dataset.
            filter_col: Name of the column to filter the dataset by (optional).
            filter_value: The value to keep in the `filter_col` (required if `filter_col` is set).
        """
        self.metadata_path = metadata_path
        self.transform = transform
        self.model_input_size = model_input_size
        self.condition_cols = condition_cols if condition_cols else []
        self.filepath_col = filepath_col
        self.subject_id_col = subject_id_col
        self.filter_col = filter_col
        self.filter_value = filter_value

        self.file_paths = []         # Stores full paths to valid NIfTI files found in CSV
        self.subject_ids = []        # Stores corresponding subject IDs
        self.conditions_raw = []     # Stores raw condition values extracted from CSV for each valid file
        self.condition_ranges = {}   # Stores {'min': X, 'max': Y} for numeric conditions

        logger.info(f"Loading metadata from: {self.metadata_path}")
        try:
            df = pd.read_csv(self.metadata_path)
            # Convert potential empty strings for numeric conditions (like Age) back to NaN
            for col in self.condition_cols:
                 if pd.api.types.is_numeric_dtype(df[col].dtype) or df[col].isnull().all(): # Check if potentially numeric or fully empty
                     # Attempt conversion, coercing errors (like empty strings) to NaN
                     df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(f"Loaded metadata with {len(df)} rows.")
        except FileNotFoundError:
            logger.error(f"Metadata CSV file not found at: {self.metadata_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading metadata CSV: {e}", exc_info=True)
            raise

        # --- Check required columns ---
        required_cols_in_csv = [self.filepath_col, self.subject_id_col] + self.condition_cols
        if self.filter_col:
            required_cols_in_csv.append(self.filter_col)
        required_cols_in_csv = list(set(required_cols_in_csv)) # Remove duplicates

        missing_cols = [col for col in required_cols_in_csv if col not in df.columns]
        if missing_cols:
             raise ValueError(f"Missing required columns in metadata CSV ({self.metadata_path}): {missing_cols}")

        # --- Filter DataFrame (Optional) ---
        if self.filter_col:
            if self.filter_value is None:
                logger.warning(f"Filter column '{self.filter_col}' provided, but no filter_value set. No filtering applied.")
            else:
                initial_rows = len(df)
                # Ensure comparison works correctly even if filter_value is numeric
                try:
                    # Attempt numeric comparison if possible
                    df_filter_val = pd.to_numeric(df[self.filter_col], errors='coerce')
                    filter_value_numeric = float(self.filter_value)
                    df = df[df_filter_val == filter_value_numeric].copy()
                except (ValueError, TypeError):
                    # Fallback to string comparison
                    df = df[df[self.filter_col].astype(str) == str(self.filter_value)].copy()

                logger.info(f"Filtered DataFrame by '{self.filter_col}' == '{self.filter_value}'. Kept {len(df)} out of {initial_rows} rows.")

        if len(df) == 0:
            raise RuntimeError("DataFrame is empty after filtering. No data to load.")

        # --- Process Filtered Rows ---
        logger.info(f"Validating file paths and extracting conditions from {len(df)} filtered rows...")
        temp_extracted_conditions = defaultdict(list)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating CSV data"):
            file_path = row[self.filepath_col]
            subject_id = str(row[self.subject_id_col]) # Ensure ID is string

            # Check if the file listed in the CSV actually exists
            if not isinstance(file_path, str) or not os.path.exists(file_path):
                logger.warning(f"File path not found or invalid in CSV row for Subject ID {subject_id}: '{file_path}'. Skipping.")
                continue

            # Extract conditions for this subject
            current_conditions = {}
            valid_conditions = True
            for cond_name in self.condition_cols:
                raw_value = row[cond_name]

                # Handle NaN/None values found in the CSV column
                if pd.isna(raw_value):
                     logger.warning(f"Condition '{cond_name}' is NaN/Null for Subject ID: {subject_id} in CSV. Skipping subject.")
                     valid_conditions = False
                     break

                # Store the raw value (could be float, int, string)
                current_conditions[cond_name] = raw_value

                # If it's numeric, add it for range calculation
                if isinstance(raw_value, (int, float, np.number)):
                    temp_extracted_conditions[cond_name].append(float(raw_value))
                # else: Handle categorical conditions later if needed

            if valid_conditions:
                self.file_paths.append(file_path)
                self.subject_ids.append(subject_id)
                self.conditions_raw.append(current_conditions)

        logger.info(f"Found {len(self.file_paths)} existing NIfTI files listed in CSV with valid conditions.")

        # --- Calculate Condition Ranges ---
        for cond_name, values_list in temp_extracted_conditions.items():
            if values_list:
                min_val = float(np.min(values_list))
                max_val = float(np.max(values_list))
                self.condition_ranges[cond_name] = {'min': min_val, 'max': max_val}
                logger.info(f"Calculated range for '{cond_name}': {min_val:.2f} to {max_val:.2f}")
            else:
                 # This might happen if the condition was categorical or always NaN/empty
                 logger.warning(f"No valid numeric values found for condition '{cond_name}' to calculate range.")

        if not self.file_paths:
             raise RuntimeError("Dataset empty after processing CSV and validating file paths. Check CSV content, file existence, and filters.")

        logger.info(f"Final dataset size: {len(self.file_paths)}")


    def __len__(self):
        return len(self.file_paths)

    def _robust_normalize(self, data):
        return robust_normalize(data)

    def _pad_data(self, data_tensor, target_size):
        return pad_to_size(data_tensor, target_size)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        raw_conditions_dict = self.conditions_raw[idx]

        try:
            # --- Load and Process Image ---
            img = nib.load(file_path)
            mri_data = img.get_fdata(dtype=np.float32)

            mri_data_normalized = self._robust_normalize(mri_data)
            if not np.isfinite(mri_data_normalized).all():
                logger.error(f"Non-finite values detected after normalization for {file_path}. Replacing with 0.")
                mri_data_normalized = np.nan_to_num(mri_data_normalized, nan=0.0, posinf=1.0, neginf=-1.0)

            tensor_data = torch.from_numpy(mri_data_normalized)

            # Pad data to the required model input size
            if tensor_data.shape != self.model_input_size:
                 tensor_data = self._pad_data(tensor_data, self.model_input_size)

            # Ensure tensor has a channel dimension (C=1)
            if tensor_data.ndim == 3:
                tensor_data = tensor_data.unsqueeze(0) # Shape: (1, D, H, W)

            # Apply transformations (e.g., augmentations)
            if self.transform:
                # MONAI transforms expect (C, D, H, W) input within Dataset __getitem__
                if tensor_data.ndim != 4:
                    # This should not happen if the above unsqueeze worked, but check just in case
                    raise ValueError(f"Unexpected tensor dimension before transform: {tensor_data.ndim}. Expected 4D (C,D,H,W). File: {file_path}")

                tensor_data = self.transform(tensor_data) # Apply transform directly to (C, D, H, W)
                tensor_data = tensor_data.float() # Ensure float

                if not torch.isfinite(tensor_data).all():
                    logger.error(f"Non-finite values detected after augmentation for {file_path}. Replacing with 0.")
                    tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)

            # Ensure tensor is 5D (N, C, D, H, W) for DWT
            if tensor_data.ndim == 4: # Should be (C, D, H, W) now
                tensor_data = tensor_data.unsqueeze(0) # Add N dim -> (1, C, D, H, W)
            elif tensor_data.ndim != 5:
                raise ValueError(f"Unexpected tensor dimension before DWT: {tensor_data.ndim} for {file_path}")

            if tensor_data.shape[1] != 1:
                 logger.warning(f"Expected 1 channel before DWT, got {tensor_data.shape[1]} for {file_path}. Ensure transforms maintain single channel.")

            if tensor_data.shape[-3:] != self.model_input_size:
                raise ValueError(f"Tensor shape {tensor_data.shape} mismatch before DWT, expected spatial {self.model_input_size} for {file_path}")

            if not torch.isfinite(tensor_data).all():
                logger.error(f"Non-finite values detected just before DWT for {file_path}. Replacing with 0.")
                tensor_data = torch.nan_to_num(tensor_data, nan=0.0, posinf=1.0, neginf=-1.0)

            # Apply DWT
            wavelet_coeffs_tuple = dwt_3d(tensor_data) # Input (N,C,D,H,W), Output tuple of (N,C,D/2,H/2,W/2)
            wavelet_coeffs_list = [wavelet_coeffs_tuple[0] / 1.0] + list(wavelet_coeffs_tuple[1:])
            for i, coeff in enumerate(wavelet_coeffs_list):
                if not torch.isfinite(coeff).all():
                     logger.error(f"Non-finite values detected in wavelet coeff {i} for {file_path}. Replacing with 0.")
                     wavelet_coeffs_list[i] = torch.nan_to_num(coeff, nan=0.0, posinf=1.0, neginf=-1.0)
            wavelet_coeffs_concat = torch.cat(wavelet_coeffs_list, dim=1) # Shape: (N, 8*C, D/2, H/2, W/2) -> (1, 8, D/2, H/2, W/2)

            # --- Process Conditions ---
            conditions_out = {}
            for cond_name in self.condition_cols: # Iterate through requested condition columns
                if cond_name not in raw_conditions_dict:
                     # This case should ideally not happen due to checks in __init__
                     logger.error(f"Condition '{cond_name}' requested but not found in raw data for index {idx}. Setting to default 0.0.")
                     conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)
                     continue

                raw_value = raw_conditions_dict[cond_name]

                if cond_name in self.condition_ranges:
                    # Normalize numeric conditions using pre-calculated ranges
                    min_v = self.condition_ranges[cond_name]['min']
                    max_v = self.condition_ranges[cond_name]['max']
                    range_v = max_v - min_v
                    try:
                        numeric_value = float(raw_value) # Ensure it's a float for normalization
                        if range_v > 1e-8:
                            norm_value = (numeric_value - min_v) / range_v
                        else:
                            norm_value = 0.5 # Default if range is zero
                        # Clip to ensure value is within [0, 1]
                        norm_value = np.clip(norm_value, 0.0, 1.0)
                        conditions_out[cond_name] = torch.tensor(norm_value, dtype=torch.float32)
                    except (ValueError, TypeError):
                        logger.error(f"Could not convert value '{raw_value}' for ranged condition '{cond_name}' to float. Setting to default 0.0.")
                        conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)

                else:
                    # Handle non-numeric or un-ranged conditions
                    # Example: Convert 'CN'/'MCI'/'AD' string to numeric encoding
                    if isinstance(raw_value, str):
                         # Example: Simple mapping for the "Condition" column generated previously
                         if cond_name == "Condition": # Using the default column name
                             condition_map = {"CN": 0.0, "MCI": 0.5, "AD": 1.0}
                             encoded_value = condition_map.get(raw_value, -1.0) # Default -1 if unknown
                             if encoded_value == -1.0:
                                 logger.warning(f"Unknown string value '{raw_value}' for condition '{cond_name}'. Using default 0.0.")
                                 encoded_value = 0.0
                             conditions_out[cond_name] = torch.tensor(encoded_value, dtype=torch.float32)
                         else:
                              logger.warning(f"Condition '{cond_name}' is string ('{raw_value}') but has no range and no specific handler. Using default 0.0.")
                              conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)
                    elif isinstance(raw_value, (int, float, np.number)):
                        logger.warning(f"Condition '{cond_name}' is numeric ('{raw_value}') but has no calculated range. Using default 0.0.")
                        conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)
                    else:
                         logger.warning(f"Condition '{cond_name}' has no range and is not string or standard numeric ('{raw_value}', type: {type(raw_value)}). Using default 0.0.")
                         conditions_out[cond_name] = torch.tensor(0.0, dtype=torch.float32)

            # Return wavelet coeffs (squeezing batch dim) and normalized/encoded conditions dict
            return wavelet_coeffs_concat.squeeze(0), conditions_out # Shape: (8, D/2, H/2, W/2), dict

        except Exception as e:
            logger.error(f"Error processing index {idx}, file: {file_path}: {e}", exc_info=True)
            # Return None to be filtered by collate_fn
            return None