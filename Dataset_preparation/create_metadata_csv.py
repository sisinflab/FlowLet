# scripts/create_metadata_csv.py

import os
import glob
import re
import csv
import argparse
import logging
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Regex for AGE extraction
AGE_PATTERN = re.compile(r'[_-]AGE[_-]([0-9.]+)', re.IGNORECASE)

def extract_age_from_filename(filename):
    """Extracts age from a filename using the predefined AGE_PATTERN."""
    match = AGE_PATTERN.search(os.path.basename(filename))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert extracted AGE '{match.group(1)}' to float in filename: {filename}")
            return None
    return None

def extract_subject_id_from_filename(filename_no_ext, custom_regex=None):
    """
    Extracts a subject ID from the filename (without extension).
    Uses a custom regex if provided, otherwise defaults to the filename itself.
    """
    if custom_regex:
        match = custom_regex.search(filename_no_ext)
        if match:
            # If the regex has groups, try to use the first captured group
            # Otherwise, use the full match
            return match.group(1) if match.groups() else match.group(0)
    # Default: use the filename without extension as subject ID
    return filename_no_ext

def main():
    parser = argparse.ArgumentParser(description="Create a CSV metadata file from NIfTI files in specified directories.")
    parser.add_argument(
        "--input_dirs",
        nargs='+',
        required=True,
        help="List of input directories containing .nii.gz files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--subject_id_regex",
        type=str,
        default=None,
        help="Optional regex to extract subject ID from filenames (e.g., '^(sub-\\d+)_'). "
             "The first captured group will be used as the ID. "
             "If not provided, the filename (without .nii.gz) is used."
    )
    parser.add_argument(
        "--condition_label",
        type=str,
        default="CN",
        help="Static label to assign to the 'Condition' column for all entries."
    )

    args = parser.parse_args()

    # Compile custom regex if provided
    subject_id_extractor_regex = None
    if args.subject_id_regex:
        try:
            subject_id_extractor_regex = re.compile(args.subject_id_regex)
            logger.info(f"Using custom regex for subject ID: {args.subject_id_regex}")
        except re.error as e:
            logger.error(f"Invalid regex provided for subject_id_regex: {e}. Will use default filename based ID.")
            subject_id_extractor_regex = None


    metadata_records = []
    files_processed = 0
    files_with_age = 0
    files_missing_age = 0

    logger.info(f"Scanning directories: {args.input_dirs}")

    for input_dir in args.input_dirs:
        if not os.path.isdir(input_dir):
            logger.warning(f"Input directory not found or not a directory: {input_dir}. Skipping.")
            continue

        logger.info(f"Processing directory: {input_dir}")
        # Recursively find all .nii.gz files
        nifti_files = glob.glob(os.path.join(input_dir, "**", "*.nii.gz"), recursive=True)

        if not nifti_files:
            logger.info(f"No .nii.gz files found in {input_dir} (and its subdirectories).")
            continue

        for file_path in tqdm(nifti_files, desc=f"Processing {os.path.basename(input_dir)}", unit="file"):
            files_processed += 1
            filename_with_ext = os.path.basename(file_path)
            filename_no_ext = filename_with_ext.replace(".nii.gz", "") # Simple removal

            subject_id = extract_subject_id_from_filename(filename_no_ext, subject_id_extractor_regex)
            age = extract_age_from_filename(filename_with_ext)

            if age is not None:
                files_with_age += 1
            else:
                files_missing_age += 1
                logger.debug(f"Age not found in filename: {filename_with_ext}")

            metadata_records.append({
                "SubjectID": subject_id,
                "FilePath": os.path.abspath(file_path), # Store absolute path
                "Age": age if age is not None else "",   # Store as empty string if None for CSV
                "Condition": args.condition_label
            })

    logger.info(f"Total files scanned: {files_processed}")
    logger.info(f"Files with AGE successfully extracted: {files_with_age}")
    logger.info(f"Files missing AGE tag or unparsable: {files_missing_age}")

    if not metadata_records:
        logger.warning("No metadata records generated. Output CSV will be empty or not created.")
        return

    # Write to CSV
    fieldnames = ["SubjectID", "FilePath", "Age", "Condition"]
    try:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir: # Ensure output directory exists if specified in path
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_records)
        logger.info(f"Successfully wrote {len(metadata_records)} records to {args.output_csv}")
    except IOError as e:
        logger.error(f"Could not write CSV file to {args.output_csv}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing the CSV: {e}", exc_info=True)

if __name__ == "__main__":
    main()