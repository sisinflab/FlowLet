#!/bin/bash
# Build the metadata catalog CSV from one or more directories of preprocessed NIfTI files.
# Run from the repository root. Replace /path/to/dataset/ with your data directory.

PYTHONPATH=. python3 Dataset_preparation/create_metadata_csv.py \
    --input_dirs /path/to/dataset/ \
    --output_csv ./Dataset_preparation/metadata/main_dataset_catalog.csv \
    --condition_label CN
