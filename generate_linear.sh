#!/bin/bash

# Define variables for paths and parameters
CHECKPOINT_PATH="/path/to/checkpoint/fmw_best.pth"
OUTPUT_DIR="./output_slim_ablations_linear" 
CONDITION_RANGES_PATH="./configs/condition_ranges.json"
CONFIG_PATH="./configs/config.json" # Or from checkpoint dir if it contains config /path/to/checkpoint/config.json

MIN_AGE=5.9
MAX_AGE=95.46
NUM_TOTAL_SAMPLES=3000 # Keeping this low for a quick test example; set to 3000 for your full run
SAVE_SIZE_D=91
SAVE_SIZE_H=109
SAVE_SIZE_W=91
NUM_FLOW_STEPS=10 # Change to 1, 2, 5, 10, 200 for different ablations
MODEL_INPUT_D=112
MODEL_INPUT_H=112
MODEL_INPUT_W=112
FILENAME_PREFIX="FlowLetAblation"

mkdir -p ${OUTPUT_DIR}

# --- Execute the Python script ---
# The Python script itself will loop through the ablation modes (baseline, film_only, crossattn_only, unconditional) if enabled
# and create subdirectories within the specified OUTPUT_DIR.

PYTHONPATH=. nohup python3 -u scripts/generate_linear.py \
                            --checkpoint_path "${CHECKPOINT_PATH}" \
                            --output_dir "${OUTPUT_DIR}" \
                            --condition_ranges_path "${CONDITION_RANGES_PATH}" \
                            --config_path "${CONFIG_PATH}" \
                            --min_age ${MIN_AGE} \
                            --max_age ${MAX_AGE} \
                            --num_total_samples ${NUM_TOTAL_SAMPLES} \
                            --save_size ${SAVE_SIZE_D} ${SAVE_SIZE_H} ${SAVE_SIZE_W} \
                            --num_flow_steps ${NUM_FLOW_STEPS} \
                            --model_input_size ${MODEL_INPUT_D} ${MODEL_INPUT_H} ${MODEL_INPUT_W} \
                            --filename_prefix "${FILENAME_PREFIX}" \
                            --seed 42 \
                            --device cuda > "${OUTPUT_DIR}/generate_linear_ablations.log" 2>&1 &

echo "Linear age generation for ablation study started in background."
echo "Output will be in subdirectories within: ${OUTPUT_DIR}"
echo "Log file: ${OUTPUT_DIR}/generate_linear_ablations.log"