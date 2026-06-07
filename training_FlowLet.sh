#!/bin/bash
# Adjust --metadata_csv to point at your prepared dataset catalog.

mkdir -p logs

PYTHONPATH=. nohup python3 -u scripts/train.py \
                            --metadata_csv ./Dataset_preparation/metadata/main_dataset_catalog.csv \
                            --csv_filter_col Condition \
                            --csv_filter_value CN \
                            --lr 3e-6 \
                            --batch_size 4 \
                            --early_stop_patience 200 \
                            --epochs 200 \
                            --save_size 91 109 91 \
                            --model_input_size 112 112 112 \
                            --unet_model_channels 128 \
                            --unet_num_heads 8 \
                            --unet_dropout 0 \
                            --unet_num_res_blocks 2 \
                            --unet_channel_mult "1, 2, 4, 8" \
                            --unet_attention_res "4, 8" \
                            --flow_type rectified \
                            --use_xformers \
                            --use_checkpointing \
                            --run_name "FlowLet_RFM" \
                            --lll_loss_weight 1 \
                            --detail_loss_weight 1 \
                            --wandb > logs/FlowLet_RFM_training.log 2>&1 &
