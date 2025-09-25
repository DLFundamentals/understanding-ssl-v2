#!/bin/bash

CONFIG_PATH=$1
BATCH_SIZES=(256 512 896)
LR=("sqrt" "sqrt_4" "constant")

for lr in "${LR[@]}"; do
    echo "LR schedule: ${lr}"
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo "---"
        echo "Starting training with batch size: ${batch_size}"

        torchrun --nproc_per_node=1 --standalone \
        scripts/parallel_batch_size_train.py \
        --config $CONFIG_PATH \
        --batch_size $batch_size \
        --lr_order "$lr"
    done
    echo "---"
    echo "✅ Completed training for all batch size settings with LR schedule: ${lr}."
done

echo "✅ Completed training for all batch size and LR schedule settings."