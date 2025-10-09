#!/bin/bash

# Check if the correct number of arguments are provided
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <config_path> <ckpt_path> <out_path> <classes_path> <dataset>"
    exit 1
fi

CONFIG_PATH=$1
CKPT_PATH=$2
OUT_PATH=$3
CLASSES_PATH=$4
DATASET=$5

if [[ $DATASET == "cifar10" ]]; then
    N_WAYS=(2 4 6 8)
elif [[ $DATASET == "cifar100" || $DATASET == "mini_imagenet" ]]; then
    N_WAYS=(4 6 8 10 50)
elif [[ $DATASET == "tiny_imagenet" ]]; then
    N_WAYS=(4 6 8 10 50 100)
elif [[ $DATASET == "full_imagenet" ]]; then
    N_WAYS=(5 10 50 100 500 1000)
else
    echo "Invalid dataset option"
    exit 1
fi

for n_way in "${N_WAYS[@]}"; do
    echo "---"
    echo "Starting training for dataset: ${DATASET}"
    echo "N-WAY set to: ${n_way}"

    python scripts/n_way_alignment_eval.py \
    --config $CONFIG_PATH \
    --ckpt_path $CKPT_PATH \
    --output_path $OUT_PATH \
    --classes_path $CLASSES_PATH \
    --n_way $n_way
done

echo "---"
echo "✅ Completed training for all n-way settings."