#!/bin/bash

# Check if the correct number of arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <config_path> <dataset_name>"
    exit 1
fi

CONFIG_PATH=$1
DATASET=$2

if [[ $DATASET == "cifar10" ]]; then
    N_WAYS=(2 4 6 8)
elif [[ $DATASET == "cifar100" || $DATASET == "mini_imagenet" ]]; then
    N_WAYS=(2 4 6 8 10 50)
elif [[ $DATASET == "tiny_imagenet" ]]; then
    N_WAYS=(2 4 6 8 10 50 100)
else
    echo "Invalid dataset option"
    exit 1
fi

for n_way in "${N_WAYS[@]}"; do
    echo "---"
    echo "Starting training for dataset: ${DATASET}"
    echo "N-WAY set to: ${n_way}"

    torchrun --nproc_per_node=1 --standalone \
    scripts/parallel_n_way_train.py \
    --config $CONFIG_PATH \
    --n_way $n_way
done

echo "---"
echo "✅ Completed training for all n-way settings."