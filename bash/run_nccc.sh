#!/bin/bash
CONFIG_PATH=$1
CKPT_PATH=$2
OUTPUT_PATH=$3

for supervision in "dcl" "nscl" "scl" "ce"; do
    echo "Evaluating with supervision type: $supervision"
    python scripts/nccc_eval.py \
        --config $CONFIG_PATH \
        --ckpt_path $CKPT_PATH \
        --output_path $OUTPUT_PATH \
        --supervision $supervision
done
echo "Evaluation completed for all supervision types."