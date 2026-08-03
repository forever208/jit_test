#!/usr/bin/env bash

set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate jit


# sampling
EXP_DIRS=(
    imgnet256_base_b2k_noctx
)
export TORCH_HOME=/home/ningmang/torch_cache

RESUME_EPOCHS=(80 120 160 200)

for EXP_DIR in "${EXP_DIRS[@]}"; do
    OUTPUT_DIR="/home/ningmang/jit/JiT_exps/$EXP_DIR"

    for EPOCH in "${RESUME_EPOCHS[@]}"; do
        RESUME_PATH="$OUTPUT_DIR/checkpoint-${EPOCH}.pth"
        LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)_checkpoint-${EPOCH}"

        if [[ ! -f "$RESUME_PATH" ]]; then
            echo "Checkpoint not found: $RESUME_PATH" >&2
            exit 1
        fi

        torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
        --log-dir "$LOG_DIR/torchrun" --tee 3 \
        main_jit.py --model JiT-B/16 \
        --img_size 256 --noise_scale 1.0 --class_num 1000 \
        --gen_bsz 626 --num_images 50000 --cfg 0 --interval_min 0.12 --interval_max 1.0 \
        --output_dir "$OUTPUT_DIR" \
        --resume "$RESUME_PATH" \
        --data_path /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train --evaluate_gen
    done
done


# bash imagenet256_sample.sh