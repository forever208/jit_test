#!/usr/bin/env bash

set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate jit

export TORCH_HOME=/home/ningmang/torch_cache

cd /home/ningmang/jit

# sampling
EXP_DIR=base_b2k
RESUME_EPOCHS=(400 440 480 520 560 600)
CFG_VALUES=(0)

OUTPUT_DIR="/home/ningmang/jit/JiT_exps/$EXP_DIR"
SUBMIT_LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)_sample"
mkdir -p "$SUBMIT_LOG_DIR"
exec > >(tee -a "$SUBMIT_LOG_DIR/stdout.log") 2> >(tee -a "$SUBMIT_LOG_DIR/stderr.log" >&2)
echo "stdout log: $SUBMIT_LOG_DIR/stdout.log"
echo "stderr log: $SUBMIT_LOG_DIR/stderr.log"

for CFG in "${CFG_VALUES[@]}"; do
    for EPOCH in "${RESUME_EPOCHS[@]}"; do
        RESUME_PATH="$OUTPUT_DIR/checkpoint-${EPOCH}.pth"
        LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)_checkpoint-${EPOCH}_cfg-${CFG}"

        if [[ ! -f "$RESUME_PATH" ]]; then
            echo "Checkpoint not found: $RESUME_PATH" >&2
            exit 1
        fi

        torchrun --standalone --nproc_per_node=8 --nnodes=1 \
        --log-dir "$LOG_DIR/torchrun" --tee 3 \
        main_jit.py --model JiT-B/16 \
        --img_size 256 --noise_scale 1.0 --class_num 1000 \
        --gen_bsz 626 --num_images 50000 --cfg "$CFG" --interval_min 0.1 --interval_max 1.0 \
        --output_dir "$OUTPUT_DIR" \
        --resume "$RESUME_PATH" \
        --data_path /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train --evaluate_gen
    done
done


# sslaunch submit -j sample -c zhongwei -q stream -n 1 --no-log -- bash imagenet256_sample.sh