#!/usr/bin/env bash

# Wrapped in braces so bash parses the whole script up front; switching
# branches while the job runs cannot change what is executed.
{
set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate jit
export TORCH_HOME=/home/ningmang/torch_cache

cd /home/ningmang/jit

# sampling
EXP_DIR=512_large_b2k
RESUME_EPOCHS=(600)
INTERVAL_MIN=0.1
CFG_VALUES=(2.3 2.4 2.5 2.6 2.7)
# CFG_VALUES=(3.2 3.3 3.4 3.5 3.6)
EMA_VALUES=(0.9999 0.9996)

OUTPUT_DIR="/home/ningmang/jit/JiT_exps/$EXP_DIR"
LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)_sample"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

for EPOCH in "${RESUME_EPOCHS[@]}"; do
    RESUME_PATH="$OUTPUT_DIR/checkpoint-${EPOCH}.pth"
    if [[ ! -f "$RESUME_PATH" ]]; then
        echo "Checkpoint not found: $RESUME_PATH" >&2
        exit 1
    fi
done

torchrun --nproc_per_node=8 --nnodes=1 \
--log-dir "$LOG_DIR/torchrun" --tee 3 \
main_jit.py --model JiT-L/32 \
--img_size 512 --noise_scale 2.0 --class_num 1000 \
--gen_bsz 256 --num_images 50000 \
--resume_epochs "${RESUME_EPOCHS[@]}" \
--cfg_values "${CFG_VALUES[@]}" \
--eval_ema_values "${EMA_VALUES[@]}" \
--interval_min "$INTERVAL_MIN" --interval_max 1.0 \
--output_dir "$OUTPUT_DIR" \
--data_path /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train --evaluate_gen

exit
}

# sslaunch submit -j sample -q stream -n 1 --no-log -- bash imagenet512_sample.sh
