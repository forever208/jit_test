#!/usr/bin/env bash

set -euo pipefail

source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate jit
export TORCH_HOME=/home/ningmang/torch_cache

# log and error settings
OUTPUT_DIR=/home/ningmang/jit/JiT_exps/base_huge_b2k
LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"
echo "torchrun log dir: $LOG_DIR/torchrun"

cd /home/ningmang/jit

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
--log-dir "$LOG_DIR/torchrun" --tee 3 \
main_jit.py --model JiT-H/16 \
--proj_dropout 0.2 --P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 --class_num 1000 \
--batch_size 256 --blr 5e-5 --epochs 601 --warmup_epochs 5 --num_workers 16 \
--gen_bsz 256 --num_images 50000 --eval_freq 100 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--data_path /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train --online_eval \
--output_dir "$OUTPUT_DIR" \
--resume /home/ningmang/jit/JiT_exps/base_huge_b2k/checkpoint-200.pth


# --resume /home/ningmang/jit/JiT_exps/imgnet256_base_p16/checkpoint-official.pth \

# sslaunch submit -j cankill -c zhongwei -q stream -n 1 --no-log -- bash imagenet256_train_huge.sh
# sslaunch submit -j huge -p low -q stream -n 1 --no-log -- bash imagenet256_train_huge.sh
