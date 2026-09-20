#!/usr/bin/env bash
# THUMOS'14, TSN RGB + TSN optical flow features (Table 1, 70.7 mAP).
#
# Paper settings (Sec. 4.2): history T = 1024 frames @ 4 FPS (256 s),
# present t_pr = 8 frames @ 4 FPS (2 s), D = 1024, L = 16, N = 2, N_heads = 16.
#
# gamma_a = focal_ac_gamma * focal_gamma = 0.05
# gamma_b = focal_bg_gamma * focal_gamma = 0.10
# These are the values used by the original training runs. Note the paper quotes
# gamma_a = 0.6 / gamma_b = 0.2 in Sec. 4.2 and gamma_a = 0.05 / gamma_b = 0.025
# in Sec. 4.4; see README for details.
set -euo pipefail

DATA_ROOT=${1:?usage: thumos_tsn.sh <feature_dir> [output_dir]}
OUTPUT_DIR=${2:-logs/thumos_tsn}

python main.py \
  --dataset THUMOS \
  --data_root "${DATA_ROOT}" \
  --feature_file 'thumos_all_feature_{}_tsn_v2.pickle' \
  --feature_stride 1 \
  --ltm_interval 1 \
  --train_stride 1 \
  --present_frames 8 \
  --history_frames 1024 \
  --latent_size 16 \
  --history_layers 2 \
  --decoder_layers 2 \
  --decoder_num_heads 16 \
  --embedding_dim 1024 \
  --decoder_embedding_dim_out 1024 \
  --dim_feature 4096 \
  --dropout_rate 0.5 \
  --decoder_attn_dropout_rate 0 \
  --positional_encoding_type learned \
  --dec_init zero \
  --batch_size 50 \
  --epochs 10 \
  --lr_scheduler OneCycle \
  --max_lr 1.2e-4 \
  --pct_start 0.25 \
  --weight_decay 5e-5 \
  --focal_alpha 0.75 \
  --focal_gamma 0.1 \
  --focal_ac_gamma 0.5 \
  --focal_bg_gamma 1 \
  --num_workers 2 \
  --output_dir "${OUTPUT_DIR}"
