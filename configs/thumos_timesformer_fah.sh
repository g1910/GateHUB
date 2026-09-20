#!/usr/bin/env bash
# THUMOS'14, TimeSformer RGB + TSN optical flow with Future-augmented History
# (Table 1, 72.5 mAP; Table 3c, t_f = 2 s).
#
# dim_feature = 2816 = 768 (TimeSformer cls token) + 2048 (TSN flow).
# History frames older than the present window are encoded from their observed
# future via --future_file (Eqn. 5); the rest use the past-window features.
set -euo pipefail

DATA_ROOT=${1:?usage: thumos_timesformer_fah.sh <feature_dir> [output_dir]}
OUTPUT_DIR=${2:-logs/thumos_timesformer_fah}

python main.py \
  --dataset THUMOS \
  --data_root "${DATA_ROOT}" \
  --tfk \
  --feature_file 'thumos_all_feature_{}_tsn_v2.pickle' \
  --rgb_feature_file thumos_timesformer_k600_96_1s_temp_end.pkl \
  --future_file thumos_timesformer_k600_96_2s_cls.pkl \
  --feature_stride 1 \
  --ltm_interval 1 \
  --train_stride 12 \
  --present_frames 8 \
  --history_frames 1024 \
  --latent_size 16 \
  --history_layers 2 \
  --decoder_layers 2 \
  --decoder_num_heads 16 \
  --embedding_dim 1024 \
  --decoder_embedding_dim_out 1024 \
  --dim_feature 2816 \
  --dropout_rate 0.5 \
  --decoder_attn_dropout_rate 0 \
  --positional_encoding_type learned \
  --dec_init zero \
  --batch_size 50 \
  --epochs 10 \
  --lr_scheduler OneCycle \
  --max_lr 1e-4 \
  --pct_start 0.3 \
  --weight_decay 5e-5 \
  --focal_alpha 0.5 \
  --focal_gamma 0.5 \
  --focal_ac_gamma 0.5 \
  --focal_bg_gamma 1 \
  --num_workers 2 \
  --output_dir "${OUTPUT_DIR}"
