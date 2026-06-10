#!/usr/bin/env bash
set -euo pipefail

# Example beta=0.1 runs. Adjust --device, batch size, and paths for your machine.
cd "$(dirname "$0")"
RUN_DIR="${RUN_DIR:-../runs/beta01}"

# Transformer open-loop baseline: u_theta(t), time-only.
python train_openloop.py \
  --model transformer --n 200 --epochs 1200 \
  --gamma 20 --alpha 1 --beta 0.1 \
  --d_model 32 --heads 4 --layers 1 \
  --out_dir "$RUN_DIR/openloop"

python train_feedback.py \
  --model transformer --n 200 --epochs 3000 --batch_size 512 \
  --gamma 20 --alpha 1 --beta 0.1 \
  --d_model 64 --heads 4 --layers 2 \
  --train_low 0.25 --train_high 2.0 --trait_noise 0.05 \
  --eval_every 100 \
  --out_dir "$RUN_DIR/feedback"

python refine_feedback.py \
  --init_feedback "$RUN_DIR/feedback/best_feedback_pmp_kkt.pt" \
  --model transformer --n 200 --epochs 2500 --batch_size 512 \
  --gamma 20 --alpha 1 --beta 0.1 \
  --d_model 64 --heads 4 --layers 2 \
  --lr 1e-4 \
  --train_low 0.25 --train_high 2.0 --trait_noise 0.05 \
  --eval_every 100 \
  --out_dir "$RUN_DIR/refined_feedback"
