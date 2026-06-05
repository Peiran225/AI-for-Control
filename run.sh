#!/usr/bin/env bash
set -euo pipefail

for lr in 1e-4 5e-4 1e-3 5e-3 1e-2; do
  echo "Running with lr=$lr"
  conda run -n EMB python train.py --lr "$lr" 
done
