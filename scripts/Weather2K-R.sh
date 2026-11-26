#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
dataset="Weather2K-R"
python -m storage.store_emb --data_path "$dataset" "$@"
case "$dataset" in
  Weather) shape=96_192 ;;
  Weather2K-R) shape=8_8 ;;
  ERA5) shape=48_24 ;;
esac
for seed in 42 1234 2024 2025 2026 9999; do
  python train.py --data_path "$dataset" --seed "$seed" --mode baseline "$@"
  python train.py --data_path "$dataset" --seed "$seed" --mode peak --baseline_checkpoint "Results/PeakCMA/$dataset/$shape/$seed/baseline/best_model.pth" "$@"
done
