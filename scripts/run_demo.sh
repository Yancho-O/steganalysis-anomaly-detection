#!/usr/bin/env bash
set -euo pipefail

# Demo workflow:
# 1) Generate toy dataset (clean vs synthetic LSB stego)
# 2) Extract features
# 3) Train + evaluate a baseline (ROC-AUC and PR)

python -m scripts.make_toy_dataset --out data/toy --n 800 --size 256 --stego_rate 0.5 --embed_rate 0.15
stegano-anomaly extract data/toy --out artifacts/features.csv --label-from-parent
stegano-anomaly train artifacts/features.csv --model iforest --out artifacts/model.joblib --out-report artifacts/report.json --out-plots-dir artifacts/plots
cat artifacts/report.json
