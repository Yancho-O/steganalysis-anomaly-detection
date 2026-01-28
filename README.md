# Steganographic Analysis (Anomaly Detection)

A reproducible, Linux-friendly CLI project for **steganalysis via anomaly detection / classification baselines**.

It provides:
- Engineered (classical) image features intended to capture subtle perturbations consistent with simple steganographic embedding.
- Baseline anomaly detectors and supervised classifiers.
- Evaluation with **ROC-AUC** and **Precision–Recall** (Average Precision), plus curve plots.
- A fully scriptable CLI workflow, suitable for Git-based versioning and CI.

## Quickstart (Linux)

### 1) Create environment and install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Run the end-to-end demo (toy dataset)
```bash
pip install -e .
python -m scripts.make_toy_dataset --out data/toy --n 800 --size 256 --stego_rate 0.5 --embed_rate 0.15
stegano-anomaly extract data/toy --out artifacts/features.csv --label-from-parent
stegano-anomaly train artifacts/features.csv --model iforest --out artifacts/model.joblib
```

Outputs:
- `artifacts/report.json` with ROC-AUC and Average Precision
- `artifacts/plots/roc.png` and `artifacts/plots/pr.png`
- `artifacts/model.joblib` (model + feature schema)

## CLI

### List baselines
```bash
stegano-anomaly models
```

### Extract features
```bash
stegano-anomaly extract /path/to/images \
  --out artifacts/features.csv \
  --labels-csv labels.csv              # optional (filename,label) \
  --label-from-parent                  # optional (clean/ vs stego/ folders) \
  --resize 256                         # stable features across sizes (0 disables) \
  --color                              # default grayscale
```

### Train (+ evaluate if labels exist)
```bash
stegano-anomaly train artifacts/features.csv --model iforest
stegano-anomaly train artifacts/features.csv --model logreg   # supervised requires labels
```

### Predict (score)
```bash
stegano-anomaly predict artifacts/model.joblib artifacts/features.csv --out artifacts/scores.csv
```

## Engineered features (overview)

The feature vector is intentionally lightweight and interpretable:
- Pixel intensity moments (mean/variance/skew/kurtosis)
- Intensity histogram (default 64 bins)
- Adjacent-pixel difference moments + histogram (captures LSB/local perturbations)
- High-pass residual moments (noise/residual artifacts)
- Blockwise DCT magnitude histogram (frequency-domain artifacts)

These are suitable for classical baselines (Isolation Forest, One-Class SVM, LOF) and simple supervised models.

## Reproducible workflow

- All commands are deterministic given `--seed` (where applicable) and pinned configs.
- Outputs are written into `artifacts/` by default.
- CI runs unit tests and linting.

## Build a standalone Linux executable (optional)

This project is packaged as a normal Python CLI. To produce a single-file executable on Linux:
```bash
pip install pyinstaller
bash scripts/build_binary.sh
./dist/stegano-anomaly --help
```

## License
MIT. See `LICENSE`.
