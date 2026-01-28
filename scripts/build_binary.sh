#!/usr/bin/env bash
set -euo pipefail

# Build a single-file Linux executable using PyInstaller.
# Run inside a Linux environment similar to the target deployment environment.

pyinstaller --clean --onefile --name stegano-anomaly -m stegano_anomaly.cli
echo "Built: dist/stegano-anomaly"
