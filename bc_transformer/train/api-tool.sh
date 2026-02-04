#!/usr/bin/env bash
set -euo pipefail

echo "--- BC TRAIN WRAPPER ---"

OUT_DIR="${OUT_DIR:-output}"
mkdir -p "$OUT_DIR"

cd /workspace
exec python -u train.py "$@" 2>&1 | tee "$OUT_DIR/train.log"
