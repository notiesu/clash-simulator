#!/usr/bin/env bash
set -euo pipefail

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --run_name) RUN_NAME="${2:?RUN_NAME missing}"; shift ;;
        --train_dir) TRAIN_DIR="${2:?TRAIN_DIR missing}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "---TRAIN API SH SCRIPT---"
echo "Passed in parameters:"
echo "RUN_NAME: ${RUN_NAME:-}"
echo "TRAIN_DIR: ${TRAIN_DIR:-}"


if [ -z "${RUN_NAME:-}" ]; then
    echo "Error: --run_name is required."
    exit 1
fi

MICROMAMBA_BIN="$(which micromamba)"


if [ ! -f "$MICROMAMBA_BIN" ]; then
    echo "Error: micromamba not found at $MICROMAMBA_BIN"
    exit 1
fi

TRAIN_PATH="/runpod-volume/cr-train-packages/$RUN_NAME"
OUT_PATH="/runpod-volume/cr-checkpts/$RUN_NAME"

if [ ! -d "$TRAIN_PATH" ]; then
    echo "Error: Directory $TRAIN_PATH not found."
    exit 1
fi

#copy train path to root
cp -r "$TRAIN_PATH" ./

mkdir -p output "$OUT_PATH"

if ! "$MICROMAMBA_BIN" run -n appenv \
    python -m "${RUN_NAME}.train" --output_dir output 2>&1 | tee output/train.log; then
    echo "Error: train.py encountered an error." >&2
    exit 1
fi

cp -r output "$OUT_PATH"
