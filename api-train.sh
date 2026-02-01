#!/usr/bin/env bash
set -eo pipefail

RUN_NAME=""
TRAIN_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --train_dir)
            TRAIN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            exit 1
            ;;
    esac
done

echo "--- TRAIN API SH SCRIPT ---"
echo "RUN_NAME: $RUN_NAME"
echo "TRAIN_DIR: $TRAIN_DIR"

if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is required" >&2
    exit 1
fi

if ! command -v micromamba >/dev/null 2>&1; then
    echo "Error: micromamba not found in PATH" >&2
    exit 1
fi

MICROMAMBA_BIN="$(command -v micromamba)"

TRAIN_PATH="/runpod-volume/cr-train-packages/$RUN_NAME"
OUT_PATH="/runpod-volume/cr-checkpts/$RUN_NAME"

if [[ ! -d "$TRAIN_PATH" ]]; then
    echo "Error: training directory not found: $TRAIN_PATH" >&2
    exit 1
fi

cp -r "$TRAIN_PATH" "./train"

mkdir -p output "$OUT_PATH"

echo "Starting training..."

"$MICROMAMBA_BIN" run -n appenv \
    python -u -m "train.train" --output_dir output "$@" \
    2>&1 | tee output/train.log

STATUS=${PIPESTATUS[0]}

if [[ $STATUS -ne 0 ]]; then
    echo "Error: training failed (exit code $STATUS)" >&2
    exit $STATUS
fi

cp -r output "$OUT_PATH"

#remove the training dir
rm -rf "./train"


echo "Training completed successfully."
echo "$OUT_PATH contents:"
ls -la "$OUT_PATH/output"