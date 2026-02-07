#!/usr/bin/env bash
set -euo pipefail

############################################
# Load .env if present
############################################
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

############################################
# USER CONFIG
############################################

# Allow passing run name as arg 1 (optional)
RUN_NAME="${1:-${RUN_NAME:-run_001}}"

# cpu | cuda
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"

# S3 folder that already contains train.jsonl + val.jsonl
# Allow override from environment
S3_URI="${S3_URI:-s3://clash-replays-irl-data/training_data_1/}"

############################################
# DERIVED PATHS
############################################
S3_TRAIN_URI="${S3_URI%/}/"
S3_VAL_URI="${S3_URI%/}/"

# Where training should save outputs inside the container
# (Your wrapper can create this dir. Serverless may not mount /runpod-volume.)
# Local save path inside container (serverless-safe)
SAVE_PATH="${SAVE_PATH:-/tmp/runs/${RUN_NAME}}"

# S3 destination for weights/artifacts
S3_SAVE_URI="${S3_SAVE_URI:-${S3_URI%/}/runs/${RUN_NAME}/}"

############################################
# SAFETY CHECKS
############################################
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  echo "ERROR: RUNPOD_API_KEY is not set"
  exit 1
fi

if [[ -z "${RUNPOD_ENDPOINT_ID:-}" ]]; then
  echo "ERROR: RUNPOD_ENDPOINT_ID is not set"
  exit 1
fi

############################################
# SUBMIT RUNPOD JOB
############################################
echo "Submitting RunPod job..."
echo "  S3 train: ${S3_TRAIN_URI}"
echo "  S3 val:   ${S3_VAL_URI}"
echo "  Run name: ${RUN_NAME}"
echo "  Device:   ${DEVICE}"
echo "  Save:     ${SAVE_PATH}"

RESPONSE=$(curl -sS -w "\nHTTP_CODE:%{http_code}\n" \
  -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "train_jsonl": "'"${S3_TRAIN_URI}"'",
      "val_jsonl": "'"${S3_VAL_URI}"'",
      "epochs": '"${EPOCHS}"',
      "device": "'"${DEVICE}"'",
      "save_path": "'"${SAVE_PATH}"'",
      "s3_save_uri": "'"${S3_SAVE_URI}"'"
    }
  }'
)

echo "$RESPONSE"
echo "RunPod job submitted."
