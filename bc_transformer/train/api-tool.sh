#!/usr/bin/env bash
set -euo pipefail

echo "--- BC TRAIN WRAPPER ---"
echo "Args: $*"

# -----------------------------
# Pick local dirs (serverless-safe)
# -----------------------------
if [[ -d "/runpod-volume" ]]; then
  LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/runpod-volume/data}"
  LOCAL_RUNS_DIR="${LOCAL_RUNS_DIR:-/runpod-volume/runs}"
else
  LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/tmp/bc_data}"
  LOCAL_RUNS_DIR="${LOCAL_RUNS_DIR:-/tmp/bc_runs}"
fi
mkdir -p "$LOCAL_DATA_DIR" "$LOCAL_RUNS_DIR"

# -----------------------------
# Parse args
# -----------------------------
ARGS=("$@")

get_arg () {
  local flag="$1"
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "$flag" ]]; then
      echo "${ARGS[$((i+1))]}"
      return 0
    fi
  done
  echo ""
}

set_arg () {
  local flag="$1"
  local newval="$2"
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "$flag" ]]; then
      ARGS[$((i+1))]="$newval"
      return 0
    fi
  done
  # if not present, append
  ARGS+=("$flag" "$newval")
}

TRAIN_JSONL="$(get_arg --train_jsonl)"
VAL_JSONL="$(get_arg --val_jsonl)"
SAVE_PATH="$(get_arg --save_path)"

echo "Parsed --train_jsonl: ${TRAIN_JSONL:-<none>}"
echo "Parsed --val_jsonl:   ${VAL_JSONL:-<none>}"
echo "Parsed --save_path:   ${SAVE_PATH:-<none>}"
echo "Local data dir: $LOCAL_DATA_DIR"
echo "Local runs dir: $LOCAL_RUNS_DIR"

# -----------------------------
# Ensure save_path is writable
# -----------------------------
if [[ -n "$SAVE_PATH" ]]; then
  # If /runpod-volume isn't present but save_path points there, rewrite into /tmp runs
  if [[ "$SAVE_PATH" == /runpod-volume/* && ! -d "/runpod-volume" ]]; then
    # keep the tail path after /runpod-volume/runs/
    tail="${SAVE_PATH#/runpod-volume/runs/}"
    NEW_SAVE="$LOCAL_RUNS_DIR/$tail"
    echo "Rewriting save_path -> $NEW_SAVE"
    set_arg --save_path "$NEW_SAVE"
    SAVE_PATH="$NEW_SAVE"
  fi
  mkdir -p "$SAVE_PATH" || true
fi

# -----------------------------
# Download helpers
# -----------------------------
download_s3_file () {
  local src="$1"
  local dst="$2"
  echo "Downloading file $src -> $dst"
  aws s3 cp "$src" "$dst"
  ls -la "$dst"
}

download_s3_dir () {
  local src="$1"   # must end with /
  local dst="$2"   # directory
  echo "Downloading dir $src -> $dst (recursive)"
  mkdir -p "$dst"
  aws s3 cp "$src" "$dst" --recursive
  echo "Downloaded dir listing (top):"
  ls -la "$dst" | head -n 50 || true
}

# -----------------------------
# If train/val are S3, download and rewrite args
# -----------------------------
if [[ -n "$TRAIN_JSONL" && "$TRAIN_JSONL" == s3://* ]]; then
  if [[ "$TRAIN_JSONL" == */ ]]; then
    TRAIN_LOCAL="$LOCAL_DATA_DIR/train"
    download_s3_dir "$TRAIN_JSONL" "$TRAIN_LOCAL"
    set_arg --train_jsonl "$TRAIN_LOCAL"
  else
    TRAIN_LOCAL="$LOCAL_DATA_DIR/train.jsonl"
    download_s3_file "$TRAIN_JSONL" "$TRAIN_LOCAL"
    set_arg --train_jsonl "$TRAIN_LOCAL"
  fi
fi

if [[ -n "$VAL_JSONL" && "$VAL_JSONL" == s3://* ]]; then
  if [[ "$VAL_JSONL" == */ ]]; then
    VAL_LOCAL="$LOCAL_DATA_DIR/val"
    download_s3_dir "$VAL_JSONL" "$VAL_LOCAL"
    set_arg --val_jsonl "$VAL_LOCAL"
  else
    VAL_LOCAL="$LOCAL_DATA_DIR/val.jsonl"
    download_s3_file "$VAL_JSONL" "$VAL_LOCAL"
    set_arg --val_jsonl "$VAL_LOCAL"
  fi
fi

echo "Final args -> ${ARGS[*]}"

# quick sanity
echo "Sanity check local paths:"
ls -la "$LOCAL_DATA_DIR" || true
if [[ -n "$(get_arg --train_jsonl)" ]]; then
  echo "train_jsonl local -> $(get_arg --train_jsonl)"
fi
if [[ -n "$(get_arg --val_jsonl)" ]]; then
  echo "val_jsonl local -> $(get_arg --val_jsonl)"
fi

cd /workspace
exec micromamba run -n base python -u /workspace/train.py "${ARGS[@]}"
