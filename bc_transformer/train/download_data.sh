#!/usr/bin/env bash
set -euo pipefail

S3_URI="${S3_URI:-}"
DEST_DIR="${DEST_DIR:-/runpod-volume/data}"

if [ -z "$S3_URI" ]; then
  echo "ERROR: S3_URI is not set. Example: s3://clash-replays-irl-data/training_data_1/" >&2
  exit 1
fi

echo "=== Downloading dataset ==="
echo "S3_URI   = $S3_URI"
echo "DEST_DIR = $DEST_DIR"

mkdir -p "$DEST_DIR"

# Sync from S3 to the volume. Safe to re-run; only downloads diffs.
aws s3 sync "$S3_URI" "$DEST_DIR"

echo "=== Download complete ==="
