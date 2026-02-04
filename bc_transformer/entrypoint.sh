#!/usr/bin/env bash
set -euo pipefail

## put s3 bucket dir & output dir

cd /workspace
exec python -u handler.py
