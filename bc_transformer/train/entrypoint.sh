#!/usr/bin/env bash
set -euo pipefail

cd /workspace
exec micromamba run -n base python -u handler.py
