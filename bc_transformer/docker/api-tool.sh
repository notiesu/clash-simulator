cat > bc_transformer/docker/api-train.sh <<'EOF'
#!/usr/bin/env bash
set -eo pipefail

echo "--- BC TRAIN WRAPPER ---"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "Error: micromamba not found" >&2
  exit 1
fi

MICROMAMBA_BIN="$(command -v micromamba)"

# We will standardize these in the next step
OUT_DIR="${OUT_DIR:-output}"

mkdir -p "$OUT_DIR"

"$MICROMAMBA_BIN" run -n appenv \
  python -u train.py "$@" \
  2>&1 | tee "$OUT_DIR/train.log"
EOF

chmod +x bc_transformer/docker/api-train.sh
