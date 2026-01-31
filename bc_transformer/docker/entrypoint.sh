cat > bc_transformer/docker/entrypoint.sh <<'EOF'
#!/bin/sh
set -e

micromamba run -n appenv python handler.py
EOF

chmod +x bc_transformer/docker/entrypoint.sh
