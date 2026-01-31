Build on a machine with Docker (e.g., RunPod):

# From the directory ABOVE bc_transformer:
cd bc_transformer

# Build base image (uses environment.yml + requirements.txt in this folder)
docker build -f docker/Dockerfile.base -t bc-transformer-base:0.1 .

# Build handler image (copies code + handler)
docker build -f docker/Dockerfile.handler -t bc-transformer-handler:0.1 .
