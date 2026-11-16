#!/usr/bin/env bash
# Build + push the pre-baked RunPod image (docker/Dockerfile.runpod), then point .env's RUNPOD_IMAGE at it.
#   docker login ghcr.io -u <github user>      (a token with write:packages)
#   ./scripts/build_runpod_image.sh [image]    default ghcr.io/tune-0-matic/tune-o-clip-pod:v1
# Needs Docker and ~40 GB free (the image is ~20 GB: CUDA + PyTorch base, Qwen 14B, Whisper large-v3-turbo).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${1:-ghcr.io/tune-0-matic/tune-o-clip-pod:v1}"
# context = backend/: the build never sees .env, cookies.txt or workspace/
docker build --platform linux/amd64 -f docker/Dockerfile.runpod -t "$IMAGE" backend
docker push "$IMAGE"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
"$PY" backend/runpod_manager.py set-env RUNPOD_IMAGE "$IMAGE"
echo "Pushed $IMAGE and saved it as RUNPOD_IMAGE in .env."
echo "RunPod pulls without credentials only from a PUBLIC package: GitHub -> Packages -> tune-o-clip-pod -> Package settings -> Change visibility."
