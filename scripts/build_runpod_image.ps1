# Build + push the pre-baked RunPod image (docker/Dockerfile.runpod), then point .env's RUNPOD_IMAGE at it.
#   docker login ghcr.io -u <github user>          (a token with write:packages)
#   .\scripts\build_runpod_image.ps1 [-Image ...]  default ghcr.io/tune-0-matic/tune-o-clip-pod:v1
# Needs Docker Desktop and ~40 GB free (the image is ~20 GB: CUDA + PyTorch base, Qwen 14B, Whisper large-v3-turbo).
param([string]$Image = "ghcr.io/tune-0-matic/tune-o-clip-pod:v1")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
# context = backend/: the build never sees .env, cookies.txt or workspace/
docker build --platform linux/amd64 -f docker/Dockerfile.runpod -t $Image backend
if ($LASTEXITCODE) { throw "docker build failed" }
docker push $Image
if ($LASTEXITCODE) { throw "docker push failed" }
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py backend/runpod_manager.py set-env RUNPOD_IMAGE $Image
if ($LASTEXITCODE) { throw "saving RUNPOD_IMAGE failed" }
Write-Host "Pushed $Image and saved it as RUNPOD_IMAGE in .env."
Write-Host "RunPod pulls without credentials only from a PUBLIC package: GitHub -> Packages -> tune-o-clip-pod -> Package settings -> Change visibility."
