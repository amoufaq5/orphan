#!/usr/bin/env bash
set -euo pipefail
cd /workspace

# 1) System basics
apt-get update -y && apt-get install -y git-lfs build-essential unzip pv jq

# 2) Create project dirs (mount a RunPod Volume at /workspace/volume first)
mkdir -p /workspace/volume/{data,out,cache}
ln -sfn /workspace/volume/data /workspace/orph/data
ln -sfn /workspace/volume/out /workspace/orph/out

# 3) Python venv
python3 -m venv /workspace/.venv
source /workspace/.venv/bin/activate
python -m pip install -U pip wheel

# 4) Torch + deps (assumes CUDA 12.x image)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio || true

# 5) Project deps
cd /workspace/orph
pip install -r requirements.txt

# 6) Optional: faster tokenizers build flags
export TOKENIZERS_PARALLELISM=true

# 7) Cache dirs
export HF_HOME=/workspace/volume/cache/hf
export HF_DATASETS_CACHE=/workspace/volume/cache/hf_datasets
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# 8) Verify GPU
python - << 'PY'
import torch, os
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
print("device=", torch.cuda.get_device_name(0))
a=torch.rand((4096,4096), device='cuda'); b=torch.mm(a,a.T); print("mm_ok=", b.sum().item())
PY

# 9) Done
echo "Init complete. Data is persisted under /workspace/volume."
