#!/usr/bin/env bash
set -euxo pipefail

# --- System deps
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
build-essential git tmux htop nvtop jq unzip zip pigz aria2 \
libaio1 tzdata cmake ninja-build python3-venv \
libssl-dev libffi-dev

# --- Project root
mkdir -p /workspace/orph && cd /workspace/orph

# --- Python venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# --- Torch for CUDA 12.x (H100)
pip install --index-url https://download.pytorch.org/whl/cu124 \
torch torchvision torchaudio

# --- Core libs
cat > requirements.txt << 'REQ'
transformers>=4.44.0
accelerate>=0.34.0
tokenizers>=0.15.2
datasets>=2.20.0
sentencepiece>=0.2.0
fastapi>=0.115 uvicorn[standard]>=0.30
httpx[socks]>=0.27.0
pydantic>=2.8.0
pyyaml>=6.0.1
orjson>=3.10.0
rich>=13.7.1
tqdm>=4.66.4
psutil>=5.9.8
numpy>=1.26.4
pandas>=2.2.2
scikit-learn>=1.5.0
xxhash>=3.4.1
zstandard>=0.23.0
REQ
pip install -r requirements.txt

# --- Project skeleton
mkdir -p conf data/raw data/cleaned data/corpus data/tokenized out/text_orphgpt out/tokenizer src scripts

# --- Env defaults (edit as needed)
cat > .env << 'ENV'
export HF_HOME=/workspace/.hf
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/orph
ENV

# --- Tiny GPU sanity
cat > scripts/verify_gpu.py << 'PY'
import torch, os
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
print('Device:', torch.cuda.get_device_name(0))
a=torch.randn(8192,8192, device='cuda'); b=torch.randn(8192,8192, device='cuda')
torch.cuda.synchronize();
c=(a@b).sum(); torch.cuda.synchronize();
print('Matmul ok, sum=', float(c))
PY

python scripts/verify_gpu.py
