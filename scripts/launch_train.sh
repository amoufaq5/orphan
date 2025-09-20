#!/usr/bin/env bash
set -euo pipefail
source /workspace/.venv/bin/activate
cd /workspace/orph
export TRAIN_YAML=conf/train_text.yaml
python -m src.models.textlm.train_text | tee -a out/train.log
