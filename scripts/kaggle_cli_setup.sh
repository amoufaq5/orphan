#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/.kaggle /workspace/.kaggle
# copy your kaggle.json into one of these paths, then:
chmod 600 /root/.kaggle/kaggle.json || true
chmod 600 /workspace/.kaggle/kaggle.json || true
kaggle --version
