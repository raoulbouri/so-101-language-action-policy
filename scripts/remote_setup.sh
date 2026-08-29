#!/usr/bin/env bash
# One-time environment setup on the remote GPU machine.
set -euo pipefail

PY=${PY:-python3}
echo "=== python ==="; $PY --version

if ! command -v uv >/dev/null 2>&1; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== creating venv ==="
uv venv --python 3.11 || uv venv

echo "=== installing (CUDA build of torch) ==="
# The default PyPI torch wheel on Linux already ships CUDA 12.x support, which
# matches the driver's CUDA 12.9. Pinning an index here avoids silently getting
# a CPU-only build.
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision
uv pip install -e ".[dev,act,tracking]"

echo
echo "=== verification ==="
.venv/bin/python - <<'PY'
import torch
print("torch     :", torch.__version__)
print("cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device    :", torch.cuda.get_device_name(0))
    print("n_gpus    :", torch.cuda.device_count())
    free, total = torch.cuda.mem_get_info()
    print(f"vram free : {free/1e9:.1f} / {total/1e9:.1f} GB")
else:
    print("!! CUDA NOT AVAILABLE -- training would fall back to CPU. Stop and fix this.")
import torchvision; print("torchvision:", torchvision.__version__)
try:
    import wandb; print("wandb     :", wandb.__version__)
except ImportError:
    print("wandb     : NOT INSTALLED")
PY

echo
echo "=== dataset check ==="
if [ -f data/train_1200.hdf5 ]; then
  ls -lh data/train_1200.hdf5
  .venv/bin/python scripts/verify_dataset.py data/train_1200.hdf5 --sample 20 2>&1 | tail -5
else
  echo "data/train_1200.hdf5 MISSING -- copy it with:"
  echo "  WITH_DATA=1 ./scripts/deploy_remote.sh    (from your laptop)"
fi
