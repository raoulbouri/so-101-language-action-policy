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

echo "=== detecting the driver's maximum CUDA version ==="
# A torch wheel built for a NEWER CUDA than the driver supports will import
# fine and then report cuda avail: False. That is the single most common way to
# end up silently training on CPU, so derive the wheel tag from the driver
# rather than assuming one.
CUDA_TAG=${CUDA_TAG:-}
if [ -z "$CUDA_TAG" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')
    echo "driver supports CUDA ${DRIVER_CUDA}"
    case "$DRIVER_CUDA" in
      13.*)         CUDA_TAG=cu130 ;;
      12.9|12.9.*)  CUDA_TAG=cu129 ;;
      12.8|12.8.*)  CUDA_TAG=cu128 ;;
      12.6|12.7|12.6.*|12.7.*) CUDA_TAG=cu126 ;;
      12.*)         CUDA_TAG=cu126 ;;
      *)            echo "unrecognised CUDA version '${DRIVER_CUDA}'"; CUDA_TAG=cu126 ;;
    esac
  else
    echo "nvidia-smi not found -- installing the CPU build"
    CUDA_TAG=cpu
  fi
fi
echo "using wheel index: ${CUDA_TAG}"

echo "=== installing torch (${CUDA_TAG}) ==="
# --index-url (NOT --extra-index-url): the extra form leaves PyPI in play, and
# uv will happily prefer PyPI's newer default-CUDA wheel over the pinned one.
# That is exactly how a cu130 build got installed against a CUDA 12.9 driver.
uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  torch torchvision

# Project deps come from PyPI as normal; torch is already satisfied above.
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
    print("!! CUDA NOT AVAILABLE -- training would fall back to CPU.")
    print("!! Most likely the torch wheel's CUDA is newer than the driver supports.")
    print("!! Re-run with an explicit tag, e.g.:  CUDA_TAG=cu129 bash scripts/remote_setup.sh")
    raise SystemExit(1)
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
