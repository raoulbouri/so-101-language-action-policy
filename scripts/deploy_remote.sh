#!/usr/bin/env bash
# Copy the code (NOT the 9.9 GB dataset) to the remote GPU box.
#
# Usage:
#   ./scripts/deploy_remote.sh                       # code only
#   WITH_DATA=1 ./scripts/deploy_remote.sh           # code + dataset (slow)
#
# Override any of: REMOTE_USER REMOTE_HOST REMOTE_DIR DATASET
#
# You will be prompted for the SSH password once per rsync invocation. If that
# gets tedious, set up a key first:
#     ssh-copy-id bbouri@172.24.170.204
# or enable connection multiplexing (see docs/REMOTE_TRAINING.md).

set -euo pipefail

REMOTE_USER=${REMOTE_USER:-bbouri}
REMOTE_HOST=${REMOTE_HOST:-172.24.170.204}
REMOTE_DIR=${REMOTE_DIR:-~/so-101-language-action-policy}
DATASET=${DATASET:-data/train_1200.hdf5}
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo "=== deploying to ${REMOTE}:${REMOTE_DIR} ==="

# Code only. Excludes everything large or machine-specific: the venv, the
# dataset, checkpoints, git history, and the vendored STL meshes (only needed
# for closed-loop rollout, not for training).
rsync -avz --progress \
  --exclude '.venv/' \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'runs/' \
  --exclude 'logs/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.mp4' \
  --exclude '*.gif' \
  ./ "${REMOTE}:${REMOTE_DIR}/"

if [ "${WITH_DATA:-0}" = "1" ]; then
  echo
  echo "=== copying dataset ($(du -h "$DATASET" | cut -f1)) -- this is the slow part ==="
  # -P keeps a partial file and resumes, so a dropped connection does not
  # restart 9.9 GB from zero.
  rsync -avP "$DATASET" "${REMOTE}:${REMOTE_DIR}/data/"
fi

echo
echo "=== done. next steps on the remote box ==="
cat <<'NEXT'
  ssh bbouri@172.24.170.204
  cd ~/so-101-language-action-policy
  bash scripts/remote_setup.sh          # creates .venv, installs CUDA torch
  # then see docs/REMOTE_TRAINING.md for the training command
NEXT
