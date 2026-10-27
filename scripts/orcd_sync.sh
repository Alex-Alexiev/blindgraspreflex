#!/usr/bin/env bash
# Push this repo's code to ORCD (orcd:~/blindgraspreflex). Needs an open SSH master
# (`ssh orcd` once in a terminal; ~/.ssh/config multiplexes it).
#
# Never add --delete / --delete-excluded: the remote holds files the local tree does not
# (logs, checkpoints, its own venv and third_party checkout), and --delete-excluded implies --delete.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -az --info=stats1 \
  --exclude .venv/ --exclude logs/ --exclude wandb/ --exclude third_party/ --exclude dist/ \
  --exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ --exclude configs/local.toml \
  ./ orcd:blindgraspreflex/
echo "synced to orcd:~/blindgraspreflex"
