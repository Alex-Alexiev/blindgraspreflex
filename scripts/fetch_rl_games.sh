#!/usr/bin/env bash
# Sparse-checkout the SAPG fork of rl_games from the reference SimToolReal repository (MIT) at the
# commit ~/brl/simtoolreal-flexiv pins, so blindgraspreflex.training.sapg runs the same algorithm code.
# Usage: scripts/fetch_rl_games.sh [destination]   (default: third_party/simtoolreal)
set -euo pipefail
COMMIT=313d5aea1f507c6cfe097b672b62945d7b0bbff5
DEST=${1:-third_party/simtoolreal}
PATHS=(/LICENSE /rl_games/)
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/tylerlum/simtoolreal "$DEST"
fi
git -C "$DEST" sparse-checkout set --no-cone "${PATHS[@]}"
git -C "$DEST" checkout -q "$COMMIT"
echo "SimToolReal rl_games @ ${COMMIT:0:7} in $DEST"
