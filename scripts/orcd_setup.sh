#!/usr/bin/env bash
# One-time setup on the ORCD login node (run from ~/blindgraspreflex after scripts/orcd_sync.sh):
# the python env with the SAPG extra, and the pinned rl_games fork. Reuses the Manipulator-Software
# checkout that ~/reflexive-rl already has on ORCD.
set -euo pipefail
cd "$HOME/blindgraspreflex"
MS="${MS:-$HOME/reflexive-rl/third_party/manipulator-software}"
for xml in Rizon4_hand_robot.xml Rizon4_new_hand_robot.xml; do
  [ -f "$MS/assets/scene/flexiv_arm/$xml" ] || { echo "no $xml under $MS"; exit 1; }
done

command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv sync --extra cu128 --extra sapg
scripts/fetch_rl_games.sh
BLINDGRASPREFLEX_MANIPULATOR_SOFTWARE="$MS" .venv/bin/python -c "
import blindgraspreflex.training.sapg as s
s.use_reference_rl_games()
import rl_games
from blindgraspreflex.tasks.grasping.clutter_task import make_clutter_task
make_clutter_task(num_envs=6)
print('rl_games fork and clutter task OK')"
echo "setup done: $(pwd)"
