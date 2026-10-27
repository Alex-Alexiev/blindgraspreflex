#!/usr/bin/env bash
# Submit a SAPG clutter run as one copy per GPU target; the first copy to start cancels the rest.
#
#   RUN=<name> [ITERS=5000] [NENV=24576] [EXTRA="<sapg flags>"] [INIT=<warm-start .pth>] \
#   [GPUS="h100 h200 a100 mit_normal_gpu:h100 mit_normal_gpu:h200"] [WALLTIME=06:00:00] \
#   scripts/orcd_submit.sh
#
# Slurm cannot OR GPU types in one request, so each target gets its own copy. A GPUS entry is a
# GPU type on mit_preemptable ("h100") or PARTITION:TYPE ("mit_normal_gpu:h200"; not preempted,
# 2 GPUs per user, 6 h). A copy that lands on a card too small for NENV (a 40 GB A100) exits without
# cancelling the others. A short WALLTIME backfills better; the job resumes and resubmits itself.
set -euo pipefail
cd "$HOME/blindgraspreflex"
: "${RUN:?set RUN=<run name>}"
ITERS=${ITERS:-5000}
NENV=${NENV:-24576}
EXTRA=${EXTRA:-}
INIT=${INIT:-}
GPUS=${GPUS:-"h100 h200 a100 mit_normal_gpu:h100 mit_normal_gpu:h200"}
WALLTIME=${WALLTIME:-06:00:00}   # HH:MM:SS
secs=$(echo "$WALLTIME" | awk -F: '{print $1 * 3600 + $2 * 60 + $3}')
CAP_SEC=$(( secs - 1800 ))       # leave 30 min for startup, saving and resubmission
EXCLUDE=${EXCLUDE:-}             # comma-separated nodes to avoid (set by orcd_train.sbatch after a crash)
export RUN ITERS NENV EXTRA INIT GPUS WALLTIME CAP_SEC EXCLUDE
for spec in $GPUS; do
  case "$spec" in
    *:*) part=${spec%%:*}; gpu=${spec#*:} ;;
    *) part=mit_preemptable; gpu=$spec ;;
  esac
  id=$(sbatch --parsable --partition="$part" --job-name="bgr-$RUN" --gres="gpu:$gpu:1" --time="$WALLTIME" \
    ${EXCLUDE:+--exclude="$EXCLUDE"} --export=ALL scripts/orcd_train.sbatch)
  echo "submitted $RUN on $part/$gpu: job $id"
done
