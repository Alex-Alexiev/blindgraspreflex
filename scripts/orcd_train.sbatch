#!/bin/bash
# SAPG training of the primitive clutter task on ORCD (blindgraspreflex.training.sapg).
#
#   Submit through scripts/orcd_submit.sh (one copy per GPU target, first to start wins).
#
# 24,576 envs (6 SAPG blocks x 4,096) need ~54 GB (measured 2.2 MiB per env + 1 GB on an RTX 4070,
# with the 64 / 1024 contact budget), so the job wants an 80 GB card or larger.
#
# Preemption-safe: --requeue, and every start resumes from the run's newest checkpoint (rl_games
# restores the epoch counter and optimiser; W&B continues the same run id). On SIGTERM the trainer
# finishes its epoch and saves. When the wall clock ends with epochs left, the job resubmits itself.
#SBATCH --job-name=bgr-sapg
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --requeue
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

set -uo pipefail
cd "$HOME/blindgraspreflex"
module load cuda/13.0.1
# The same environment as the local teacher runs. REFLEXIVE_TRACKING_QCMD stays unset there (they
# are launched with python -m, not the blindgraspreflex CLI), so the tracking-error observation is
# hand_target - q; keep it that way here.
unset REFLEXIVE_TRACKING_QCMD
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_MODE=online WANDB_ENTITY="${WANDB_ENTITY:-aalexiev-mit}"
export BLINDGRASPREFLEX_MANIPULATOR_SOFTWARE="${MS:-$HOME/reflexive-rl/third_party/manipulator-software}"
RUN="${RUN:?set RUN=<run name>}"
ITERS="${ITERS:-5000}"
NENV="${NENV:-24576}"
EXTRA="${EXTRA:-}"   # extra blindgraspreflex.training.sapg flags
INIT="${INIT:-}"     # optional checkpoint to warm-start from (only before the run has its own)
echo "=== $(hostname) $(date) RUN=$RUN ITERS=$ITERS NENV=$NENV EXTRA=$EXTRA ==="; nvidia-smi -L

MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
MIN_MEM=$(( NENV * 3 + 2048 ))   # MiB: 2.2 MiB per env + 1 GB measured, with margin
if [ "$MEM" -lt "$MIN_MEM" ]; then
  echo "=== ${MEM} MiB card is too small for ${NENV} envs (needs ${MIN_MEM}); leaving it to the other copies ==="
  exit 0
fi

# Copies submitted by orcd_submit.sh share the job name "bgr-<RUN>". The first to start cancels the
# pending ones. A copy that starts while another is running steps aside: one that started earlier,
# or at the same second with a lower job id, keeps the run. (Parsing squeue for the pending ids
# missed 3 of 4 copies on 2026-10-22, so scancel filters them itself.)
if [ "${SLURM_JOB_NAME:-}" = "bgr-$RUN" ]; then
  mystart=$(squeue -h -j "$SLURM_JOB_ID" -o %S)
  older=$(squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t R,CG -o "%i %S" |
    awk -v me="$SLURM_JOB_ID" -v ms="$mystart" '$1 != me && ($2 < ms || ($2 == ms && $1 < me)) {print $1}')
  if [ -n "$older" ]; then echo "=== copy $older of $RUN is already running; exiting ==="; exit 0; fi
  scancel -u "$USER" --name="$SLURM_JOB_NAME" --state=PENDING
  echo "=== cancelled pending copies of $RUN; queue now: $(squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -o '%i %T' | tr '\n' ' ') ==="
fi
( while sleep 600; do echo "[gpu] $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"; done ) &
GPU_LOG=$!

RUNDIR="logs/sapg/$RUN/00_$RUN"
latest(){
  if [ -f "$RUNDIR/last/model.pth" ]; then echo "$RUNDIR/last/model.pth"
  else ls -1t "$RUNDIR"/snapshots/*.pth "$RUNDIR"/nn/*.pth 2>/dev/null | head -1; fi
}
CAP=${CAP_SEC:-$(( 5 * 3600 + 1800 ))}   # walltime minus room for startup and resubmission

for try in 1 2 3; do
  RESUME=(); CK=$(latest)
  if [ -n "$CK" ]; then RESUME=(--resume-from "$CK"); echo "[resume] $CK"
  elif [ -n "$INIT" ]; then RESUME=(--init-from "$INIT"); echo "[init] warm start from $INIT"; fi
  timeout "$CAP" .venv/bin/python -u -m blindgraspreflex.training.sapg --device cuda:0 \
    --num-envs "$NENV" --iterations "$ITERS" --run-name "$RUN" --log-dir logs/sapg \
    $EXTRA "${RESUME[@]}" 2>&1 | tee "/tmp/bgr_sapg_$$.log"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then break; fi
  if grep -qiE "out of memory|OutOfMemoryError|Failed to allocate" "/tmp/bgr_sapg_$$.log"; then
    NENV=$(( NENV / 2 )); echo "!! OOM -> ${NENV} envs"
    [ "$NENV" -lt 3072 ] && { echo "!! giving up"; break; }
  else
    echo "!! train failed rc=$rc (not an OOM); resubmitting from the latest checkpoint without node $(hostname -s)"
    NODE_CRASH=1
    break
  fi
done
rm -f "/tmp/bgr_sapg_$$.log"
kill "$GPU_LOG" 2>/dev/null

if [ "${NODE_CRASH:-0}" -eq 1 ]; then
  EXCLUDE="${EXCLUDE:+$EXCLUDE,}$(hostname -s)"
  if [ "$(echo "$EXCLUDE" | tr ',' '\n' | wc -l)" -le 5 ]; then
    RUN="$RUN" ITERS="$ITERS" NENV="$NENV" EXTRA="$EXTRA" INIT="$INIT" GPUS="${GPUS:-h100 h200}" \
      WALLTIME="${WALLTIME:-06:00:00}" EXCLUDE="$EXCLUDE" "$HOME/blindgraspreflex/scripts/orcd_submit.sh"
  else
    echo "!! crashed on 6 different nodes ($EXCLUDE); giving up"
  fi
elif [ "${rc:-1}" -eq 124 ]; then
  echo "=== wall-clock cap reached; resubmitting to continue $RUN ==="
  RUN="$RUN" ITERS="$ITERS" NENV="$NENV" EXTRA="$EXTRA" INIT="$INIT" GPUS="${GPUS:-h100 h200}" \
    WALLTIME="${WALLTIME:-06:00:00}" EXCLUDE="${EXCLUDE:-}" "$HOME/blindgraspreflex/scripts/orcd_submit.sh"
fi
echo "=== DONE $(date) rc=${rc:-?} checkpoint=$(latest) ==="
