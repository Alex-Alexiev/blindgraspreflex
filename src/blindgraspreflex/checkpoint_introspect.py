"""Read a policy's action-space shape back out of its checkpoint.

The env must be built with the SAME action layout the checkpoint was trained with, or
``load_state_dict`` fails on a size mismatch -- after the env has been constructed and, in the
distillation case, after hours of upstream compute. Every consumer of a checkpoint previously
hardcoded ``arm_residual = True``, which was correct until policies started training without the
palm residual; then eval, dislodge and distillation all broke the same way in the same place.

Action layouts (see ``GraspAction._n_pre``)::

    residual   : [grasp_score(1), pos_res(3), rot_res(3), hand(20 joints | 5 pca)]  -> 27 | 12
    no residual: [grasp_score(1),                         hand(20 joints | 5 pca)]  -> 21 |  6
"""

from __future__ import annotations

import glob
from pathlib import Path

_ACTION_DIMS = (6, 12, 21, 26, 27)
"""26 = the task-space baseline: palm pos(3) + rpy(3) + hand(20)."""
_NO_RESIDUAL_DIMS = (6, 21)


def latest_checkpoint(run_dir: str | Path) -> str | None:
    """Newest ``model_<n>.pt`` in ``run_dir`` by iteration number (not mtime, not lexical)."""
    files = glob.glob(f"{run_dir}/model_*.pt")
    if not files:
        return None
    return max(files, key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))


def action_dim(checkpoint: str | Path) -> int | None:
    """Action dimension of the policy in ``checkpoint``, or None if it cannot be determined.

    Reads the final layer's bias width from the actor state dict. Returns None rather than guessing
    when nothing plausible is found -- callers should keep their configured default in that case,
    since a wrong guess here produces exactly the silent mismatch this module exists to prevent.
    """
    import torch

    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # PPO (train_grasp) saves "actor_state_dict"; DistillationRunner saves "student_state_dict"
    # alongside a frozen "teacher_state_dict". Checking only the PPO key silently returned None for
    # every distilled student, which fell back to the default and reintroduced the mismatch.
    sd = next(
        (
            blob[k]
            for k in ("actor_state_dict", "student_state_dict", "model_state_dict")
            if isinstance(blob.get(k), dict)
        ),
        {},
    )
    dims = [
        v.shape[0]
        for k, v in sd.items()
        if k.endswith("bias") and getattr(v, "ndim", 0) == 1 and v.shape[0] in _ACTION_DIMS
    ]
    return dims[-1] if dims else None


def obs_dim(checkpoint: str | Path) -> int | None:
    """Observation width the policy in ``checkpoint`` was trained on, or None.

    Read off the observation normaliser's buffers, whose width IS the obs dim. Lets a consumer
    detect an env/checkpoint mismatch BEFORE building the env, rather than discovering it as a
    state_dict shape error afterwards -- which is how the joint-target baseline's eval failed three
    times in a row, each time on a flag that had to be remembered by hand.
    """
    import torch

    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    sd = next(
        (
            blob[k]
            for k in ("actor_state_dict", "student_state_dict", "model_state_dict")
            if isinstance(blob.get(k), dict)
        ),
        {},
    )
    for key in ("obs_normalizer._mean", "obs_normalizer.running_mean"):
        v = sd.get(key)
        if v is not None:
            return int(v.shape[-1])
    return None


def uses_residual(checkpoint: str | Path, default: bool = True) -> bool:
    """Whether the policy in ``checkpoint`` emits the 6-D palm residual.

    ``default`` is returned when the action dim cannot be read, so an unreadable checkpoint keeps
    the caller's configured behaviour instead of silently flipping it.
    """
    dim = action_dim(checkpoint)
    if dim is None:
        return default
    return dim not in _NO_RESIDUAL_DIMS


def resolve_residual(checkpoint: str | Path | None, configured: bool, *, tag: str = "eval") -> bool:
    """Return the residual setting to build the env with, announcing any override.

    The checkpoint's own action space always wins over the flag: the flag expresses intent, the
    checkpoint is ground truth about what will actually load.
    """
    if not checkpoint:
        return configured
    dim = action_dim(checkpoint)
    if dim is None:
        return configured
    detected = dim not in _NO_RESIDUAL_DIMS
    if detected != configured:
        print(
            f"[{tag}] checkpoint action dim {dim} -> arm_residual={detected} "
            f"(overriding the flag; the checkpoint's space wins)"
        )
    return detected
