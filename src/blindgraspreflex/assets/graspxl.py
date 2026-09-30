"""GraspXL Objaverse assets as free-floating mjlab entities -- the RobustDexGrasp eval set.

RobustDexGrasp (arXiv 2504.05287, CoRL 2025) reports its simulation success rates over 247,786
Objaverse objects taken from GraspXL's size-normalised set. ``scripts/fetch_graspxl.py`` samples
that set (450,862 meshes on HuggingFace, in ``small``/``medium``/``large`` size buckets) and
prepares a subset into the exact same on-disk layout as the Objaverse and GSO trees
(``<bucket>_<uid>/{visual.obj, collision.obj, inertial.json}``, one uniform low-vertex collision
hull). Objects are keyed by (bucket, uid) because GraspXL rescales most meshes into all three size
buckets, so a uid alone names neither one object nor one size.

Because the layout is identical, this module is a thin wrapper like ``assets.gso``: it locates the
tree and lists its objects, then delegates spec/entity construction to ``assets.objaverse``.

Unlike GSO -- raw scans that must be size-normalised by our own pipeline -- GraspXL meshes arrive
already normalised into buckets by *shortest* OBB extent (small 3-5 cm, medium 5-7 cm, large
7-9 cm), so ``manifest.json`` carries each object's bucket and gives eval a ready-made size
stratification matching the paper's Table 2.

Run ``python scripts/fetch_graspxl.py`` to build the tree before using this task.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from blindgraspreflex import BLINDGRASPREFLEX_ROOT_PATH
from blindgraspreflex.assets.objaverse import (
    _is_object_dir,
    get_objaverse_object_cfg,
    get_objaverse_variant_object_cfg,
)

__all__ = [
    "find_graspxl_root",
    "get_graspxl_object_cfg",
    "get_graspxl_variant_object_cfg",
    "graspxl_buckets",
    "list_graspxl_objects",
]


def find_graspxl_root() -> Path:
    """Locate the prepared GraspXL tree (``BLINDGRASPREFLEX_GRASPXL_ROOT`` or ``<repo>/datasets/graspxl``)."""
    candidates = []
    if env_root := os.environ.get("BLINDGRASPREFLEX_GRASPXL_ROOT"):
        candidates.append(Path(env_root))
    candidates.append(BLINDGRASPREFLEX_ROOT_PATH / "datasets" / "graspxl")
    for candidate in candidates:
        if candidate.is_dir() and any(_is_object_dir(p) for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Prepared GraspXL dataset not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set BLINDGRASPREFLEX_GRASPXL_ROOT or run scripts/fetch_graspxl.py."
    )


def list_graspxl_objects(root: Path | None = None, bucket: str | None = None) -> tuple[str, ...]:
    """Sorted uids of all prepared GraspXL objects, or ``()`` if none/no tree (never raises).

    ``bucket`` filters to one GraspXL size bucket ("small", "medium" or "large") using
    ``manifest.json``; objects missing from the manifest are excluded when a bucket is requested.
    """
    try:
        root = root if root is not None else find_graspxl_root()
    except FileNotFoundError:
        return ()
    names = sorted(p.name for p in root.iterdir() if _is_object_dir(p))
    if bucket is None:
        return tuple(names)
    buckets = graspxl_buckets(root)
    return tuple(n for n in names if buckets.get(n) == bucket)


BUCKETS = ("small", "medium", "large")


@lru_cache(maxsize=4)
def _manifest(root_str: str) -> dict:
    path = Path(root_str) / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def graspxl_buckets(root: Path | None = None) -> dict[str, str]:
    """``{name: bucket}`` for every prepared object.

    Object directories are named ``<bucket>_<uid>``, so the size bucket is carried by the name
    itself and needs no lookup -- deliberately, because a uid alone does NOT determine a bucket
    (GraspXL rescales most meshes into all three). ``manifest.json`` is consulted only as a
    fallback for names that do not carry a recognised prefix.
    """
    try:
        root = root if root is not None else find_graspxl_root()
    except FileNotFoundError:
        return {}
    from_manifest = {o["uid"]: o["bucket"] for o in _manifest(str(root)).get("objects", [])}
    out = {}
    for name in list_graspxl_objects(root):
        prefix = name.split("_", 1)[0]
        out[name] = prefix if prefix in BUCKETS else from_manifest.get(name, "?")
    return out


def get_graspxl_object_cfg(name: str, root: Path | None = None, **kwargs):
    """mjlab entity config for a single free-floating GraspXL object."""
    return get_objaverse_object_cfg(
        name, root=root if root is not None else find_graspxl_root(), **kwargs
    )


def get_graspxl_variant_object_cfg(names: Sequence[str], root: Path | None = None, **kwargs):
    """A single ``object`` entity whose GraspXL mesh varies per world (see assets.objaverse)."""
    return get_objaverse_variant_object_cfg(
        names, root=root if root is not None else find_graspxl_root(), **kwargs
    )
