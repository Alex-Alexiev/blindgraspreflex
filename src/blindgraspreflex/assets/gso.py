"""Google Scanned Objects (GSO) assets as free-floating mjlab entities.

A curated, size-normalised, graspable subset of GSO prepared by ``scripts/fetch_gso.py`` into the
exact same on-disk layout as the Objaverse tree (``<name>/{visual.obj, collision.obj,
inertial.json}``, a single uniform low-vertex collision hull). Because the layout is identical,
this module is a thin wrapper: it only locates the GSO tree and lists its objects, then delegates
spec/entity construction to ``assets.objaverse`` with ``root`` pointed at the GSO tree.

Run ``python scripts/fetch_gso.py`` to build the tree before using this task.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from blindgraspreflex import BLINDGRASPREFLEX_ROOT_PATH
from blindgraspreflex.assets.objaverse import (
    _is_object_dir,
    get_objaverse_object_cfg,
    get_objaverse_variant_object_cfg,
)

__all__ = [
    "find_gso_root",
    "get_gso_object_cfg",
    "get_gso_variant_object_cfg",
    "list_gso_objects",
]


def find_gso_root() -> Path:
    """Locate the prepared GSO tree (``BLINDGRASPREFLEX_GSO_ROOT`` or ``<repo>/datasets/gso``)."""
    candidates = []
    if env_root := os.environ.get("BLINDGRASPREFLEX_GSO_ROOT"):
        candidates.append(Path(env_root))
    candidates.append(BLINDGRASPREFLEX_ROOT_PATH / "datasets" / "gso")
    for candidate in candidates:
        if candidate.is_dir() and any(_is_object_dir(p) for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Prepared GSO dataset not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set BLINDGRASPREFLEX_GSO_ROOT or run scripts/fetch_gso.py."
    )


def list_gso_objects(root: Path | None = None) -> tuple[str, ...]:
    """Sorted names of all prepared GSO objects, or ``()`` if none/no tree (never raises)."""
    try:
        root = root if root is not None else find_gso_root()
    except FileNotFoundError:
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if _is_object_dir(p)))


def get_gso_object_cfg(name: str, root: Path | None = None, **kwargs):
    """mjlab entity config for a single free-floating GSO object."""
    return get_objaverse_object_cfg(
        name, root=root if root is not None else find_gso_root(), **kwargs
    )


def get_gso_variant_object_cfg(names: Sequence[str], root: Path | None = None, **kwargs):
    """A single ``object`` entity whose GSO mesh varies per world (see assets.objaverse)."""
    return get_objaverse_variant_object_cfg(
        names, root=root if root is not None else find_gso_root(), **kwargs
    )
