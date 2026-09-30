"""Objaverse object assets as free-floating mjlab entities.

The Objaverse counterpart to :mod:`blindgraspreflex.assets.ycb`. Consumes the *prepared* tree that
``scripts/fetch_objaverse.py`` produces from GraspXL's 30-object example set (decimated,
texture-free Objaverse meshes). Each object directory, keyed by its Objaverse UID, provides:

* ``visual.obj``     canonical visual mesh (OBB-aligned, origin at the OBB centre, native scale)
* ``collision.obj``  canonical collision mesh, CoACD-decomposed into disjoint convex hulls
* ``inertial.json``  ``{mass, diaginertia, com}`` inferred from the convex hull at uniform density

The layout mirrors the prepared YCB tree closely enough that this module is a near-copy of
``assets.ycb`` -- same per-hull collision split, same explicit-inertial body, same single- vs
per-world-variant entity configs -- differing only in where the mesh/inertia data is read from
(a UID directory + JSON here, vs ``google_16k`` + URDF for YCB) and in the object list being
discovered on disk (Objaverse UIDs are opaque) rather than hard-coded.

Run ``python scripts/fetch_objaverse.py`` to build the tree before using these tasks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from mjlab.entity import EntityCfg

from blindgraspreflex import BLINDGRASPREFLEX_ROOT_PATH

__all__ = [
    "ObjaverseInertial",
    "find_objaverse_root",
    "get_objaverse_object_cfg",
    "get_objaverse_spec",
    "get_objaverse_variant_object_cfg",
    "list_objaverse_objects",
    "load_objaverse_inertial",
]

##
# Dataset location.
##


def find_objaverse_root() -> Path:
    """Locate the prepared Objaverse tree.

    Checked in order: ``BLINDGRASPREFLEX_OBJAVERSE_ROOT`` then ``<repo>/datasets/objaverse``. The
    tree is built by ``scripts/fetch_objaverse.py`` and is gitignored (like the YCB tree).
    """
    candidates = []
    if env_root := os.environ.get("BLINDGRASPREFLEX_OBJAVERSE_ROOT"):
        candidates.append(Path(env_root))
    candidates.append(BLINDGRASPREFLEX_ROOT_PATH / "datasets" / "objaverse")
    for candidate in candidates:
        if candidate.is_dir() and any(_is_object_dir(p) for p in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(
        "Prepared Objaverse dataset not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set BLINDGRASPREFLEX_OBJAVERSE_ROOT or run scripts/fetch_objaverse.py."
    )


def _is_object_dir(p: Path) -> bool:
    return p.is_dir() and (p / "visual.obj").exists() and (p / "inertial.json").exists()


def list_objaverse_objects(root: Path | None = None) -> tuple[str, ...]:
    """Return the sorted UIDs of all prepared Objaverse objects, or ``()`` if none/no tree.

    Never raises: task registration calls this at import, and a missing dataset should simply
    leave the Objaverse task unregistered rather than break the YCB tasks.
    """
    try:
        root = root if root is not None else find_objaverse_root()
    except FileNotFoundError:
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if _is_object_dir(p)))


##
# Inertial properties.
##


@dataclass(frozen=True)
class ObjaverseInertial:
    """Mass and diagonal inertia read from ``inertial.json``, in the canonical mesh frame."""

    mass: float
    diaginertia: tuple[float, float, float]
    com: tuple[float, float, float]


def load_objaverse_inertial(name: str, root: Path | None = None) -> ObjaverseInertial:
    """Read mass/inertia from the object's ``inertial.json``.

    Read rather than inferred by MuJoCo from geometry: the values were fitted once from the
    convex hull at a chosen density (the raw meshes are non-watertight, so MuJoCo's own
    geometry-based inertia would be unreliable), and baking them keeps runs reproducible.
    """
    root = root if root is not None else find_objaverse_root()
    data = json.loads((root / name / "inertial.json").read_text())
    return ObjaverseInertial(
        mass=float(data["mass"]),
        diaginertia=tuple(float(v) for v in data["diaginertia"]),  # type: ignore[arg-type]
        com=tuple(float(v) for v in data["com"]),  # type: ignore[arg-type]
    )


@lru_cache(maxsize=64)
def _load_meshes(name: str, root_str: str) -> tuple[trimesh.Trimesh, tuple[trimesh.Trimesh, ...]]:
    """Return (visual mesh, per-hull collision meshes). Cached: repeated across worlds/resets."""
    obj_dir = Path(root_str) / name
    visual = trimesh.load(obj_dir / "visual.obj", force="mesh", process=False)
    collision = trimesh.load(obj_dir / "collision.obj", force="mesh", process=False)
    hulls = collision.split(only_watertight=False)
    if len(hulls) == 0:  # single-component decomposition comes back unsplit
        hulls = [collision]
    return visual, tuple(hulls)


##
# Spec construction.
##


def add_objaverse_object(
    spec: mujoco.MjSpec,
    name: str,
    root: Path | None = None,
    body_name: str = "object",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.55, 0.6, 0.85, 1.0),
    max_collision_hulls: int | None = None,
    free: bool = True,
    scale: float = 1.0,
):
    """Add one Objaverse object to an existing spec and return its body.

    The mesh is used unscaled: the prepared tree has already baked in the canonical transform,
    so the body frame is the OBB centre and no further transform should be applied. Identical in
    structure to :func:`blindgraspreflex.assets.ycb.add_ycb_object`; only the data source differs.
    """
    root = root if root is not None else find_objaverse_root()
    visual, hulls = _load_meshes(name, str(root))
    inertial = load_objaverse_inertial(name, root)

    if max_collision_hulls is not None and len(hulls) > max_collision_hulls:
        # Keep the largest hulls by volume: dropping small ones loses surface detail but
        # preserves the gross shape, whereas truncating in file order can delete a whole side.
        hulls = tuple(sorted(hulls, key=lambda h: h.volume, reverse=True)[:max_collision_hulls])

    spec.add_mesh(
        name=f"{name}_visual_mesh",
        uservert=(np.asarray(visual.vertices, dtype=np.float64) * scale).flatten(),
        userface=np.asarray(visual.faces, dtype=np.int32).flatten(),
    )
    body = spec.worldbody.add_body(name=body_name, pos=pos)
    if free:
        body.add_freejoint(name=f"{body_name}_free")

    # Inertia is set explicitly from inertial.json rather than inferred from the hulls.
    # Scaled to a uniform-density rescale (mass ~ s^3, inertia ~ s^5, COM ~ s), matching ycb.py.
    body.explicitinertial = True
    body.mass = inertial.mass * scale**3
    body.inertia = tuple(float(v) * scale**5 for v in inertial.diaginertia)
    body.ipos = tuple(float(v) * scale for v in inertial.com)

    body.add_geom(
        name=f"{body_name}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_visual_mesh",
        contype=0,
        conaffinity=0,
        group=1,
        density=0.0,
        rgba=rgba,
    )
    for i, hull in enumerate(hulls):
        # The prepared collision.obj is already a single uniform low-vertex convex hull (~42
        # verts, sampled along a fixed direction set for every object -- see
        # scripts/fetch_objaverse.py). Do NOT re-cap with maxhullvert: that would let MuJoCo's
        # qhull keep a *different* subset of vertices per object, re-introducing the cross-world
        # heterogeneity whose ~10x mjwarp variant-collision slowdown this representation exists
        # to avoid.
        spec.add_mesh(
            name=f"{name}_hull_{i}",
            uservert=(np.asarray(hull.vertices, dtype=np.float64) * scale).flatten(),
            userface=np.asarray(hull.faces, dtype=np.int32).flatten(),
        )
        body.add_geom(
            name=f"{body_name}_collision_{i}",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"{name}_hull_{i}",
            group=3,
            # Mass comes from the explicit inertial block above; density 0 stops the hulls
            # from also contributing, which would double-count and overshoot the true mass.
            density=0.0,
            # condim 6 to match the hand's grasp surfaces -- without torsional and rolling
            # friction a grasped object spins freely about the contact normals.
            condim=6,
            friction=(1.0, 5e-3, 1e-4),
            # Stiff contact so fast-closing fingers don't visibly sink into the object (see the
            # matching note in assets/ycb.py): 0.01 s solref + near-rigid solimp, priority 2 so
            # these params win the hand<->object pair. Fixes the "hand inside object" clipping at
            # 250 Hz without the ~14x cost of decimation-20 500 Hz physics.
            solref=(0.01, 1.0),
            # Stiff contact so the finger barely sinks in; ~2-3mm at 250Hz is a physics-rate
            # floor (see assets/ycb.py). Guarded by the instability termination + SafeVecEnvWrapper.
            solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
            priority=2,
        )
    return body


def get_objaverse_spec(name: str, **kwargs) -> mujoco.MjSpec:
    """Build a standalone spec holding one free-floating Objaverse object."""
    spec = mujoco.MjSpec()
    add_objaverse_object(spec, name, **kwargs)
    return spec


def get_objaverse_object_cfg(
    name: str,
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    root: Path | None = None,
    max_collision_hulls: int | None = None,
    **spec_kwargs,
) -> EntityCfg:
    """mjlab entity config for a single free-floating Objaverse object."""

    def spec_fn() -> mujoco.MjSpec:
        return get_objaverse_spec(
            name, root=root, max_collision_hulls=max_collision_hulls, **spec_kwargs
        )

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}),
        spec_fn=spec_fn,
    )


def get_objaverse_variant_object_cfg(
    names: Sequence[str],
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    root: Path | None = None,
    max_collision_hulls: int | None = None,
    assignment=None,
    **spec_kwargs,
):
    """A single ``object`` entity whose *mesh varies per world* across ``names``.

    The Objaverse analogue of :func:`blindgraspreflex.assets.ycb.get_ycb_variant_object_cfg`; see
    there for the per-world mesh-variant mechanics. Every world shares one compiled scene and
    the same kinematic structure (one free body named ``object``); only the geometry, mass and
    inertia differ per world. The merged model is padded to the largest variant's geom count,
    so a bounded ``max_collision_hulls`` keeps model size flat as the object set grows.
    """
    import os

    from mjlab.entity.variants import VariantEntityCfg

    if not names:
        raise ValueError("get_objaverse_variant_object_cfg needs at least one object name.")

    # Optional per-world size domain randomisation, read from the same REFLEXIVE_OBJECT_SCALES
    # env var ycb.py uses -- without this the Objaverse/GSO tasks silently ignored the scale DR
    # every YCB caller sets, so a GSO eval ran every object at scale 1.0 while the YCB eval it
    # was compared against spanned 0.7-1.0. Each (object, scale) becomes its own per-world variant.
    _scales_env = os.environ.get("REFLEXIVE_OBJECT_SCALES", "").strip()
    scales = tuple(float(s) for s in _scales_env.split(",") if s) if _scales_env else (1.0,)

    def make(name: str, s: float) -> Callable[[], mujoco.MjSpec]:
        return lambda: get_objaverse_spec(
            name, root=root, max_collision_hulls=max_collision_hulls, scale=s, **spec_kwargs
        )

    if scales == (1.0,):
        variants = {name: make(name, 1.0) for name in names}
    else:
        variants = {f"{name}@{s:g}": make(name, s) for name in names for s in scales}

    return VariantEntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}),
        variants=variants,
        assignment=assignment,
    )
