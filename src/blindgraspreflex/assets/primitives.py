"""Procedural primitive-shape objects as free-floating mjlab entities.

The simplest, cleanest object "dataset" for the grasping task: a deterministic set of
**superquadric** shapes, one continuous family that spans box <-> cylinder <-> ellipsoid <->
sphere via two shape exponents plus per-axis scale. Nothing is downloaded or scanned -- every
object is generated analytically at scene-build time.

Why superquadrics specifically:

* **Fast in mjwarp.** Like ``assets.objaverse``, the collision geom is a single low-vertex
  convex hull sampled along one *fixed* set of ~42 directions, so every object shares the same
  collision topology. That uniformity is what keeps the per-world mesh-variant scene from
  hitting the ~10x lockstep-divergence slowdown that heterogeneous scan meshes cause.
* **Clean and controllable.** Watertight convex shapes, analytic mass/inertia, and a built-in
  easy->hard knob: the shape exponents and aspect ratios are an explicit difficulty axis for a
  grasp curriculum, and every shape is trivially graspable.

The public API mirrors ``assets.ycb`` / ``assets.objaverse`` (``get_primitive_spec``,
``get_primitive_object_cfg``, ``get_primitive_variant_object_cfg``) so ``make_task`` treats it
as just another object ``source``. Object "names" are synthetic ids ``prim_000`` .. that index
:data:`PRIMITIVE_OBJECT_LIST`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import mujoco
import numpy as np
import trimesh
from mjlab.entity import EntityCfg

__all__ = [
    "PRIMITIVE_OBJECT_LIST",
    "get_primitive_object_cfg",
    "get_primitive_spec",
    "get_primitive_variant_object_cfg",
]

# Uniform-density inertia, same convention/range as assets.objaverse.
DENSITY = 300.0  # kg/m^3
MASS_MIN, MASS_MAX = 0.03, 0.6  # kg
DIAGINERTIA_MIN = 1e-6

# Shape catalogue. Each object is (eps1, eps2, aspect, short_extent):
#   eps1/eps2 -- superquadric exponents; ~0.2 is boxy, 1.0 is round. (eps1,eps2)=(0.2,1.0) is a
#     cylinder, (1.0,1.0) an ellipsoid/sphere, (0.2,0.2) a box, etc. Kept <=1 so shapes stay
#     convex (a single cheap collision hull).
#   aspect -- relative x:y:z extents before scaling (cube, tall, long, flat, ...).
#   short_extent -- target shortest full extent [m], kept in the hand's grasp range.
_EPS_PAIRS = (
    (0.2, 0.2),  # box
    (0.2, 1.0),  # cylinder
    (1.0, 1.0),  # ellipsoid / sphere
    (0.5, 0.5),  # rounded box
    (0.4, 1.0),  # rounded cylinder
    (1.0, 0.4),  # square-section, rounded profile
    (0.7, 0.7),  # blobby
    (0.3, 0.7),  # boxy-blobby
)
_ASPECTS = (
    (1.0, 1.0, 1.0),  # cube / sphere
    (1.0, 1.0, 1.6),  # tall
    (1.5, 1.0, 1.0),  # long
    (1.0, 1.0, 0.55),  # flat
    (1.3, 1.0, 0.8),  # slab-ish
)
_SHORT = (0.05, 0.065, 0.08)  # cycled across shapes for size diversity


def _catalogue() -> list[tuple[tuple[float, float], tuple[float, float, float], float]]:
    out = []
    i = 0
    for eps in _EPS_PAIRS:
        for aspect in _ASPECTS:
            out.append((eps, aspect, _SHORT[i % len(_SHORT)]))
            i += 1
    return out


_CATALOGUE = _catalogue()
PRIMITIVE_OBJECT_LIST = tuple(f"prim_{i:03d}" for i in range(len(_CATALOGUE)))


def _params(name: str):
    if not name.startswith("prim_"):
        raise ValueError(f"Unknown primitive object {name!r} (expected 'prim_NNN').")
    idx = int(name.split("_")[1])
    if not 0 <= idx < len(_CATALOGUE):
        raise ValueError(f"Primitive index {idx} out of range 0..{len(_CATALOGUE) - 1}.")
    return _CATALOGUE[idx]


# Fixed support directions (icosphere vertices) for the uniform collision hull -- identical for
# every object so all collision hulls share topology. Cached.
_HULL_DIRS: np.ndarray | None = None


def _hull_dirs() -> np.ndarray:
    global _HULL_DIRS
    if _HULL_DIRS is None:
        ico = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        _HULL_DIRS = np.asarray(ico.vertices) / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    return _HULL_DIRS


def _sgnpow(v: np.ndarray, p: float) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** p


def _superquadric_points(half: np.ndarray, eps: tuple[float, float], n: int = 24) -> np.ndarray:
    """Dense point cloud on a superquadric surface with half-extents ``half`` (a1, a2, a3)."""
    e1, e2 = eps
    eta = np.linspace(-np.pi / 2, np.pi / 2, n)
    omega = np.linspace(-np.pi, np.pi, n)
    e, o = np.meshgrid(eta, omega)
    x = half[0] * _sgnpow(np.cos(e), e1) * _sgnpow(np.cos(o), e2)
    y = half[1] * _sgnpow(np.cos(e), e1) * _sgnpow(np.sin(o), e2)
    z = half[2] * _sgnpow(np.sin(e), e1)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


@lru_cache(maxsize=128)
def _build_meshes(name: str) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Return (visual hull, uniform collision hull, inertial dict) for one primitive."""
    eps, aspect, short = _params(name)
    aspect = np.asarray(aspect, dtype=np.float64)
    # Half-extents so the shortest full extent equals `short`.
    half = aspect * (short / (2.0 * float(aspect.min())))
    pts = _superquadric_points(half, eps)
    visual = trimesh.convex.convex_hull(pts)  # convex (eps <= 1) -> clean watertight shape

    # Uniform collision hull: support point of the shape along each fixed direction.
    centre = np.asarray(visual.center_mass, dtype=np.float64)
    v = np.asarray(visual.vertices, dtype=np.float64) - centre
    support = np.array([v[int(np.argmax(v @ d))] for d in _hull_dirs()]) + centre
    collision = trimesh.convex.convex_hull(support)

    vol = float(visual.volume)
    mass = float(np.clip(DENSITY * vol, MASS_MIN, MASS_MAX))
    visual.density = mass / max(vol, 1e-9)
    diag = np.clip(
        np.diag(np.asarray(visual.moment_inertia, dtype=np.float64)), DIAGINERTIA_MIN, None
    )
    inertial = {
        "mass": mass,
        "diaginertia": tuple(float(x) for x in diag),
        "com": tuple(float(x) for x in np.asarray(visual.center_mass, dtype=np.float64)),
    }
    return visual, collision, inertial


def add_primitive_object(
    spec: mujoco.MjSpec,
    name: str,
    body_name: str = "object",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.45, 0.75, 0.5, 1.0),
    max_collision_hulls: int | None = None,  # accepted for API parity; primitives have one hull
    free: bool = True,
):
    """Add one procedural primitive object to a spec and return its body.

    Structurally identical to ``assets.objaverse.add_objaverse_object`` -- a visual mesh geom, a
    single uniform convex collision hull with stiff contact params, and an explicit inertial --
    only the mesh data is generated analytically rather than read from disk.
    """
    del max_collision_hulls
    visual, collision, inertial = _build_meshes(name)

    spec.add_mesh(
        name=f"{name}_visual_mesh",
        uservert=np.asarray(visual.vertices, dtype=np.float64).flatten(),
        userface=np.asarray(visual.faces, dtype=np.int32).flatten(),
    )
    body = spec.worldbody.add_body(name=body_name, pos=pos)
    if free:
        body.add_freejoint(name=f"{body_name}_free")

    body.explicitinertial = True
    body.mass = inertial["mass"]
    body.inertia = inertial["diaginertia"]
    body.ipos = inertial["com"]

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
    spec.add_mesh(
        name=f"{name}_hull_0",
        uservert=np.asarray(collision.vertices, dtype=np.float64).flatten(),
        userface=np.asarray(collision.faces, dtype=np.int32).flatten(),
    )
    body.add_geom(
        name=f"{body_name}_collision_0",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_hull_0",
        group=3,
        density=0.0,
        condim=6,
        friction=(1.0, 5e-3, 1e-4),
        # Stiff contact (matches assets.ycb / assets.objaverse): keeps fast-closing fingers from
        # sinking into the object; priority 2 so these params win the hand<->object pair.
        solref=(0.01, 1.0),
        # Stiff contact so the finger barely sinks in; ~2-3mm at 250Hz is a physics-rate
        # floor (see assets/ycb.py). Guarded by the instability termination + SafeVecEnvWrapper.
        solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        priority=2,
    )
    return body


def get_primitive_spec(name: str, **kwargs) -> mujoco.MjSpec:
    """Build a standalone spec holding one free-floating primitive object."""
    spec = mujoco.MjSpec()
    add_primitive_object(spec, name, **kwargs)
    return spec


def get_primitive_object_cfg(
    name: str,
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    max_collision_hulls: int | None = None,
    **spec_kwargs,
) -> EntityCfg:
    """mjlab entity config for a single free-floating primitive object."""
    del max_collision_hulls

    def spec_fn() -> mujoco.MjSpec:
        return get_primitive_spec(name, **spec_kwargs)

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}),
        spec_fn=spec_fn,
    )


def get_primitive_variant_object_cfg(
    names: Sequence[str],
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    max_collision_hulls: int | None = None,
    assignment=None,
    **spec_kwargs,
):
    """A single ``object`` entity whose *primitive shape varies per world* across ``names``.

    Same per-world mesh-variant mechanics as ``assets.objaverse.get_objaverse_variant_object_cfg``;
    because every primitive's collision hull shares one topology, the variant scene stays fast.
    """
    from mjlab.entity.variants import VariantEntityCfg

    del max_collision_hulls
    if not names:
        raise ValueError("get_primitive_variant_object_cfg needs at least one object name.")

    def make(name: str) -> Callable[[], mujoco.MjSpec]:
        return lambda: get_primitive_spec(name, **spec_kwargs)

    return VariantEntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}),
        variants={name: make(name) for name in names},
        assignment=assignment,
    )
