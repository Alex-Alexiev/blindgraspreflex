"""YCB object assets as free-floating mjlab entities.

Consumes the *prepared* YCB tree -- the one whose meshes have already been moved into their
canonical frame (OBB-aligned, origin at the OBB centre, longest extent on X) and rescaled so
the shortest OBB dimension lands in [0.02, 0.09] m. Each object directory provides:

* ``google_16k/norm_textured.obj``       canonical visual mesh
* ``google_16k/norm_textured_vhacd.obj`` canonical collision mesh, convex-decomposed
* ``<name>.urdf``                        hand-curated mass and diagonal inertia, already
                                         scaled to match the normalised mesh

Convex decomposition
--------------------
``norm_textured_vhacd.obj`` is a *single* OBJ holding many disjoint convex hulls. MuJoCo
treats one mesh asset as one convex hull, so loading the file as-is silently yields the hull
of the entire decomposition -- a bowl becomes a solid lump with no interior and a mug loses
its handle. The parts are therefore split and emitted as one geom each. Hull counts vary
enormously (apple 1, mug 54, bowl 80), which is why :func:`get_ycb_object_cfg` exposes
``max_collision_hulls``: at a few thousand environments the geom count, not the mesh detail,
is what bounds the scene.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
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
    "EASY_YCB_OBJECT_LIST",
    "MEDIUM_YCB_OBJECT_LIST",
    "YCB_OBJECT_LIST",
    "YcbInertial",
    "find_ycb_root",
    "get_ycb_object_cfg",
    "get_ycb_spec",
    "get_ycb_variant_object_cfg",
    "load_ycb_inertial",
]

##
# Object lists. Ported verbatim from the IsaacLab task so the two stay comparable.
##

YCB_OBJECT_LIST = (
    "v002_master_chef_can",
    "v003_cracker_box",
    "v004_sugar_box",
    "v005_tomato_soup_can",
    "v006_mustard_bottle",
    "v007_tuna_fish_can",
    "v008_pudding_box",
    "v009_gelatin_box",
    "v010_potted_meat_can",
    "v011_banana",
    "v012_strawberry",
    "v013_apple",
    "v014_lemon",
    "v015_peach",
    "v016_pear",
    "v017_orange",
    "v018_plum",
    "v019_pitcher_base",
    "v021_bleach_cleanser",
    "v022_windex_bottle",
    "v024_bowl",
    "v025_mug",
    "v026_sponge",
    "v028_skillet_lid",
    "v029_plate",
    "v030_fork",
    "v031_spoon",
    "v032_knife",
    "v033_spatula",
    "v035_power_drill",
    "v036_wood_block",
    "v037_scissors",
    "v038_padlock",
    "v040_large_marker",
    "v042_adjustable_wrench",
    "v043_phillips_screwdriver",
    "v044_flat_screwdriver",
    "v048_hammer",
    "v050_medium_clamp",
    "v051_large_clamp",
    "v052_extra_large_clamp",
    "v053_mini_soccer_ball",
    "v054_softball",
    "v055_baseball",
    "v056_tennis_ball",
    "v057_racquetball",
    "v058_golf_ball",
    "v059_chain",
    "v061_foam_brick",
    "v062_dice",
    "v063a_marbles",
    "v063b_marbles",
    "v065a_cups",
    "v065b_cups",
    "v065c_cups",
    "v065d_cups",
    "v065e_cups",
    "v065f_cups",
    "v065g_cups",
    "v065h_cups",
    "v065i_cups",
    "v065j_cups",
    "v070a_colored_wood_blocks",
    "v070b_colored_wood_blocks",
    "v071_nine_hole_peg_test",
    "v072a_toy_airplane",
    "v072b_toy_airplane",
    "v072c_toy_airplane",
    "v072d_toy_airplane",
    "v072e_toy_airplane",
    "v073a_lego_duplo",
    "v073b_lego_duplo",
    "v073c_lego_duplo",
    "v073d_lego_duplo",
    "v073e_lego_duplo",
    "v073f_lego_duplo",
    "v073g_lego_duplo",
    "v077_rubiks_cube",
)

# Round, chunky, no thin features -- the subset a five-finger hand can close on from any
# approach direction.
EASY_YCB_OBJECT_LIST = (
    "v013_apple",
    "v014_lemon",
    "v015_peach",
    "v016_pear",
    "v017_orange",
    "v018_plum",
    "v053_mini_soccer_ball",
    "v054_softball",
    "v055_baseball",
    "v056_tennis_ball",
    "v057_racquetball",
)

# Everything except objects that are too thin, too small, or degenerate to grasp: cutlery,
# flat plates and lids, markers, wrenches, chains, dice and loose marbles.
_HARD_YCB_OBJECTS = frozenset(
    {
        "v028_skillet_lid",
        "v029_plate",
        "v030_fork",
        "v031_spoon",
        "v032_knife",
        "v033_spatula",
        "v037_scissors",
        "v040_large_marker",
        "v042_adjustable_wrench",
        "v044_flat_screwdriver",
        "v059_chain",
        "v062_dice",
        "v063a_marbles",
        "v063b_marbles",
        "v071_nine_hole_peg_test",
    }
)

MEDIUM_YCB_OBJECT_LIST = tuple(n for n in YCB_OBJECT_LIST if n not in _HARD_YCB_OBJECTS)

##
# Dataset location.
##


def find_ycb_root() -> Path:
    """Locate the prepared YCB tree.

    Checked in order: ``BLINDGRASPREFLEX_YCB_ROOT`` and ``<repo>/datasets/ycb``. The dataset is
    not distributed with this package; set the environment variable or configure it in the local
    TOML file.
    """
    candidates = []
    if env_root := os.environ.get("BLINDGRASPREFLEX_YCB_ROOT"):
        candidates.append(Path(env_root))
    candidates.append(BLINDGRASPREFLEX_ROOT_PATH / "datasets" / "ycb")
    for candidate in candidates:
        if (candidate / "v013_apple" / "google_16k" / "norm_textured.obj").exists():
            return candidate
    raise FileNotFoundError(
        "Prepared YCB dataset not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Set BLINDGRASPREFLEX_YCB_ROOT or run scripts/fetch_ycb.py."
    )


##
# Inertial properties.
##


@dataclass(frozen=True)
class YcbInertial:
    """Mass and diagonal inertia read from the prepared URDF, in the canonical mesh frame."""

    mass: float
    diaginertia: tuple[float, float, float]
    com: tuple[float, float, float]


def load_ycb_inertial(name: str, ycb_root: Path | None = None) -> YcbInertial:
    """Read mass/inertia from the object's URDF.

    Read rather than recomputed from the mesh: the values are hand-curated per object and
    already carry the mesh normalisation's ``s^3`` mass and ``s^5`` inertia scaling. Letting
    MuJoCo infer inertia from geometry would instead assume uniform density and silently
    disagree with the IsaacLab task.
    """
    root = ycb_root if ycb_root is not None else find_ycb_root()
    inertial = ET.parse(root / f"{name}.urdf").getroot().find(".//link/inertial")
    if inertial is None:
        raise ValueError(f"No <inertial> block in {name}.urdf")
    inertia = inertial.find("inertia")
    origin = inertial.find("origin")
    mass_el = inertial.find("mass")
    if inertia is None or mass_el is None:
        raise ValueError(f"Incomplete <inertial> block in {name}.urdf")
    com = (0.0, 0.0, 0.0)
    if origin is not None and (xyz := origin.get("xyz")):
        com = tuple(float(v) for v in xyz.split())  # type: ignore[assignment]
    return YcbInertial(
        mass=float(mass_el.attrib["value"]),
        diaginertia=(
            float(inertia.attrib["ixx"]),
            float(inertia.attrib["iyy"]),
            float(inertia.attrib["izz"]),
        ),
        com=com,
    )


@lru_cache(maxsize=32)
def _load_meshes(name: str, root_str: str) -> tuple[trimesh.Trimesh, tuple[trimesh.Trimesh, ...]]:
    """Return (visual mesh, per-hull collision meshes). Cached: these are ~10k-vertex loads."""
    mesh_dir = Path(root_str) / name / "google_16k"
    visual = trimesh.load(mesh_dir / "norm_textured.obj", force="mesh", process=False)
    collision = trimesh.load(mesh_dir / "norm_textured_vhacd.obj", force="mesh", process=False)
    hulls = collision.split(only_watertight=False)
    if len(hulls) == 0:  # single-component decomposition comes back unsplit
        hulls = [collision]
    # Drop degenerate slivers. VHACD leaves components that are not solids -- v065c_cups has one
    # with fewer than 4 vertices -- and MuJoCo refuses to compile them ("at least 4 vertices
    # required"). At a hull budget of 16 this never surfaced, because the budget keeps the largest
    # hulls by VOLUME and the degenerates are the smallest things in the file; raising the budget
    # walks straight into them. Filtering here rather than at the budget means every caller gets
    # hulls that compile, and the budget only ever has to decide how much detail to keep.
    viable = [h for h in hulls if len(h.vertices) >= 4 and h.volume > 0.0]
    if not viable:  # pathological object: fall back to the whole decomposition's hull
        viable = [collision.convex_hull]
    return visual, tuple(viable)


##
# Spec construction.
##


def add_ycb_object(
    spec: mujoco.MjSpec,
    name: str,
    ycb_root: Path | None = None,
    body_name: str = "object",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.85, 0.55, 0.2, 1.0),
    max_collision_hulls: int | None = None,
    free: bool = True,
    scale: float = 1.0,
):
    """Add one YCB object to an existing spec and return its body.

    The prepared tree bakes in the canonical transform + size normalisation, so the body frame
    is the OBB centre and no further transform is applied. ``scale`` uniformly resizes the object:
    the mesh vertices are multiplied by ``scale`` and the explicit inertial block is scaled to
    match a uniform-density rescale (mass x scale^3, diagonal inertia x scale^5, COM x scale).
    """
    root = ycb_root if ycb_root is not None else find_ycb_root()
    if name not in YCB_OBJECT_LIST:
        raise ValueError(f"Unknown YCB object {name!r}.")
    visual, hulls = _load_meshes(name, str(root))
    inertial = load_ycb_inertial(name, root)

    if max_collision_hulls is not None and len(hulls) > max_collision_hulls:
        # Keep the largest hulls by volume: dropping the small ones loses surface detail but
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

    # Inertia is set explicitly from the URDF rather than inferred from the hulls, scaled to the
    # uniform-density rescale (mass ~ s^3, inertia ~ s^5, COM ~ s).
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
            # from also contributing, which would otherwise double-count and, because hulls
            # overlap, overshoot the true mass.
            density=0.0,
            # condim 6 to match the hand's grasp surfaces -- without torsional and rolling
            # friction a grasped object spins freely about the contact normals.
            condim=6,
            friction=(1.0, 5e-3, 1e-4),
            # Stiff contact so fast-closing fingers don't visibly sink into the object. The
            # MuJoCo default solref (0.02 s) is ~5 steps at 250 Hz, soft enough that the finger
            # penetrates before the contact force builds -- the "hand going inside" clipping,
            # and the likely source of the deep-penetration force spikes that NaN'd training.
            # 0.01 s (matches the mat) plus a near-rigid solimp (dmax 0.99) resolves it without
            # the ~14x throughput hit of raising physics to 500 Hz. priority 2 makes these
            # params win the hand<->object pair regardless of the hand geoms' settings.
            solref=(0.01, 1.0),
            # Stiff contact -- 0.01 s is ~2.5 physics steps at 250 Hz, near the stable minimum of
            # 2*dt. This cut the worst finger-into-object clipping, but ~2-3 mm of penetration
            # remains under a firm grip and is a *physics-rate floor*, not a stiffness one: at
            # 250 Hz the contact time-constant can't go below ~2*dt, and raising solimp past 0.99
            # was measured to do nothing. Reduce it only with a higher physics rate or a softer
            # grip. A contact margin (the gentler fix) is unusable -- mjwarp rejects non-zero
            # margin on mesh pairs with MULTICCD. The instability termination + SafeVecEnvWrapper
            # catch the rare blow-up this stiffness can trigger.
            solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
            priority=2,
        )
    return body


def get_ycb_spec(name: str, **kwargs) -> mujoco.MjSpec:
    """Build a standalone spec holding one free-floating YCB object."""
    spec = mujoco.MjSpec()
    add_ycb_object(spec, name, **kwargs)
    return spec


def get_ycb_object_cfg(
    name: str,
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    ycb_root: Path | None = None,
    max_collision_hulls: int | None = None,
    **spec_kwargs,
) -> EntityCfg:
    """mjlab entity config for a single free-floating YCB object."""

    def spec_fn() -> mujoco.MjSpec:
        return get_ycb_spec(
            name, ycb_root=ycb_root, max_collision_hulls=max_collision_hulls, **spec_kwargs
        )

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=pos, joint_pos={}, joint_vel={}),
        spec_fn=spec_fn,
    )


def get_ycb_variant_object_cfg(
    names: Sequence[str],
    pos: tuple[float, float, float] = (0.55, 0.0, 0.1),
    ycb_root: Path | None = None,
    max_collision_hulls: int | None = None,
    assignment=None,
    **spec_kwargs,
):
    """A single ``object`` entity whose *mesh varies per world* across ``names``.

    Uses mjlab's per-world mesh variants: every world shares one compiled scene and the same
    kinematic structure (one free body named ``object``), but different worlds simulate
    different YCB meshes -- and, since mass/inertia are variant-dependent fields, the correct
    per-object mass too. Because the body name and joint are identical across variants, every
    observation, reward, contact-sensor match and reset that references ``object`` works
    unchanged; only the geometry the hand actually touches differs between worlds.

    The merged model is padded to the *largest* variant's geom count, not the sum, so adding
    objects costs almost nothing in model size as long as ``max_collision_hulls`` is bounded.
    All variants share one inertial representation (diagonal, from each URDF).

    ``assignment`` is passed through to ``VariantEntityCfg``: ``None`` spreads worlds evenly
    over the objects; a ``{name: weight}`` dict biases the split; a callable gives an explicit
    per-world index list.
    """
    from mjlab.entity.variants import VariantEntityCfg

    if not names:
        raise ValueError("get_ycb_variant_object_cfg needs at least one object name.")

    import os

    # Optional per-world size domain randomization. REFLEXIVE_OBJECT_SCALES="0.6,0.7,0.8,0.9,1.0"
    # turns each (object, scale) into its own per-world variant, so worlds simulate the objects at
    # a range of sizes. Note: variant assignment is fixed at sim init -- a world keeps its scale
    # for the whole run (per-world DR, not per-episode). The (object x scale) product grows the
    # variant *count*, not the merged model size (padded to the largest variant's geom count).
    _scales_env = os.environ.get("REFLEXIVE_OBJECT_SCALES", "").strip()
    scales = tuple(float(s) for s in _scales_env.split(",") if s) if _scales_env else (1.0,)

    def make(name: str, s: float) -> Callable[[], mujoco.MjSpec]:
        # Bind name + scale per-variant; each callable builds that object's standalone spec.
        return lambda: get_ycb_spec(
            name, ycb_root=ycb_root, max_collision_hulls=max_collision_hulls, scale=s, **spec_kwargs
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


# The 21 objects of the YCB-VIDEO subset -- the most widely cited "YCB objects" set in the
# manipulation literature, and what published success rates are usually quoted over. Chosen by
# Xiang et al. from the 77-object YCB set for model quality and depth coverage.
#
# Two caveats worth keeping in mind when quoting a number against other papers:
#   * the subset was selected for 6-D POSE ESTIMATION, not graspability, so it is not curated for
#     what a five-finger hand can pick up -- it includes a power drill and scissors;
#   * the grasping-specific alternative is the 28-object YCB Gripper Assessment Benchmark, which is
#     less often used as a policy evaluation set.
# Using it as the TEST set keeps our success rate comparable; training still sees all 78.
YCB_VIDEO_21 = (
    "v002_master_chef_can",
    "v003_cracker_box",
    "v004_sugar_box",
    "v005_tomato_soup_can",
    "v006_mustard_bottle",
    "v007_tuna_fish_can",
    "v008_pudding_box",
    "v009_gelatin_box",
    "v010_potted_meat_can",
    "v011_banana",
    "v019_pitcher_base",
    "v021_bleach_cleanser",
    "v024_bowl",
    "v025_mug",
    "v035_power_drill",
    "v036_wood_block",
    "v037_scissors",
    "v040_large_marker",
    "v051_large_clamp",
    "v052_extra_large_clamp",
    "v061_foam_brick",
)
