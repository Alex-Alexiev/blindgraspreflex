"""Where the arm is bolted down: one robot, one hand, one checkpoint, six base transforms.

Why this module exists
----------------------
The grasp policy this repo trains is *arm-agnostic by construction*. The distilled student
(``benchmark/checkpoints/reflex_proj05``) observes 20 hand joint positions x a 5-step history --
100 numbers, none of which is an arm joint, an arm command, a palm pose or an object pose -- and
emits a grasp score plus 20 finger targets. The arm is driven by a scripted phase machine
(REACH standoff -> LIFT retreat) through a Cartesian velocity field, a damped Jacobian
pseudoinverse and a joint PD. The only thing crossing from hand to arm is the scalar score that
latches the phase; nothing crosses back.

If that factorisation is real, then *how the arm is mounted* is a property of the arm, and the
hand policy should not be able to tell. This module is the experiment: it re-bolts the identical
Rizon 4 to the ceiling, to a wall, to the far side of the table and to a tilted pedestal, and
re-runs the identical checkpoint.

    REFLEXIVE_MOUNT=floor      upright at the table edge -- where every policy here trained
    REFLEXIVE_MOUNT=opposite   translated across the workspace and yawed 180 deg
    REFLEXIVE_MOUNT=left       yawed 90 deg, reaching sideways across the table
    REFLEXIVE_MOUNT=ceiling    inverted overhead, hanging down at the table
    REFLEXIVE_MOUNT=wall       rolled 90 deg off a vertical wall, base axis horizontal
    REFLEXIVE_MOUNT=pedestal   raised and tilted 30 deg -- an off-axis, deliberately awkward mount

Relation to :mod:`blindgraspreflex.assets.arms`
------------------------------------------
That module changes *which arm* carries the hand; this one changes *where that arm is bolted*.
They compose and are applied in that order: the arm registry produces the robot config, and
:func:`remount` then overwrites its base pose and arm home joints. A home pose therefore belongs
to an (arm, mount) PAIR, and ``mount_homes.json`` is keyed ``"<arm>/<mount>"`` -- the joint vector
that puts the Rizon's palm on target from the ceiling says nothing about where the UR5e's should
be. The floor mount is a strict no-op, so every existing arm, script and number is untouched
unless ``REFLEXIVE_MOUNT`` is set.

What is held identical across mounts
------------------------------------
Everything the policy or the reward can see:

* the robot -- same MJCF, same hand, same joint limits, same PD gains, same effort caps;
* **the home palm pose in the world**. ``mount_homes.json`` stores a per-mount 7-joint vector
  solved by IK (``scripts/fit_mount_home.py``) so that ``grasp_site`` lands on the *floor* mount's
  home pose to well under a millimetre and a tenth of a degree. That matters more than it looks:
  ``GraspAction`` takes the LIFT/hold goal AND the held palm attitude from FK of the home pose,
  and the phase controller builds every REACH target as (object position + a world offset) with
  an attitude relative to that same held attitude. Pin the home palm pose and the arm is handed
  *the identical sequence of world-frame setpoints* on every mount -- so the success criterion,
  the hold goal and the pre-grasp pose distribution all mean exactly the same thing;
* the table, the mat, the spawn point, the object set, the DR and the episode length, which live
  in the task and never look at the robot.

What necessarily differs: the base transform, and therefore the joint configuration the arm
holds, the Jacobian it inverts, the gravity load on every joint, and the null space it has left.

The mount fixtures (the plate under the base, the ceiling slab, the wall panel) are **visual
only** -- ``contype=0 conaffinity=0``. They are there so a rendering reads as a ceiling mount
rather than a floating arm. Making them collidable would add a scene difference that is not the
base transform, and any change in the numbers could then be blamed on the furniture.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

HOMES_JSON = Path(__file__).parent / "mount_homes.json"
"""Per-mount home joint vectors, solved once by scripts/fit_mount_home.py. Tracked in git: it is
a few hundred bytes, and a home pose that silently changed would move the lift goal and quietly
invalidate every cross-mount number."""

MOUNT_ENV_VAR = "REFLEXIVE_MOUNT"
DEFAULT_MOUNT = "floor"


##
# Quaternion helpers (wxyz, MuJoCo convention). Small enough not to import mujoco for.
##


def _axis_quat(axis: int, angle: float) -> tuple[float, float, float, float]:
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = math.cos(angle / 2.0)
    q[1 + axis] = math.sin(angle / 2.0)
    return _clean(q)


def _clean(q) -> tuple[float, float, float, float]:
    """Snap the float dust a right-angle composition leaves behind (``cos(pi/2) = 6.1e-17``).

    Cosmetic, but these quaternions are written into an MJCF, echoed into results metadata and
    read by a person checking a mount is what it claims to be; ``6.1e-17`` in that column is
    noise to read past.
    """
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in q)  # type: ignore[return-value]


def _qmul(a, b) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _clean(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _qrot(q, v) -> tuple[float, float, float]:
    w, x, y, z = q
    tx, ty, tz = (
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    )
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


##
# Registry.
##


@dataclass(frozen=True)
class MountSpec:
    """One way of bolting the arm down."""

    key: str
    label: str
    """Name for tables and plots."""
    base_pos: tuple[float, float, float]
    base_quat: tuple[float, float, float, float]
    """ABSOLUTE base pose of the robot root, in the world/env frame. Written straight into
    ``EntityCfg.InitialStateCfg`` -- mjlab wraps a fixed-base entity in a mocap body and the
    task's ``reset_robot_base`` event writes this pose into it every reset."""
    blurb: str
    panel: (
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float, float],
        ]
        | None
    ) = None
    """Optional visual-only slab (half-extent, world position, world quaternion) standing in for
    the ceiling or wall the base is bolted to. ``None`` -> just the plate under the base."""

    @property
    def is_default(self) -> bool:
        return self.key == DEFAULT_MOUNT

    def q_home(self, arm_key: str) -> dict[str, float]:
        """Home joint angles [rad] for this mount carrying arm ``arm_key``.

        Keyed by the pair: the base transform is what the IK solved against, so a home fitted
        for one arm is meaningless on another.
        """
        homes = _load_homes()
        entry = homes.get(f"{arm_key}/{self.key}")
        if entry is None:
            raise RuntimeError(
                f"No home pose for arm {arm_key!r} on mount {self.key!r} in {HOMES_JSON}. "
                f"Run: uv run python scripts/fit_mount_home.py --arm {arm_key} "
                f"--mount {self.key}"
            )
        return {n: float(v) for n, v in entry.items()}


# Base poses. The workspace they all have to serve is fixed by the task: the mat spans
# x 0.08-1.02, y -0.43-0.43 at z = -0.03, objects spawn at (0.55, 0.15), the home palm pose is
# (0.522, 0.127, 0.213) and the lift/hold goal 0.30 m above it. Each pose below was checked by
# scripts/fit_mount_home.py, which refuses a mount whose home cannot be reached exactly or whose
# home configuration cannot then reach the REACH/LIFT workspace.
MOUNTS: dict[str, MountSpec] = {
    "floor": MountSpec(
        key="floor",
        label="Floor (trained)",
        base_pos=(0.0, 0.0, 0.0),
        base_quat=(1.0, 0.0, 0.0, 0.0),
        blurb=(
            "The reference. Upright at the near edge of the table, exactly as every policy in "
            "this repo was trained and every other number was produced."
        ),
    ),
    "opposite": MountSpec(
        key="opposite",
        label="Opposite side",
        base_pos=(1.06, 0.28, 0.0),
        base_quat=_axis_quat(2, math.pi),
        blurb=(
            "Same upright bolt pattern, moved across the workspace and yawed 180 deg, so the arm "
            "reaches the identical objects from the far side of the table. Pure change of base "
            "translation plus yaw: the easiest mount, and the one that isolates whether anything "
            "in the pipeline was quietly anchored to the world x axis."
        ),
    ),
    "left": MountSpec(
        key="left",
        label="Side (yaw 90 deg)",
        base_pos=(0.45, -0.60, 0.0),
        base_quat=_axis_quat(2, math.pi / 2),
        blurb=(
            "Bolted to the -y side of the table and yawed 90 deg, so the whole reach is lateral "
            "and the wrist has to present the same palm attitude from an orthogonal approach. "
            "Pulled in from (0.55, -0.80): at 0.94 m the palm home pose is near the Rizon's reach "
            "limit and the fit could serve only 71% of the commanded REACH/LIFT workspace, which "
            "would have cost this mount episodes for a reason that is about the arm's reach and "
            "not about the hand policy."
        ),
    ),
    "ceiling": MountSpec(
        key="ceiling",
        label="Ceiling (inverted)",
        base_pos=(0.30, 0.15, 1.25),
        base_quat=_qmul(_axis_quat(2, math.pi), _axis_quat(0, math.pi)),
        blurb=(
            "Hung upside down from an overhead slab. The base z axis points at the floor, so "
            "every joint carries gravity with the opposite sign to the one it was tuned under "
            "and the elbow works above the palm instead of below it."
        ),
        panel=((0.45, 0.45, 0.02), (0.30, 0.15, 1.30), (1.0, 0.0, 0.0, 0.0)),
    ),
    "wall": MountSpec(
        key="wall",
        label="Wall (roll 90 deg)",
        base_pos=(-0.20, 0.15, 0.80),
        base_quat=_axis_quat(1, math.pi / 2),
        blurb=(
            "Bolted to a vertical wall behind the table with the base axis horizontal, 0.80 m up. "
            "Gravity now acts sideways across the shoulder rather than along it."
        ),
        panel=((0.02, 0.55, 0.62), (-0.25, 0.15, 0.62), (1.0, 0.0, 0.0, 0.0)),
    ),
    "right": MountSpec(
        key="right",
        label="Side (yaw -90 deg)",
        base_pos=(0.45, 0.60, 0.0),
        base_quat=_axis_quat(2, -math.pi / 2),
        blurb=(
            "The mirror of the side mount, bolted to the +y edge and yawed the other way. Worth "
            "having as a pair because the arm's wrist range is NOT symmetric, so approaching the "
            "same object from the two sides is not the same problem for the arm even though it is "
            "the same problem for the hand."
        ),
    ),
    "overhead_tilt": MountSpec(
        key="overhead_tilt",
        label="Overhead bracket (tilted)",
        base_pos=(0.25, 0.05, 1.15),
        base_quat=_qmul(
            _qmul(_axis_quat(2, math.pi), _axis_quat(0, math.pi)), _axis_quat(0, math.radians(40.0))
        ),
        blurb=(
            "Inverted like the ceiling mount but on a bracket raked 40 deg, so the arm hangs at a "
            "slant rather than straight down -- the overhead case as it is actually installed when "
            "the ceiling is not directly above the work."
        ),
        panel=(
            (0.30, 0.42, 0.02),
            (0.25, 0.05, 1.19),
            _qmul(_axis_quat(0, math.radians(40.0)), (1.0, 0.0, 0.0, 0.0)),
        ),
    ),
    "pedestal": MountSpec(
        key="pedestal",
        label="Tilted pedestal",
        base_pos=(0.10, -0.30, 0.35),
        base_quat=_qmul(_axis_quat(2, math.radians(35.0)), _axis_quat(0, math.radians(30.0))),
        blurb=(
            "Raised 0.35 m and tilted 30 deg about x on top of a 35 deg yaw -- an off-axis mount "
            "aligned with no world axis at all, which is the case a pipeline with a hidden "
            "assumption about 'up' or 'forward' is most likely to fail."
        ),
        # A column from the table surface up to the base, so the arm is standing on something.
        panel=((0.06, 0.06, 0.19), (0.10, -0.30, 0.16), (1.0, 0.0, 0.0, 0.0)),
    ),
}

ORDER = ("floor", "opposite", "left", "right", "ceiling", "overhead_tilt", "wall", "pedestal")
"""Presentation order: the trained mount first, then increasing departure from it."""


def _load_homes() -> dict[str, dict[str, float]]:
    """``{"<arm>/<mount>": {joint: angle}}`` as written by scripts/fit_mount_home.py."""
    if not HOMES_JSON.exists():
        return {}
    return json.loads(HOMES_JSON.read_text())["homes"]


def resolve(key: str | None) -> MountSpec:
    """Look one mount up by key, with a listing in the error rather than a KeyError."""
    key = (key or DEFAULT_MOUNT).strip().lower()
    if key not in MOUNTS:
        raise SystemExit(f"Unknown mount {key!r}; choose from {list(MOUNTS)}")
    return MOUNTS[key]


def active() -> MountSpec:
    """The mount selected by ``REFLEXIVE_MOUNT``, defaulting to the trained floor mount.

    Read at import of :mod:`blindgraspreflex.assets.rizon4_hand`, so it has to be set BEFORE the
    task module is imported -- the same rule ``REFLEXIVE_HAND`` and ``REFLEXIVE_OBJECT_SCALES``
    already follow.
    """
    return resolve(os.environ.get(MOUNT_ENV_VAR))


def arm_q_home(
    arm_key: str,
    joint_names: tuple[str, ...],
    default_home: dict[str, float],
    mount: MountSpec | None = None,
) -> dict[str, float]:
    """The arm home pose for ``arm_key`` on the active mount.

    Returns ``default_home`` unchanged on the floor mount, so a run that never sets
    ``REFLEXIVE_MOUNT`` sees exactly the pose the arm registry gave it.
    """
    mount = active() if mount is None else mount
    if mount.is_default:
        return dict(default_home)
    q = mount.q_home(arm_key)
    missing = [n for n in joint_names if n not in q]
    if missing:
        raise RuntimeError(
            f"{HOMES_JSON} gives {arm_key}/{mount.key} no angle for {missing}. "
            f"Re-run: uv run python scripts/fit_mount_home.py --arm {arm_key} "
            f"--mount {mount.key}"
        )
    return {n: float(q[n]) for n in joint_names}


def remount(cfg, mount: MountSpec | None = None, q_home: dict[str, float] | None = None):
    """Re-bolt an arm's ``EntityCfg`` onto ``mount``, in place, and return it.

    Two fields change and nothing else: ``init_state.pos``/``rot`` (mjlab writes these into the
    mocap body that carries a fixed-base entity, so this IS the base transform) and the arm
    entries of ``init_state.joint_pos``. The hand's entries are left exactly as the arm registry
    set them -- a mount changes where the arm is bolted, never how the hand starts.

    A no-op on the floor mount, by identity rather than by producing the same values: the
    default path must not depend on this function being correct.
    """
    mount = active() if mount is None else mount
    if mount.is_default:
        return cfg
    from mjlab.entity import EntityCfg

    # The Rizon MJCF hangs a ~1 x 1.9 x 1.9 m ``enclosure`` mesh off the robot ROOT -- the lab
    # cage the real arm stands in. It rides the base transform like any other child body, so on
    # the ceiling mount the cage is on the ceiling too, and every rendering of a non-floor mount
    # is a room drawn upside down around the arm. It is already ``contype=0`` (visual only), so
    # hiding it cannot touch the physics; see _hide_enclosure.
    _spec_fn = cfg.spec_fn
    cfg.spec_fn = lambda: _hide_enclosure(_spec_fn())

    base = cfg.init_state
    joint_pos = dict(base.joint_pos or {})
    joint_pos.update(q_home or {})
    cfg.init_state = EntityCfg.InitialStateCfg(
        pos=tuple(float(v) for v in mount.base_pos),
        rot=tuple(float(v) for v in mount.base_quat),
        joint_pos=joint_pos,
        joint_vel=base.joint_vel,
    )
    return cfg


##
# Scene fixtures (visual only -- see the module docstring).
##

_PLATE_HALF = (0.10, 0.10, 0.015)
_PLATE_RGBA = (0.32, 0.34, 0.38, 1.0)
_PANEL_RGBA = (0.55, 0.56, 0.60, 1.0)


def add_fixture(spec, mount: MountSpec | None = None) -> None:
    """Draw the active mount's bolt plate (and ceiling/wall slab) into a worldbody spec.

    Called from the task's ``add_ground``. Every geom here is ``contype=0 conaffinity=0``: the
    physics of the scene must be identical across mounts so that a difference in the numbers can
    only come from the base transform.
    """
    import mujoco

    mount = active() if mount is None else mount

    # Bolt plate, in the base's own frame, just behind the base origin along its -z.
    back = _qrot(mount.base_quat, (0.0, 0.0, -_PLATE_HALF[2]))
    spec.worldbody.add_geom(
        name="mount_plate",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=_PLATE_HALF,
        pos=tuple(p + b for p, b in zip(mount.base_pos, back, strict=True)),
        quat=mount.base_quat,
        contype=0,
        conaffinity=0,
        group=1,
        rgba=_PLATE_RGBA,
    )
    if mount.panel is not None:
        half, pos, quat = mount.panel
        spec.worldbody.add_geom(
            name="mount_panel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half,
            pos=pos,
            quat=quat,
            contype=0,
            conaffinity=0,
            group=1,
            rgba=_PANEL_RGBA,
        )


_ENCLOSURE_GEOMS = ("enclosure_visual",)


def _hide_enclosure(spec):
    """Make the arm's static lab-cage mesh invisible, in place, and return the spec.

    Alpha, not deletion: removing a geom renumbers every geom after it, and this project has at
    least two consumers that address geoms by index (``mjlab_patches.disable_padded_variant_geoms``
    and the contact sensors' geom sets). The geom is already non-colliding, so zero alpha is the
    whole change -- no mass, no contact, no index moves.
    """
    for geom in spec.geoms:
        if geom.name in _ENCLOSURE_GEOMS:
            rgba = list(geom.rgba)
            rgba[3] = 0.0
            geom.rgba = rgba
    return spec


##
# Camera framing for renderings.
##

CAM_ELEVATION = -22.0
CAM_MARGIN = 1.0
_LOOKAT_BIAS = 0.38
"""How far from the object toward the base the camera looks; 0.5 would be the plain midpoint."""
"""Metres of padding added to the base-to-object separation when choosing camera distance."""


def camera(
    mount: MountSpec | None = None,
    target: tuple[float, float, float] = (0.55, 0.15, 0.06),
    goal: tuple[float, float, float] | None = (0.52, 0.13, 0.51),
):
    """``(distance, azimuth_deg, elevation_deg, lookat)`` framing this mount's arm and its work.

    Derived from the mount geometry rather than hand-set per mount, so a mount whose base pose is
    retuned reframes itself instead of quietly drifting out of shot. Two things have to be in the
    picture -- where the arm is bolted, and the object it is picking up -- so the camera looks at
    the midpoint of the two and stands off far enough to contain both, viewing the base-to-object
    line side-on (azimuth perpendicular to it) rather than down its length, which would foreshorten
    the whole arm into a blob.
    """
    mount = active() if mount is None else mount
    base = mount.base_pos
    # Weighted toward the object rather than the midpoint: both have to be in shot, but the
    # grasp is what the clip is of and the base is context, so it gets the smaller share of frame.
    mid = tuple(t + _LOOKAT_BIAS * (b - t) for b, t in zip(base, target, strict=True))
    dx, dy = target[0] - base[0], target[1] - base[1]
    sep = math.dist(base, target)
    if goal is not None:
        # The LIFT goal has to be in frame too. Framing only the base and the object crops the
        # arm off the top of the picture at exactly the moment it is holding the object up, which
        # is the frame the figure is actually about.
        mid = (mid[0], mid[1], (mid[2] + goal[2]) / 2.0)
        sep = max(sep, math.dist(base, goal), math.dist(target, goal) * 1.6)
    if math.hypot(dx, dy) < 1e-6:  # bolted directly above/below the work (the ceiling mount)
        azimuth = 140.0
    else:
        azimuth = math.degrees(math.atan2(dy, dx)) + 90.0
    return sep * 1.05 + CAM_MARGIN, azimuth, CAM_ELEVATION, mid


def brighten(mj_model, ambient: float = 0.42, diffuse: float = 0.62) -> None:
    """Turn the headlight up on a host model before rendering.

    The task scene carries no lights of its own, so MuJoCo's default headlight is all there is and
    the stock intensity renders a near-black frame. Visual only -- ``mjvisual`` fields play no part
    in the dynamics.
    """
    mj_model.vis.headlight.ambient[:] = ambient
    mj_model.vis.headlight.diffuse[:] = diffuse
    mj_model.vis.headlight.specular[:] = 0.1


def hide_enclosure_geoms(mj_model) -> int:
    """Zero the alpha of the arm's lab-cage geoms on a compiled HOST model, for rendering.

    :func:`remount` already does this in the spec for non-floor mounts, where the cage would be
    upside down or sideways. A rendering wants it gone on the floor mount too: it is a 1 x 1.9 x
    1.9 m mesh that fills the frame and hides the thing the clip is about. Done here rather than in
    the spec so the simulated model on the default mount is left exactly as it was.
    """
    import mujoco

    # mjlab namespaces every scene element by entity name, so the geom is "robot/enclosure_visual"
    # in a built scene and "enclosure_visual" in a standalone entity compile. Both are tried
    # rather than assumed, because a miss here fails silently -- as a cage in front of the shot.
    hidden = 0
    for base in _ENCLOSURE_GEOMS:
        for name in (base, f"robot/{base}"):
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                mj_model.geom_rgba[gid, 3] = 0.0
                hidden += 1
    return hidden


##
# Studio rendering (figures only -- never on in a training or eval run).
##

STUDIO_ENV_VAR = "REFLEXIVE_STUDIO_RENDER"

# The real Rizon 4 is an off-white arm with dark joint rings on a dark base; the RH-5 hand is
# near-black. The shared MJCF ships the arm's visual meshes with no material at all, so they
# compile to a flat 0.5 grey -- correct geometry, no identity. These are the colours the renders
# use; the hand's own 0.2 grey already matches the hardware and is left alone.
_TABLE_THICKNESS = 0.055
"""Visible thickness of the studio slab [m]. The task's mat is 1 cm and reads as paper."""
_TABLE_RGBA = (0.62, 0.60, 0.55, 1.0)
"""The obstacle figure's table tone -- a warm grey that separates from both the white arm and the
black background."""
_WORK_POINT = (0.55, 0.15, 0.06)
_LIGHT_AZIMUTH_OFFSET = 22.0
"""Degrees the key light sits round from the camera -- enough to model the arm, not so much that
it lights it from behind. At 52 deg elevation a fixture throws its shadow about 0.8x its own
height sideways, which clears the table."""
_LIGHT_ELEVATION = 52.0
_LIGHT_RANGE = 3.0

_ARM_SHELL_RGBA = (0.93, 0.93, 0.94, 1.0)
_ARM_TRIM_RGBA = (0.22, 0.23, 0.25, 1.0)
_TRIM_MESHES = ("ring", "ring1", "link0")
"""Meshes that are dark on the hardware: the joint rings and the base casting."""


def studio_enabled() -> bool:
    return bool(os.environ.get(STUDIO_ENV_VAR))


def studio_scene(spec) -> None:
    """Studio scene for figures: ONE table, ONE light on the arm, no ground plane, no skybox.

    Matches the look of the obstacle-avoidance figure (``scripts/paper_fig_obstacles.py``, frames
    rendered by the deploy repo's ``obstacle_demo --figure-clean``): the robot and its work lit
    against black, standing on a single slab, with the cell scenery hidden rather than deleted so
    the geometry that was SIMULATED is untouched.

    Figure rendering only, gated on ``REFLEXIVE_STUDIO_RENDER``. Everything here is visual -- a
    non-colliding slab, a light, and materials on geoms that already exist -- so physics is
    identical with it on or off. It has to happen at spec level because MuJoCo cannot gain a light
    after compilation.
    """
    import mujoco

    mount = active()

    # ONE table. The task's ground is a 3 m plane at FLOOR_HEIGHT with a 1 cm mat on top of it --
    # two surfaces a centimetre apart, which read as a sheet of paper on a floor rather than as a
    # table. Both are hidden (studio_model) and replaced by this single slab, whose TOP FACE sits
    # exactly at the mat's top so nothing appears to hover: the objects still rest on the mat's
    # collision surface, which is now invisible and in exactly the same place.
    from blindgraspreflex.tasks.grasping.mjlab_task import (
        MAT_CENTRE_XY,
        MAT_HALF_EXTENT,
        TABLE_HEIGHT,
    )

    half_t = _TABLE_THICKNESS / 2.0
    spec.worldbody.add_geom(
        name="studio_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(MAT_HALF_EXTENT[0], MAT_HALF_EXTENT[1], half_t),
        pos=(MAT_CENTRE_XY[0], MAT_CENTRE_XY[1], TABLE_HEIGHT - half_t),
        contype=0,
        conaffinity=0,
        group=1,
        rgba=_TABLE_RGBA,
    )

    # ONE light, placed on the ARM rather than at a fixed point in the room. The obstacle figure
    # can hard-code a light because its arm never moves; here the base is on the floor in one
    # render and on the ceiling in the next, and a fixed overhead light leaves the wall and
    # ceiling mounts lit from behind. Anchoring it above the midpoint of base and work keeps the
    # same key light on the arm at every mount.
    # Aimed from the camera's shoulder at _LIGHT_ELEVATION above the horizon rather than from
    # straight overhead. Overhead is the obvious choice and it is wrong here: the ceiling slab and
    # the wall panel sit directly over the work, so a near-vertical key drops their shadow onto
    # exactly the patch of table the grasp happens on. At 48 deg the same fixture throws its
    # shadow roughly its own height away, which is off the table entirely.
    _dist, az, _el, _la = camera(mount)
    a = math.radians(az + _LIGHT_AZIMUTH_OFFSET)
    e = math.radians(_LIGHT_ELEVATION)
    ux, uy, uz = math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)
    tx, ty, tz = _WORK_POINT
    spec.worldbody.add_light(
        pos=[tx + ux * _LIGHT_RANGE, ty + uy * _LIGHT_RANGE, tz + uz * _LIGHT_RANGE],
        dir=[-ux, -uy, -uz],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=1,
        diffuse=[0.88, 0.88, 0.88],
        specular=[0.30, 0.30, 0.30],
    )


def studio_model(mj_model) -> None:
    """Post-compile visual fixes for a figure render: arm colours, hidden scenery, soft ambient.

    Colour has to be applied here rather than in the spec because the arm's visual geoms are
    grafted in from the shared hardware MJCF, which this project does not edit.
    """
    import mujoco

    # Hide the two task surfaces the studio slab replaces. Alpha, not deletion: the mat is what
    # the objects actually rest on, and removing it would change the simulation this is a picture
    # of. The same reason obstacle_demo hides its floor and walls instead of deleting them.
    # Also the mount hardware: the bolt plate, the ceiling slab, the wall panel and the pedestal
    # column. They exist to make a clip legible, but in a figure of eight panels they are four
    # different grey shapes competing with the robot, and the arm's own pose already says where it
    # is bolted. What is left is the table and the robot.
    for name in ("floor", "mat", "mount_plate", "mount_panel"):
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            mj_model.geom_rgba[gid, 3] = 0.0

    for g in range(mj_model.ngeom):
        if mj_model.geom_group[g] != 1:  # group 1 is the visual layer; 0 is collision
            continue
        if mj_model.geom_matid[g] >= 0:
            continue
        rgba = mj_model.geom_rgba[g]
        if abs(float(rgba[0]) - 0.5) > 1e-3:  # already coloured (the hand's 0.2 grey): leave it
            continue
        if float(rgba[3]) == 0.0:
            # Already hidden -- the lab cage. Recolouring it would rewrite the alpha that hid it
            # and put the whole enclosure back in the shot, which is exactly what it did once.
            continue
        did = int(mj_model.geom_dataid[g])
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MESH, did) if did >= 0 else ""
        rgb = _ARM_TRIM_RGBA if (name or "") in _TRIM_MESHES else _ARM_SHELL_RGBA
        mj_model.geom_rgba[g] = (*rgb[:3], float(rgba[3]))

    # The headlight is camera-attached and casts NO shadow, which is exactly what this scene needs
    # as fill: the ceiling and overhead mounts hang the arm directly beneath their own slab, so the
    # key light -- which does cast shadows -- leaves the arm in the fixture's shadow and the panel
    # renders grey. A strong shadowless fill keeps the arm lit wherever it is, and the key still
    # supplies the modelling and the shadow on the table.
    mj_model.vis.headlight.ambient[:] = 0.26
    mj_model.vis.headlight.diffuse[:] = 0.55
    mj_model.vis.headlight.specular[:] = 0.06
    mj_model.vis.quality.shadowsize = 4096
    mj_model.vis.map.shadowclip = 3.0
    mj_model.vis.map.shadowscale = 1.0


def studio_light_for_azimuth(mj_model, camera_azimuth: float, light_index: int = 0) -> None:
    """Re-aim the studio key light for a camera that has moved, on the compiled model.

    ``studio_scene`` places the light relative to the mount's own camera. The hero pass searches
    azimuths for one that shows the grasped object, and can land 50 deg away -- at which point the
    key is lighting the arm from the side and the panel goes grey. Light position and direction are
    plain model fields, so they can follow. Visual only.
    """
    if mj_model.nlight <= light_index:
        return
    a = math.radians(camera_azimuth + _LIGHT_AZIMUTH_OFFSET)
    e = math.radians(_LIGHT_ELEVATION)
    ux, uy, uz = math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)
    tx, ty, tz = _WORK_POINT
    mj_model.light_pos[light_index] = (
        tx + ux * _LIGHT_RANGE,
        ty + uy * _LIGHT_RANGE,
        tz + uz * _LIGHT_RANGE,
    )
    mj_model.light_dir[light_index] = (-ux, -uy, -uz)


def studio_renderer(renderer) -> None:
    """Black background and no cast shadows, on the renderer's own scene.

    Renderer-level flags, not model fields -- which is why this is a third call rather than part
    of studio_model. With no skybox MuJoCo clears to black, which is the background the obstacle
    figure uses. Shadows are off because the arm's shadow lands across the table in every panel
    and, with eight panels side by side, reads as scene clutter rather than as depth.
    """
    import mujoco

    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def studio_option():
    """``MjvOption`` showing only the visual layer -- no collision hulls, no sites."""
    import mujoco

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    # Group 1 is the visual layer everywhere that matters: the arm's meshes, the object's visual
    # mesh (its hulls are group 3), and -- after studio_scene -- the table and the fixtures.
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = 1
    return opt


__all__ = [
    "MOUNTS",
    "ORDER",
    "MountSpec",
    "MOUNT_ENV_VAR",
    "DEFAULT_MOUNT",
    "HOMES_JSON",
    "active",
    "resolve",
    "arm_q_home",
    "remount",
    "add_fixture",
    "studio_scene",
    "studio_model",
    "studio_option",
    "studio_renderer",
    "studio_light_for_azimuth",
    "studio_enabled",
    "STUDIO_ENV_VAR",
    "camera",
    "brighten",
    "hide_enclosure_geoms",
]
