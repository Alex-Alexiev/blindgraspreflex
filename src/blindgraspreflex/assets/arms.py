"""Cross-embodiment arm registry: the SAME Robotis RH-5 hand on four different arms.

Why this module exists
----------------------
The grasp policy this repo trains is *arm-agnostic by construction*, not by accident. Its
observation is 20 hand joint positions (x a 5-step history) and its action is a grasp score
plus 20 finger targets -- 100 -> 21, with no arm state on either side. The arm is driven by a
scripted phase machine (REACH standoff -> LIFT retreat) through a Cartesian velocity field, a
damped Jacobian pseudoinverse and a joint PD, none of which the policy sees or commands. The
only thing crossing between them is the scalar grasp score that latches the phase.

That factorisation is the claim. This module is what tests it: it bolts the identical hand,
with the identical checkpoint, onto three arms the policy has never seen, and lets the whole
comparison be driven by one environment variable.

    REFLEXIVE_ARM=rizon4   Flexiv Rizon 4, 7 DOF  -- the arm every policy here trained on
    REFLEXIVE_ARM=panda    Franka Emika Panda, 7 DOF
    REFLEXIVE_ARM=ur5e     Universal Robots UR5e, 6 DOF
    REFLEXIVE_ARM=gen3     Kinova Gen3, 7 DOF

UR5e is deliberately in the set: at 6 DOF it removes the null space entirely, so it is the one
arm where the task-space law has no redundancy to spend and the "same policy" claim cannot be
resting on the two arms happening to share a kinematic family.

How a merged robot is built
---------------------------
``build_arm_hand_spec`` takes the shared hardware MJCF (``Rizon4_new_hand_robot.xml``), keeps
only the ``hand_mount`` subtree -- the adapter, the palm and the 20 finger joints -- and grafts
it onto the new arm's tool frame with ``MjsFrame.attach_body``. Nothing is renamed: the hand
joints keep their ``a08..a27`` names, ``palm`` stays ``palm``, and every contact sensor, reward
term and observation in the task therefore resolves unchanged. The hand's own mount offset
(``hand_mount`` at (0.0315, 0.0554, 0) with a 180 deg z flip, palm 17 mm further out) rides
along with the body, so the palm sits on the flange axis exactly as it does on the Rizon.

The three arm MJCFs come from MuJoCo Menagerie (Apache-2.0), which converts each vendor's
official URDF; ``scripts/fetch_arms.py`` fetches them into ``datasets/arms/``. Their own
actuators and keyframes are stripped, because mjlab supplies actuators through ``ActuatorCfg``
and the task supplies the home pose -- but their *effort limits* are read off those actuators
first, so each arm keeps the torque envelope its publisher shipped rather than one invented
here.

What is held identical across arms
----------------------------------
Everything the policy or the reward can see:

* the hand -- same MJCF subtree, same joint limits, same PD gains, same effort cap;
* the home palm pose. ``arm_homes.json`` stores a per-arm joint vector solved by IK so that
  ``grasp_site`` lands on the *Rizon's* home pose to within a fraction of a millimetre. Since
  the LIFT/hold goal and the held palm attitude are both FK of the home pose, every arm carries
  the object to the same place and holds it the same way, and the success criterion means the
  same thing on all four;
* the object, table, spawn point, DR and episode length, which live in the task.

What necessarily differs: joint count, kinematics, link inertias, torque limits, and the joint
PD gains that track the velocity field (see ``arm_kp``/``arm_kd`` below).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from blindgraspreflex import BLINDGRASPREFLEX_ROOT_PATH

ARMS_ROOT = BLINDGRASPREFLEX_ROOT_PATH / "datasets" / "arms"
"""Where scripts/fetch_arms.py puts the Menagerie trees. Ignored by git, like every other
fetched asset in this repo."""

HOMES_JSON = Path(__file__).parent / "arm_homes.json"
"""Per-arm home joint vectors, solved once by scripts/fit_arm_home.py. Tracked in git: it is a
few hundred bytes, and a home pose that silently changed would move the lift goal and quietly
invalidate every cross-arm number."""

ARM_ENV_VAR = "REFLEXIVE_ARM"
DEFAULT_ARM = "rizon4"


##
# Spec.
##


@dataclass(frozen=True)
class ArmSpec:
    """One arm, and everything needed to hang the RH-5 hand off it."""

    key: str
    label: str
    """Name for tables and plots."""
    dof: int

    # --- geometry -------------------------------------------------------------------------
    mjcf: str | None = None
    """Path under ``datasets/arms/``. None means the arm is the Rizon, whose merged arm+hand
    MJCF already exists in the manipulator-software submodule and is loaded by its own module."""
    root_body: str = ""
    """Body directly under ``worldbody``; repositioned to ``base_pos``/``base_quat``."""
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """ABSOLUTE base pose, overwriting whatever the vendor MJCF declares. The UR5e ships with
    ``quat="0 0 0 -1"`` (facing -x), which would put the whole workspace behind the table."""

    tool_body: str = ""
    """Body carrying the tool flange -- the hand's parent after the graft."""
    tool_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tool_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Where, inside ``tool_body``, the Rizon's ``flange`` frame lands. Taken from each vendor
    model's own published tool frame (Panda ``attachment``, UR5e ``attachment_site``, Gen3
    ``pinch_site``), all of which put +z out of the flange, which is the convention the Rizon
    ``flange`` body uses and therefore the one the hand's mount offset assumes."""

    # --- joints ---------------------------------------------------------------------------
    joint_names: tuple[str, ...] = ()
    joint_range_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Explicit limits for joints the vendor MJCF leaves *continuous*. The Gen3's joints 1, 3,
    5 and 7 are unlimited, which compiles to ``jnt_range = [0, 0]``; the IK solver and the
    action term's joint decode both read ``jnt_range`` and would clamp every command to zero."""
    joint_armature: float | None = None
    """Reflected rotor inertia [kg m^2] forced onto every arm joint, when the vendor MJCF ships
    none. See :func:`pd_margin`: the velocity-field law's ``kd`` term is integrated EXPLICITLY, so
    it is stable only while ``kd*dt < 2*(I + damping*dt)``, and a geared joint whose published
    model carries neither rotor inertia nor damping has almost no ``I`` left at the wrist."""

    # --- control --------------------------------------------------------------------------
    effort_limits: tuple[float, ...] = ()
    """Per-joint torque cap [Nm]. Read off the vendor MJCF's own actuator ``forcerange`` by
    :func:`_read_effort_limits` -- the same rule the Rizon config follows -- so each arm keeps
    its publisher's envelope rather than a number invented for this comparison."""
    arm_kp: tuple[float, ...] = ()
    arm_kd: tuple[float, ...] = ()
    """Joint PD gains for the velocity-field law (``tau = kp*qd_des*dt + kd*(qd_des - qd) +
    grav``). The dominant term is ``kd``, which is a velocity-loop gain, so the gains have to
    scale with the arm's torque envelope: on the Rizon they were hand-tuned to
    ``kp ~ 0.8 * effort`` with ``kd = kp / 8``, and the other three arms take that same ratio
    off their own effort limits (:func:`_gains_from_effort`). Verified in free space by
    scripts/arm_freespace_track.py, which is what would catch a badly scaled arm before it
    reached an object."""

    # --- provenance -----------------------------------------------------------------------
    source: str = "third_party/manipulator-software (submodule)"
    license: str = "see the submodule"

    @property
    def mjcf_path(self) -> Path | None:
        return None if self.mjcf is None else ARMS_ROOT / self.mjcf

    @property
    def q_home(self) -> dict[str, float]:
        """Home joint angles [rad]. Solved by IK to put ``grasp_site`` on the Rizon's home
        palm pose, so the lift goal and hold attitude are common to every arm."""
        if self.key == DEFAULT_ARM:
            from blindgraspreflex.assets.rizon4_hand import ARM_Q_HOME

            return dict(ARM_Q_HOME)
        homes = _load_homes()
        if self.key not in homes:
            raise RuntimeError(
                f"No home pose for arm {self.key!r} in {HOMES_JSON}. "
                f"Run: uv run python scripts/fit_arm_home.py --arm {self.key}"
            )
        return {n: float(v) for n, v in zip(self.joint_names, homes[self.key], strict=True)}

    def robot_cfg(self) -> EntityCfg:
        """mjlab entity config for this arm + the RH-5 hand.

        The Rizon keeps its own module untouched, including the ``REFLEXIVE_HAND=v1`` escape
        hatch that keeps the old hand geometry runnable; only the three grafted arms go through
        :func:`_merged_robot_cfg`.
        """
        if self.key != DEFAULT_ARM:
            return _merged_robot_cfg(self)
        if os.environ.get("REFLEXIVE_HAND", "v2").strip().lower() in ("v1", "old", "1", "orig"):
            from blindgraspreflex.assets.rizon4_hand import get_rizon4_hand_robot_cfg

            return get_rizon4_hand_robot_cfg()
        from blindgraspreflex.assets import get_rizon4_hand_v2_robot_cfg

        return get_rizon4_hand_v2_robot_cfg()


def _load_homes() -> dict[str, list[float]]:
    if not HOMES_JSON.exists():
        return {}
    return json.loads(HOMES_JSON.read_text())["homes"]


##
# Merged spec: hand subtree grafted onto an arm.
##

_HAND_MOUNT_BODY = "hand_mount"
"""The Rizon MJCF's adapter body. Everything from here down -- the mount plate, the palm and
all 20 finger joints -- is the hand; everything above it is the Rizon arm and is discarded."""


def _subtree(body: mujoco.MjsBody):
    yield body
    for child in body.bodies:
        yield from _subtree(child)


def _hand_only_spec() -> mujoco.MjSpec:
    """The shared hardware MJCF with every mesh the hand does not use deleted.

    The file is arm+hand in one document, and its arm meshes are named ``link0..link7`` --
    which collide head-on with the Panda's mesh names and make the merged spec fail to compile
    with ``repeated name 'link1' in mesh``. Attaching under a prefix would fix the collision
    and break everything else, since the hand joints' ``a08..a27`` names are what the task's
    observations, rewards and contact sensors bind to. So the unused assets go instead.
    """
    from blindgraspreflex.assets.rizon4_hand_v2 import RIZON4_HAND_V2_XML

    spec = mujoco.MjSpec.from_file(str(RIZON4_HAND_V2_XML))
    mount = spec.body(_HAND_MOUNT_BODY)
    used = {g.meshname for b in _subtree(mount) for g in b.geoms if g.meshname}
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
    return spec


def _read_effort_limits(spec: mujoco.MjSpec, joint_names: tuple[str, ...]) -> tuple[float, ...]:
    """Per-joint |forcerange| from the vendor MJCF's own actuators, before they are stripped.

    Every Menagerie arm ships position actuators carrying the vendor's torque limits; mjlab
    needs torque actuators instead, so the actuators go but their envelope stays. Falls back to
    the joint's ``actuatorfrcrange`` (which is where the Rizon keeps the same numbers).
    """
    by_joint = {a.target: a for a in spec.actuators if a.target}
    limits = []
    for name in joint_names:
        act = by_joint.get(name)
        if act is not None and act.forcerange[1] > 0:
            limits.append(float(abs(act.forcerange[1])))
            continue
        jnt = spec.joint(name)
        rng = getattr(jnt, "actfrcrange", None)
        if rng is not None and rng[1] > 0:
            limits.append(float(abs(rng[1])))
            continue
        raise ValueError(
            f"No torque limit for joint {name!r}: neither an actuator forcerange nor an "
            "actuatorfrcrange. Set effort_limits explicitly on the ArmSpec."
        )
    return tuple(limits)


def _gains_from_effort(effort: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(kp, kd)`` for the velocity-field joint PD, from the arm's torque envelope.

    The Rizon's hand-tuned gains sit at roughly ``kp = 0.8 * effort`` with ``kd = kp / 8``
    (effort [123,123,64,64,39,39,39] against kp [80,80,80,80,40,40,20]). Both terms of the law
    produce a torque, so both have to move with the torque the arm can actually deliver: reused
    verbatim, the Rizon's kd would be a 5x-too-weak velocity loop on the UR5e's 150 Nm shoulder
    and a 3x-too-stiff one on the Gen3's 52 Nm wrist.
    """
    kp = tuple(0.8 * e for e in effort)
    return kp, tuple(k / 8.0 for k in kp)


def build_arm_hand_spec(arm: ArmSpec) -> mujoco.MjSpec:
    """Graft the RH-5 hand onto ``arm``'s tool flange and return the merged, uncompiled spec.

    The result is spec-for-spec what ``rizon4_hand_v2.get_spec`` returns for the Rizon: every
    geom named, the palm ``grasp_site`` added, the palm/proximal-phalanx contact exclusions
    applied, and the hand frictionloss override honoured -- so the task cannot tell the two
    apart other than by which joints move the palm.
    """
    from blindgraspreflex.assets.rizon4_hand import _CONTACT_EXCLUDES, _name_geoms
    from blindgraspreflex.assets.rizon4_hand_v2 import (
        GRASP_SITE_NAME,
        _override_hand_frictionloss,
    )

    path = arm.mjcf_path
    assert path is not None and path.exists(), (
        f"Arm MJCF not found at {path}. These come from MuJoCo Menagerie and are not vendored; "
        "run: uv run python scripts/fetch_arms.py"
    )
    spec = mujoco.MjSpec.from_file(str(path))

    # Mount the base where the task expects a robot: at its env origin, facing the table.
    root = spec.body(arm.root_body)
    root.pos = list(arm.base_pos)
    root.quat = list(arm.base_quat)

    # The vendor actuators go: mjlab builds its own torque actuators from ARTICULATION. Their
    # torque limits are kept, read separately by _cached_effort_limits off a fresh parse of the
    # same file. Keyframes go too -- one sized for the arm alone is the wrong length once 20
    # hand joints are attached, and mjlab takes the home pose from EntityCfg.init_state anyway.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)

    # Continuous joints compile to jnt_range [0, 0]; the IK solver and the action term both
    # read jnt_range as a real limit, so an unlimited joint has to be given one here.
    for name, (lo, hi) in arm.joint_range_overrides.items():
        joint = spec.joint(name)
        joint.limited = True
        joint.range = [lo, hi]

    # Reflected rotor inertia, where the vendor model omits it. Set here rather than folded into
    # the actuator config because it is a property of the PLANT -- the IK model, the Jacobian and
    # the mass matrix all have to see the same joint the controller is stabilising.
    if arm.joint_armature is not None:
        for name in arm.joint_names:
            spec.joint(name).armature = arm.joint_armature

    # The graft. attach_body preserves the child's own pos/quat, so the hand's mount offset
    # (and with it the palm's on-axis position 17 mm past the flange) is carried across.
    hand = _hand_only_spec()
    frame = spec.body(arm.tool_body).add_frame(pos=list(arm.tool_pos), quat=list(arm.tool_quat))
    frame.attach_body(hand.body(_HAND_MOUNT_BODY), "", "")

    _name_geoms(spec)
    spec.body("palm").add_site(name=GRASP_SITE_NAME, pos=(0.0, 0.0, 0.1))
    _override_hand_frictionloss(spec)
    for body1, body2 in _CONTACT_EXCLUDES:
        spec.add_exclude(name=f"exclude_{body1}_{body2}", bodyname1=body1, bodyname2=body2)
    _exclude_root_shoulder(spec, arm.root_body)

    return spec


def _exclude_root_shoulder(spec: mujoco.MjSpec, root_body: str) -> None:
    """Exclude the fixed base from its own first link.

    MuJoCo's ``filterparent`` drops body/parent contact pairs, but not when the parent is the
    world -- and a fixed-base arm's root link is *welded* to the world, so its weld id IS the
    world's and the exemption catches it. Its first moving link therefore collides with it
    forever. That is not hypothetical: the Menagerie Gen3 sits 12 mm inside its own shoulder at
    the zero configuration, giving four permanently-active contacts in every world of a 1024-env
    scene, and it made every IK home solution look like it was in collision.

    Excluded rather than worked around, for the same reason the palm/proximal-phalanx pairs in
    the hand MJCF are: a contact between two links that cannot move relative to each other
    carries no information and costs constraint rows.
    """
    for child in spec.body(root_body).bodies:
        spec.add_exclude(
            name=f"exclude_{root_body}_{child.name}",
            bodyname1=root_body,
            bodyname2=child.name,
        )


def _merged_robot_cfg(arm: ArmSpec) -> EntityCfg:
    """EntityCfg for a non-Rizon arm: vendor arm, torque-controlled; RH-5 hand, unchanged.

    The hand's actuators, gains, effort cap, armature and collision surfaces are imported from
    the Rizon config rather than redeclared, so there is exactly one definition of the hand and
    a cross-arm comparison cannot drift into being a cross-*hand* one.
    """
    from blindgraspreflex.assets.rizon4_hand import (
        COLLISION,
        HAND_ACTUATORS,
        HAND_Q_OPEN,
    )

    effort = arm.effort_limits or _cached_effort_limits(arm.key)
    arm_actuators = tuple(
        BuiltinMotorActuatorCfg(target_names_expr=(name,), effort_limit=limit)
        for name, limit in zip(arm.joint_names, effort, strict=True)
    )
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={**arm.q_home, **HAND_Q_OPEN},
            joint_vel={".*": 0.0},
        ),
        collisions=(COLLISION,),
        spec_fn=lambda: build_arm_hand_spec(arm),
        articulation=EntityArticulationInfoCfg(
            actuators=(*arm_actuators, *HAND_ACTUATORS),
            # Same reason as the Rizon config: the hand's open pose puts five joints exactly on
            # a hard limit, so any shrink factor below 1.0 places the default pose outside the
            # soft limits.
            soft_joint_pos_limit_factor=1.0,
        ),
    )


@cache
def _cached_effort_limits(key: str) -> tuple[float, ...]:
    """Compile-free read of the vendor torque limits, cached so it costs one XML parse."""
    arm = ARMS[key]
    spec = mujoco.MjSpec.from_file(str(arm.mjcf_path))
    return _read_effort_limits(spec, arm.joint_names)


##
# Explicit-PD stability.
##

PHYSICS_DT = 0.004
"""The task's physics timestep. The velocity-field law runs at this rate, so it is what the
stability bound below is evaluated at. Kept here rather than imported to avoid a circular
dependency on the task module; the task pins the same number (decimation 21 x 0.004 s)."""

PD_MARGIN_LIMIT = 2.0
"""``kd*dt / (I + damping*dt)`` must stay under this. It is the textbook explicit-Euler bound for
a velocity feedback term: above 2 the correction overshoots the velocity it was cancelling and
grows every step."""


def pd_margin(arm: ArmSpec, q: dict[str, float] | None = None) -> dict[str, float]:
    """Per-joint ``kd*dt / (I + damping*dt)`` at pose ``q`` (default: the arm's home).

    THIS IS THE CHECK THAT WOULD HAVE CAUGHT THE GEN3. The arm's joint PD

        tau = kp * (qd_des * dt) + kd * (qd_des - qd_meas) + grav

    is computed in the action term and applied as a torque, which means the ``kd`` term is
    integrated *explicitly* -- unlike a MuJoCo ``<position>`` actuator's damping, or the MJCF's
    own ``damping`` attribute, both of which the ``implicitfast`` integrator handles implicitly
    and which are therefore unconditionally stable. An explicit velocity feedback is stable only
    while ``kd*dt < 2*I_eff``, and ``I_eff`` at a wrist joint is dominated by reflected rotor
    inertia, not by the link.

    The failure mode is loud but its cause is not: the Menagerie Gen3 ships neither armature nor
    damping, which put its wrist at ``kd*dt/I = 8.0``. The arm thrashed at 20-55 rad/s against a
    2.5 rad/s command clamp, the instability termination reset the episode every few steps, and
    the score was simply zero -- with nothing in any log naming a gain or an inertia.

    ``I`` is read from the diagonal of the full mass matrix (which already includes armature) at
    the given pose, so it accounts for configuration-dependent inertia coupling.
    """
    import numpy as np

    if arm.mjcf is None:
        from blindgraspreflex.assets.rizon4_hand_v2 import get_spec

        model = get_spec().compile()
    else:
        model = build_arm_hand_spec(arm).compile()
    data = mujoco.MjData(model)
    for name, value in (q or arm.q_home).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    full_m = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full_m)

    out = {}
    for name, kd in zip(arm.joint_names, arm.arm_kd, strict=True):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[jid]
        # damping*dt: the MJCF's own damping is integrated implicitly, so to first order it adds
        # damping*dt to the effective inertia the explicit term sees rather than subtracting from
        # its stability budget. This is what keeps the Rizon's 0.0028 kg m^2 wrist stable.
        inertia = float(full_m[dof, dof]) + float(model.dof_damping[dof]) * PHYSICS_DT
        out[name] = float(kd) * PHYSICS_DT / max(inertia, 1e-12)
    return out


##
# The registry.
##

_MENAGERIE = "MuJoCo Menagerie (google-deepmind/mujoco_menagerie), Apache-2.0"

_PANDA_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_UR5E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_GEN3_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))

ARMS: dict[str, ArmSpec] = {
    "rizon4": ArmSpec(
        key="rizon4",
        label="Flexiv Rizon 4 (trained)",
        dof=7,
        joint_names=tuple(f"joint{i}" for i in range(1, 8)),
        effort_limits=(123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0),
        # The hand-tuned gains every policy in this repo trained under. Left exactly as they
        # were: the whole point of the comparison is that the trained arm is unchanged.
        arm_kp=(80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 20.0),
        arm_kd=(10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.5),
    ),
    "panda": ArmSpec(
        key="panda",
        label="Franka Emika Panda",
        dof=7,
        mjcf="franka_emika_panda/panda_nohand.xml",
        root_body="link0",
        tool_body="attachment",
        # `attachment` IS Franka's published flange frame (link7 + 107 mm, rotated -45 deg about
        # z), with +z out of the flange, so the hand mounts onto it with no extra offset.
        joint_names=_PANDA_JOINTS,
        source=_MENAGERIE,
        license="Apache-2.0",
    ),
    "ur5e": ArmSpec(
        key="ur5e",
        label="Universal Robots UR5e",
        dof=6,
        mjcf="universal_robots_ur5e/ur5e.xml",
        root_body="base",
        # The vendor MJCF declares quat="0 0 0 -1", i.e. the arm facing -x, which points the
        # whole workspace away from the table. Overridden to identity so it faces the object.
        base_quat=(1.0, 0.0, 0.0, 0.0),
        tool_body="wrist_3_link",
        # UR5e's `attachment_site`: 100 mm along wrist_3's +y, rotated -90 deg about x so that
        # +z points out of the flange.
        tool_pos=(0.0, 0.1, 0.0),
        tool_quat=(-0.7071067811865476, 0.7071067811865476, 0.0, 0.0),
        joint_names=_UR5E_JOINTS,
        source=_MENAGERIE,
        license="Apache-2.0 (BSD-3-Clause meshes)",
    ),
    "gen3": ArmSpec(
        key="gen3",
        label="Kinova Gen3",
        dof=7,
        mjcf="kinova_gen3/gen3.xml",
        root_body="base_link",
        tool_body="bracelet_link",
        # Gen3's `pinch_site`: 61.5 mm along the bracelet's -z, flipped 180 deg about x so +z
        # points out of the flange.
        tool_pos=(0.0, 0.0, -0.061525),
        tool_quat=(0.0, 1.0, 0.0, 0.0),
        joint_names=_GEN3_JOINTS,
        # Joints 1, 3, 5 and 7 are continuous on the real Gen3 and unlimited in the MJCF. A
        # revolute joint with jnt_range [0, 0] would be clamped shut by the IK solver and by
        # the action term's joint decode, so they are given the full turn they physically have.
        joint_range_overrides={
            f"joint_{i}": (-3.141592653589793, 3.141592653589793) for i in (1, 3, 5, 7)
        },
        # The Menagerie Gen3 ships armature=0 AND damping=0, alone among the four arms here (the
        # Panda and UR5e carry armature 0.1; the Rizon carries damping 2.0). Its wrist link
        # inertia is 0.0026 kg m^2, so with nothing added the velocity-field law's explicitly
        # integrated kd term ran at kd*dt/I = 8.0 -- four times the stability bound. It was not
        # subtle: the arm thrashed at 20-55 rad/s against a 2.5 rad/s command clamp, tripped the
        # instability termination every few steps, and the policy never got a single object off
        # the table. 0.1 is what its two Menagerie siblings use for the same class of geared
        # joint, and a real harmonic-drive actuator certainly has reflected rotor inertia -- the
        # zero is a gap in the published model, not a property of the robot. See pd_margin().
        joint_armature=0.1,
        source=_MENAGERIE,
        license="Apache-2.0 (BSD-3-Clause meshes)",
    ),
}

ORDER = ("rizon4", "panda", "ur5e", "gen3")
"""Presentation order: the trained arm first, then the three it has never seen."""


def get_arm(key: str | None = None) -> ArmSpec:
    """The selected arm, with gains filled in from its torque envelope.

    ``key`` defaults to ``$REFLEXIVE_ARM`` and then to the Rizon, so every existing script and
    checkpoint keeps the arm it was written for unless the variable is set. Like
    ``REFLEXIVE_HAND`` and ``REFLEXIVE_OBJECT_SCALES``, it has to be set BEFORE the task module
    is imported: the joint names it resolves are read at import time.
    """
    key = (key or os.environ.get(ARM_ENV_VAR) or DEFAULT_ARM).strip().lower()
    if key not in ARMS:
        raise SystemExit(f"Unknown {ARM_ENV_VAR}={key!r}; choose from {list(ARMS)}")
    return _resolved(key)


@cache
def _resolved(key: str) -> ArmSpec:
    """Cached on the RESOLVED key, not on the caller's ``None`` -- otherwise one lookup made
    before ``$REFLEXIVE_ARM`` was read would pin the default for the rest of the process."""
    arm = ARMS[key]
    if arm.effort_limits and arm.arm_kp:
        return arm
    from dataclasses import replace

    effort = arm.effort_limits or _cached_effort_limits(key)
    kp, kd = _gains_from_effort(effort)
    return replace(arm, effort_limits=effort, arm_kp=kp, arm_kd=kd)
