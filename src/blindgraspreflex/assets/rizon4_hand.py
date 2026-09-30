"""Flexiv Rizon 4 + Robotis RH-5 five-finger hand entity config.

The MJCF is the shared hardware model from the ``manipulator-software`` submodule. It is
deliberately left untouched: it declares no actuators (``nu=0``) and no sites
(``nsite=0``), so everything mjlab needs is added here via ``spec_fn`` and ``EntityCfg``.

What this module adds on top of the raw MJCF:

* names for every geom (the MJCF leaves most unnamed, and ``CollisionCfg`` matches on
  names -- ``spec.geom("")`` is ambiguous when several geoms share the empty name);
* fingertip / palm sites, needed by the reaching and contact reward terms;
* the palm-vs-proximal-phalanx contact exclusions that ``hand_default.xml`` carries;
* actuators: torque on the arm, position on the hand (see "Control choice" below).

Work cell
---------
The ``enclosure`` body in the MJCF is a visual-only mesh of the physical cell
(``contype=0 conaffinity=0``); it is kept so the workspace is visible, but it does not
constrain the robot. The cell's collision geometry -- a floor and two walls -- is defined
in ``hand_default.xml``, which is a *scene* file this entity does not include, and whose
own comments mark those wall positions as placeholders. If training should respect the
cell boundaries, add those walls to the scene config (mjlab's ``SceneCfg``), not here:
they are world geometry shared by all environments, not part of the robot.

Control choice
--------------
The arm gets *torque* actuators because the ported grasp controller
(``GraspPhaseSpringDamperControlAction``) implements its own Cartesian impedance law and
needs direct torque authority. The hand gets *position* actuators because the hand action
term commands joint position targets through a PCA latent. Effort limits are the
``ctrlrange`` values from ``hand_default.xml``, which is the hardware contract.
"""

from __future__ import annotations

import os
import re

import mujoco
import numpy as np
from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from blindgraspreflex import MANIPULATOR_SOFTWARE_PATH

##
# MJCF and assets.
##

RIZON4_HAND_XML = (
    MANIPULATOR_SOFTWARE_PATH / "assets" / "scene" / "flexiv_arm" / "Rizon4_hand_robot.xml"
)
assert RIZON4_HAND_XML.exists(), (
    f"Robot MJCF not found at {RIZON4_HAND_XML}. Did you init the submodule? "
    "Run: git submodule update --init --recursive"
)

##
# Joint groups.
##

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

FINGERS = ("1_thumb", "2_index", "3_middle", "4_ring", "5_little")

# Hand joints are numbered a08..a27, five fingers x four joints, finger-major. Naming
# matches the IsaacLab config exactly, so ported term configs need no renaming.
HAND_JOINT_NAMES = tuple(
    f"a{8 + 4 * f + j:02d}_finger_r_joint_{finger}{j + 1}"
    for f, finger in enumerate(FINGERS)
    for j in range(4)
)

FINGERTIP_BODY_NAMES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")
FINGERTIP_SITE_NAMES = tuple(f"{name}_site" for name in FINGERTIP_BODY_NAMES)
GRASP_SITE_NAME = "grasp_site"

##
# Spec editing.
##

# The palm hull and each finger's *second* phalanx overlap slightly at rest, so they
# would generate spurious contacts every step. MuJoCo's filterparent already handles
# parent<->child pairs. Copied from hand_default.xml, which the hardware sim uses.
_CONTACT_EXCLUDES = tuple(("palm", f"finger_r_link_{finger}2") for finger in FINGERS)


def _is_collision_geom(geom: mujoco.MjsGeom) -> bool:
    return bool(geom.contype) or bool(geom.conaffinity)


def _name_geoms(spec: mujoco.MjSpec) -> None:
    """Give every geom a deterministic ``<body>_collision`` / ``<body>_visual`` name.

    The MJCF names only four geoms and leaves the other 64 blank. mjlab's ``CollisionCfg``
    resolves geoms by regex over names and then looks each one up with ``spec.geom(name)``,
    which cannot disambiguate several geoms all named "". Existing names are overwritten
    rather than preserved so a single regex covers the whole robot.
    """
    for body in spec.bodies:
        collision_idx = 0
        visual_idx = 0
        for geom in body.geoms:
            if _is_collision_geom(geom):
                suffix = "" if collision_idx == 0 else f"_{collision_idx}"
                geom.name = f"{body.name}_collision{suffix}"
                collision_idx += 1
            else:
                suffix = "" if visual_idx == 0 else f"_{visual_idx}"
                geom.name = f"{body.name}_visual{suffix}"
                visual_idx += 1


def _add_sites(spec: mujoco.MjSpec) -> None:
    """Add fingertip and palm grasp sites.

    The ``*_tip`` bodies are already massless frames positioned at the finger tips, so the
    sites sit at their origins. ``grasp_site`` reproduces the IsaacLab
    ``palm_grasp_point_offset`` -- 10 cm along the palm's local +z, which points toward
    the fingertips -- and marks the nominal grasp centre.
    """
    for body_name, site_name in zip(FINGERTIP_BODY_NAMES, FINGERTIP_SITE_NAMES, strict=True):
        spec.body(body_name).add_site(name=site_name, pos=(0.0, 0.0, 0.0))
    spec.body("palm").add_site(name=GRASP_SITE_NAME, pos=(0.0, 0.0, 0.1))


def get_spec() -> mujoco.MjSpec:
    """Load the shared MJCF and apply the edits mjlab needs. Returns a fresh spec.

    The world-parented ``enclosure`` body (a decimated mesh of the physical work cell) is
    kept. It is visual-only in this MJCF -- ``contype=0 conaffinity=0`` -- so it costs
    render geometry but nothing in physics. The cell's *collision* boxes live in
    hand_default.xml, which this file does not include; see the module docstring.
    """
    spec = mujoco.MjSpec.from_file(str(RIZON4_HAND_XML))

    _name_geoms(spec)
    _add_sites(spec)

    for body1, body2 in _CONTACT_EXCLUDES:
        spec.add_exclude(name=f"exclude_{body1}_{body2}", bodyname1=body1, bodyname2=body2)

    return spec


##
# Actuators.
##

# Effort limits are the ctrlrange values from hand_default.xml (arm values equal the
# MJCF's per-joint actuatorfrcrange; the hand's actuatorfrcrange is an unset +-1000
# placeholder, so the 10 Nm ctrlrange is the meaningful bound).
_ARM_EFFORT_LIMITS = {
    "joint1": 123.0,
    "joint2": 123.0,
    "joint3": 64.0,
    "joint4": 64.0,
    "joint5": 39.0,
    "joint6": 39.0,
    "joint7": 39.0,
}
# Grip-force cap per hand joint [Nm] -- the real RH-5 finger motors' limit. A blocked finger
# presses at most this hard, which sets how far it sinks into a grasped object; the earlier
# placeholder 10 Nm was ~10x the hardware value and drove ~3 mm of visible clipping. 1 Nm is
# both physically correct and a gentle, quasi-static grip. Overridable via REFLEXIVE_HAND_EFFORT.
HAND_EFFORT_LIMIT = float(os.environ.get("REFLEXIVE_HAND_EFFORT", "1.0"))

# The finger links are near-massless (tip phalanx effective inertia ~8e-6 kg m^2), which
# makes the position servo stiff to integrate. A small rotor inertia dominates that term
# and conditions the mass matrix without meaningfully changing finger dynamics. Tunable.
HAND_ARMATURE = 0.002418  # physical XM335 rotor inertia (matches the deploy MJCF)

ARM_ACTUATORS = tuple(
    BuiltinMotorActuatorCfg(
        target_names_expr=(name,),
        effort_limit=effort_limit,
        # Arm armature is left at the MJCF value (0). Flexiv does not publish reflected
        # rotor inertias; the MJCF's damping=2.0 per joint is the hardware team's tuned
        # value and is deliberately not overridden here.
    )
    for name, effort_limit in _ARM_EFFORT_LIMITS.items()
)

# Gains fall off distally, matching the decreasing effective inertia along each finger.
# These are starting values, not identified from hardware -- expect to tune them once the
# grasp task is closing the loop.
_HAND_GAINS_BY_LEVEL = {
    1: (6.0, 0.25),  # MCP abduction / thumb rotation
    2: (5.0, 0.20),  # MCP flexion
    3: (3.0, 0.12),  # PIP
    4: (2.0, 0.08),  # DIP
}

# Multiply every hand kp/kd by this. 1.0 = the table above (what training has always used); the
# deploy YAML runs kp=0.5 flat, i.e. ~0.083-0.25x these, so REFLEXIVE_HAND_KP_SCALE is how you ask
# "what would this policy do on the softer hand the robot actually has?" without editing the table.
# kd scales with kp so the damping ratio is preserved (zeta ~ kd/(2*sqrt(kp*I))) -- scaling kp alone
# would silently change the finger's damping and confound a stiffness sweep.
_HAND_GAIN_SCALE = float(os.environ.get("REFLEXIVE_HAND_KP_SCALE", "1.0"))
if _HAND_GAIN_SCALE != 1.0:
    _HAND_GAINS_BY_LEVEL = {
        k: (kp * _HAND_GAIN_SCALE, kd * _HAND_GAIN_SCALE**0.5)
        for k, (kp, kd) in _HAND_GAINS_BY_LEVEL.items()
    }


def _joint_level(joint_name: str) -> int:
    """Phalanx index 1-4 encoded in the joint suffix, e.g. ...index3 -> 3."""
    match = re.search(r"(\d)$", joint_name)
    assert match is not None, f"Unexpected hand joint name: {joint_name}"
    return int(match.group(1))


# Per-joint gain vectors in HAND_JOINT_NAMES order, so the action term can run the PD law
# itself in EXPLICIT mode (below) and so any consumer reads the same numbers the actuators use.
HAND_KP = tuple(_HAND_GAINS_BY_LEVEL[_joint_level(n)][0] for n in HAND_JOINT_NAMES)
HAND_KD = tuple(_HAND_GAINS_BY_LEVEL[_joint_level(n)][1] for n in HAND_JOINT_NAMES)

# EXPLICIT PD mode (REFLEXIVE_HAND_EXPLICIT_PD=1). Default OFF -- switching changes the plant, so
# every existing checkpoint keeps the implicit actuator it trained on.
#
#   implicit (default): a MuJoCo <position> actuator. The solver forms
#       tau = kp*(q_cmd - q) - kd*qdot
#   INSIDE the integrator, so with integrator="implicitfast" the damping term is handled
#   implicitly and the servo is unconditionally stable at large kp.
#
#   explicit: a <motor> actuator fed a torque this module computes in Python -- byte-for-byte the
#   law Manipulator-Software's sim applies (SimRobotis5FHandZ.apply_sim_control:
#       tau_command = kp*(q_des - q) + kd*(qd_des - qd) + tau_ff,  qd_des = 0, tau_ff = 0)
#   and the law the XM335 servo runs on hardware. The point is that sim and deploy then evaluate
#   the SAME expression rather than two that merely agree in the limit, and tau becomes an
#   observable signal rather than a quantity buried in the solver.
#
# Stability: the PD leaves the implicit integrator, so it must be explicitly integrable. With the
# stiffest joint (kp 6, kd 0.25) against the rotor inertia (armature 0.002418) at physics dt 0.004:
#   omega = sqrt(kp/I) = 49.8 rad/s -> omega*dt = 0.20   (stable, needs < ~2)
#   kd/I * dt = 0.41                                     (stable, needs < 2)
# so 250 Hz has ~10x margin. Raising kp above ~150 or dt above ~0.02 would not.
HAND_EXPLICIT_PD = bool(os.environ.get("REFLEXIVE_HAND_EXPLICIT_PD"))

if HAND_EXPLICIT_PD:
    HAND_ACTUATORS = tuple(
        BuiltinMotorActuatorCfg(
            target_names_expr=(name,),
            effort_limit=HAND_EFFORT_LIMIT,
            armature=HAND_ARMATURE,
        )
        for name in HAND_JOINT_NAMES
    )
else:
    HAND_ACTUATORS = tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name,),
            stiffness=_HAND_GAINS_BY_LEVEL[_joint_level(name)][0],
            damping=_HAND_GAINS_BY_LEVEL[_joint_level(name)][1],
            effort_limit=HAND_EFFORT_LIMIT,
            armature=HAND_ARMATURE,
        )
        for name in HAND_JOINT_NAMES
    )

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(*ARM_ACTUATORS, *HAND_ACTUATORS),
    # Left at 1.0 (no shrink) deliberately. The hardware's open-hand pose puts five joints
    # exactly ON a hard limit (thumb1 at its upper stop, the four fingerN2 flexors at their
    # lower stop of 0.0), so any factor below 1.0 would place the nominal default pose
    # outside the soft limits and make a ported joint_pos_limits penalty fire at reset.
    # Note these limits are reported data only -- mjlab never clamps ctrl to them.
    soft_joint_pos_limit_factor=1.0,
)

##
# Initial state.
##

# Hardware default pose: the arm at [-25, -40, 35, 100, 50, 30, 60] deg and the hand fully
# open at zero. Taken verbatim from the Manipulator-Software config for this exact robot
# (configs/flexiv_arm_5F_hand/env/default.yaml, matched by env/collision.yaml and
# controllers/default_gravcomp.yaml), so sim and hardware start from the same state.
HAND_Q_OPEN = dict.fromkeys(HAND_JOINT_NAMES, 0.0)

# Endpoints of the hand's 1-D closure, carried over from the IsaacLab
# HandPoseVelocityActionCfg. These are *task* poses for the hand action term, not the
# hardware rest pose -- note that IsaacLab's open endpoint splays the fingers (thumb2
# -1.9, index1 -1.0, ring1 0.5, little1 1.2) rather than zeroing them, so it is kept
# separate from HAND_Q_OPEN above rather than conflated with it.
HAND_Q_GRASP_OPEN = {
    "a08_finger_r_joint_1_thumb1": 0.0,
    "a09_finger_r_joint_1_thumb2": -1.9,
    "a10_finger_r_joint_1_thumb3": 0.0,
    "a11_finger_r_joint_1_thumb4": 0.0,
    "a12_finger_r_joint_2_index1": -1.0,
    "a13_finger_r_joint_2_index2": 0.0,
    "a14_finger_r_joint_2_index3": 0.0,
    "a15_finger_r_joint_2_index4": 0.0,
    "a16_finger_r_joint_3_middle1": 0.0,
    "a17_finger_r_joint_3_middle2": 0.0,
    "a18_finger_r_joint_3_middle3": 0.0,
    "a19_finger_r_joint_3_middle4": 0.0,
    "a20_finger_r_joint_4_ring1": 0.5,
    "a21_finger_r_joint_4_ring2": 0.0,
    "a22_finger_r_joint_4_ring3": 0.0,
    "a23_finger_r_joint_4_ring4": 0.0,
    "a24_finger_r_joint_5_little1": 1.2,
    "a25_finger_r_joint_5_little2": 0.0,
    "a26_finger_r_joint_5_little3": 0.0,
    "a27_finger_r_joint_5_little4": 0.0,
}

HAND_Q_CLOSE = {
    "a08_finger_r_joint_1_thumb1": -2.0,
    "a09_finger_r_joint_1_thumb2": -1.9,
    "a10_finger_r_joint_1_thumb3": 1.4,
    "a11_finger_r_joint_1_thumb4": 1.4,
    "a12_finger_r_joint_2_index1": 0.5,
    "a13_finger_r_joint_2_index2": 1.8,
    "a14_finger_r_joint_2_index3": 1.4,
    "a15_finger_r_joint_2_index4": 1.4,
    "a16_finger_r_joint_3_middle1": 0.5,
    "a17_finger_r_joint_3_middle2": 1.8,
    "a18_finger_r_joint_3_middle3": 1.4,
    "a19_finger_r_joint_3_middle4": 1.4,
    "a20_finger_r_joint_4_ring1": 0.9,
    "a21_finger_r_joint_4_ring2": 1.8,
    "a22_finger_r_joint_4_ring3": 1.4,
    "a23_finger_r_joint_4_ring4": 1.4,
    "a24_finger_r_joint_5_little1": 1.8,
    "a25_finger_r_joint_5_little2": 1.8,
    "a26_finger_r_joint_5_little3": 1.4,
    "a27_finger_r_joint_5_little4": 1.4,
}

# See HAND_Q_OPEN: same hardware config, [-25, -40, 35, 100, 50, 30, 60] deg.
ARM_Q_HOME = {
    "joint1": -0.4363323,
    "joint2": -0.6981317,
    "joint3": 0.6108652,
    "joint4": 1.7453293,
    "joint5": 0.8726646,
    "joint6": 0.5235988,
    "joint7": 1.0471976,
}

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={**ARM_Q_HOME, **HAND_Q_OPEN},
    joint_vel={".*": 0.0},
)

##
# Collisions.
##

# Phalanges 2-4 and the palm are the surfaces that actually load an object, so they get
# condim=6 (torsional and rolling friction). Without it a grasped object spins freely
# about the contact normals and pinch grasps will not hold. priority=1 makes these
# settings win over whatever the object geom specifies.
_GRASP_SURFACES = r"(palm|finger_r_link_\d_[a-z]+[234])_collision"

COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={_GRASP_SURFACES: 6, r".*_collision": 3},
    friction={_GRASP_SURFACES: (1.0, 5e-3, 5e-4), r".*_collision": (0.6,)},
    solref={_GRASP_SURFACES: (0.01, 1.0)},
    priority={_GRASP_SURFACES: 1},
)

##
# Final config.
##


def get_rizon4_hand_robot_cfg() -> EntityCfg:
    """Rizon 4 + RH-5 hand: 7 torque-controlled arm joints, 20 position-controlled hand joints."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# Per-joint action scale for the position-controlled hand joints, using mjlab's
# convention of a quarter of the actuator's saturated position error.
# Read kp from the gain TABLE, not off the actuator cfg: a <motor> actuator (explicit-PD mode)
# has no `stiffness` field, and the scale must come out identical in both modes or the same action
# would mean a different commanded pose depending on the actuator -- which would silently
# invalidate every checkpoint on the switch.
HAND_ACTION_SCALE: dict[str, float] = {
    _name: 0.25 * HAND_EFFORT_LIMIT / _HAND_GAINS_BY_LEVEL[_joint_level(_name)][0]
    for _name in HAND_JOINT_NAMES
}


##
# Standalone inspection.
##

# Arm hold bandwidth for the inspection viewers below. The arm actuators are torque
# sources, so with zero ctrl the arm simply collapses under gravity and there is nothing
# to look at. This holds it at ARM_Q_HOME so the hand can be driven by hand.
_HOLD_NATURAL_FREQ = 2.0 * np.pi * 3.0  # 3 Hz
_HOLD_DAMPING_RATIO = 1.5


def _make_hold_home_step_fn(model: mujoco.MjModel):
    """Build a step function that keeps the arm at the home pose under gravity.

    Computed torque, not a plain joint-space PD: the commanded acceleration is mapped
    through the arm block of the mass matrix. This arm is strongly inertia-coupled and its
    effective inertias span three orders of magnitude (joint2 ~3.9, joint7 ~0.0025 kg m^2),
    so gains sized per joint from ``diag(M)`` leave joint3 buzzing at several rad/s.
    Routing through the full ``M`` decouples the closed loop and holds the pose exactly.

    ``qfrc_bias`` cancels gravity and Coriolis terms -- the ``grav_comp`` the IsaacLab task
    assumes. Hand actuators are left alone so the viewer's sliders drive them.
    """
    arm_act = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINT_NAMES]
    )
    arm_jnt = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    arm_qadr = np.array([model.jnt_qposadr[j] for j in arm_jnt])
    arm_dof = np.array([model.jnt_dofadr[j] for j in arm_jnt])
    q_target = np.array([ARM_Q_HOME[n] for n in ARM_JOINT_NAMES])
    mass_matrix = np.zeros((model.nv, model.nv))

    def step_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_fullM(model, data, mass_matrix)
        desired_acc = _HOLD_NATURAL_FREQ**2 * (q_target - data.qpos[arm_qadr]) - (
            2.0 * _HOLD_DAMPING_RATIO * _HOLD_NATURAL_FREQ * data.qvel[arm_dof]
        )
        data.ctrl[arm_act] = (
            data.qfrc_bias[arm_dof] + mass_matrix[np.ix_(arm_dof, arm_dof)] @ desired_acc
        )
        mujoco.mj_step(model, data)

    return step_fn


def _reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    _seed_hand_ctrl(model, data)
    mujoco.mj_forward(model, data)


def _seed_hand_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Point the hand position servos at the current pose.

    The viewer builds its actuator sliders from ``data.ctrl``, so without this the hand
    starts commanded to zero and snaps out of the open pose the moment it is unpaused.
    """
    for name in HAND_JOINT_NAMES:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.ctrl[act_id] = data.qpos[model.jnt_qposadr[jnt_id]]


if __name__ == "__main__":
    import argparse

    from mjlab.entity.entity import Entity

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--viewer",
        choices=("viser", "native"),
        default="viser",
        help="viser serves a browser UI (works headless/over SSH); native opens a GL window.",
    )
    parser.add_argument("--port", type=int, default=8080, help="viser server port.")
    args = parser.parse_args()

    robot = Entity(get_rizon4_hand_robot_cfg())
    mj_model = robot.spec.compile()
    mj_data = mujoco.MjData(mj_model)
    _reset_to_home(mj_model, mj_data)

    if args.viewer == "native":
        import mujoco.viewer as native_viewer

        native_viewer.launch(mj_model, mj_data)
    else:
        import viser
        from mjviser import Viewer

        Viewer(
            mj_model,
            mj_data,
            step_fn=_make_hold_home_step_fn(mj_model),
            reset_fn=_reset_to_home,
            server=viser.ViserServer(port=args.port, label="rizon4_hand"),
        ).run()
