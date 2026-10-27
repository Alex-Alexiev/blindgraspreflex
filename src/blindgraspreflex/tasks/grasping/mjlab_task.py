"""mjlab task for reflexive grasping.

Wraps the verified batched controllers as an mjlab ``ActionTerm`` and the MDP terms as
manager terms, so training runs on mujoco-warp (measured ~57k env-steps/s at 256 envs against
388/s for the CPU reference env in ``env.py``).

Object diversity via per-world mesh variants
--------------------------------------------
mjlab compiles one model shared by every environment, but its per-world mesh variant feature
lets a single ``object`` free body hold a *different* YCB mesh (and its own mass/inertia) in
each parallel world, all sharing one kinematic structure. So ``make_task`` accepts either one
object name (same object everywhere, e.g. ``Reflexive-Grasp-Apple``) or a list (a different
object per world, evenly allocated, e.g. ``Reflexive-Grasp-YCB`` over the EASY set). Because
the body is always named ``object``, every observation, reward and contact match is identical
in both cases.

Stateful rewards
----------------
mjlab reward terms are plain functions, so the improvement-reward watermarks live on the env
(lazily created, keyed by name) and are cleared by a ``mode="reset"`` event term. That mirrors
how the IsaacLab env carried them and keeps the reset in one place.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg

from blindgraspreflex.assets import (
    HAND_JOINT_NAMES,
    HAND_Q_CLOSE,
    mounts,
)
from blindgraspreflex.assets.arms import get_arm
from blindgraspreflex.assets.ycb import get_ycb_object_cfg, get_ycb_variant_object_cfg
from blindgraspreflex.controllers.batched import (
    BatchedArmImpedance,
    BatchedGraspPhase,
    BatchedGraspPhaseCfg,
    BatchedImpedanceGains,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_rotvec,
)
from blindgraspreflex.mjlab_patches import apply_variant_indexing_patch
from blindgraspreflex.tasks.grasping.mdp.terms import CONTACT_WEIGHTS, REWARD_CONTACT_LINKS

# THE ARM, resolved once at import time from $REFLEXIVE_ARM (default: the Flexiv Rizon 4 every
# policy in this repo trained on). The hand is unchanged in every case, and the grasp policy sees
# only hand joints -- so swapping this swaps the embodiment underneath a fixed checkpoint, which
# is what the cross-embodiment evaluation measures. Like $REFLEXIVE_HAND and
# $REFLEXIVE_OBJECT_SCALES it must be set BEFORE this module is imported: the names below are
# bound here and read by the action term, the observations and the pre-grasp IK.
ARM = get_arm()
ARM_JOINT_NAMES = ARM.joint_names

# WHERE THAT ARM IS BOLTED, resolved from $REFLEXIVE_MOUNT (default: the floor mount at the table
# edge, which is a strict no-op). The arm registry says which arm carries the hand; the mount
# registry says where its base sits -- ceiling, wall, far side, tilted pedestal. The two compose,
# and the home pose belongs to the pair: MOUNT.q_home is fitted by IK per (arm, mount) so
# grasp_site lands on the SAME world palm pose regardless, which is what keeps the LIFT/hold goal,
# the held palm attitude and the REACH target distribution identical across both registries.
# See blindgraspreflex.assets.mounts; same import-time rule as $REFLEXIVE_ARM.
MOUNT = mounts.active()
ARM_Q_HOME = mounts.arm_q_home(ARM.key, ARM_JOINT_NAMES, ARM.q_home, MOUNT)
# Fix mjlab's per-world-variant sibling-entity indexing bug before any env is built. Without
# this the robot's forward-kinematics data is garbage in any multi-object (variant) scene.
apply_variant_indexing_patch()

FLOOR_HEIGHT = -0.04
MAT_THICKNESS = 0.01
TABLE_HEIGHT = FLOOR_HEIGHT + MAT_THICKNESS
MAT_CENTRE_XY = (0.55, 0.0)
MAT_HALF_EXTENT = (0.47, 0.43)
SPAWN_XY = (0.55, 0.15)
SPAWN_Z = TABLE_HEIGHT + 0.06

HAND_LINK_PATTERN = r"(palm|finger_r_link_.*)"

# Fixed pre-grasp / REACH hand pose (degrees, in HAND_JOINT_NAMES order: thumb, index, middle,
# ring, little x 4 joints each). Each finger's base/spread joint is splayed; distal joints straight
# -- an open pose the hand holds while the arm approaches (before engagement). Replaces the learned
# reach pose; use with a run that has the learned reach off so the reach pose is this constant.
HAND_Q_REACH_DEG = (-8, 0, 0, 0, -28, 0, 0, 0, -5, 0, 0, 0, 5, 0, 0, 0, 28, 0, 0, 0)


##
# Action term.
##


def _grasp_robot_cfg():
    """Robot entity config for the grasp task: the arm selected by ``$REFLEXIVE_ARM`` carrying
    the RH-5 hand.

    On the Rizon the hand is still switchable via ``REFLEXIVE_HAND`` (``v1``/``old`` for the
    original geometry), so the old hand stays runnable. The three cross-embodiment arms graft
    the v2 hand subtree onto a vendor MJCF -- see :mod:`blindgraspreflex.assets.arms`.

    ``$REFLEXIVE_MOUNT`` then re-bolts that robot: :func:`mounts.remount` overwrites the base
    pose and the arm's home joints, and nothing else. It returns the config untouched on the
    default floor mount.

    Used for BOTH the scene robot and the GraspAction home-pose FK model, so the two agree.
    """
    return mounts.remount(ARM.robot_cfg(), MOUNT, ARM_Q_HOME)


@dataclass(kw_only=True)
class GraspActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    object_name: str = "object"
    site_name: str = "grasp_site"
    impedance: BatchedImpedanceGains = field(default_factory=BatchedImpedanceGains)
    phase: BatchedGraspPhaseCfg = field(
        # Two REACH-pose regimes, selected by reach_randomize (the --arm-residual flag turns it
        # off). Both were tuned in the play --tune-reach GUI.
        #  * reach_randomize True  -> DR: object + sampled world offset, flat-hold x sampled rpy.
        #  * reach_randomize False -> fixed tuned pose: object + pre_grasp_offset (world),
        #    flat-hold x reach_rpy_fixed, with the hand's bounded pos/rot residual on top.
        default_factory=lambda: BatchedGraspPhaseCfg(
            min_setpoint_z=TABLE_HEIGHT,
            reach_randomize=True,
            reach_pos_bounds=((-0.045, 0.05), (-0.04, 0.03), (0.04, 0.075)),
            reach_rpy_bounds=(
                (math.radians(-80.0), math.radians(80.0)),
                (math.radians(-5.0), math.radians(30.0)),
                (math.radians(-44.0), math.radians(44.0)),
            ),
            # Fixed tuned grasp pose (used when reach_randomize is False): read from the tuner.
            pre_grasp_offset=(0.03, 0.015, 0.065),
            offset_frame="world",
            reach_rpy_fixed=(0.0, math.radians(11.0), 0.0),
            # Room for the +-3 cm position residual around the fixed offset without clipping.
            reach_object_radius=0.15,
        )
    )
    pose_box_centre: tuple[float, float, float] = (0.55, 0.15, 0.30)
    pose_box_half: tuple[float, float, float] = (0.20, 0.20, 0.32)
    """arm_control="pose_policy": the workspace box the commanded grasp-centre position spans.

    A network's raw output has no units, so the task-space action has to be bounded the way the
    joint-space one is bounded by jnt_range -- otherwise a std-1 Gaussian commands palm poses
    metres outside the arm's reach and the velocity controller spends every episode saturated.
    The default box is sized from the scene it has to cover, with margin: objects rest inside
    x 0.50-0.60, y 0.10-0.20, z 0.03, and the lift/hold goal (home_goal_pos) sits at
    (0.522, 0.127, 0.513). The box spans x 0.35-0.75, y -0.05-0.35, z -0.02-0.62, which contains
    both the table and the goal.
    """

    pose_rpy_span: float = math.radians(45.0)
    """arm_control="pose_policy": half-range [rad] of the commanded palm attitude about each axis,
    applied as an offset to the flat-hold home attitude rather than as an absolute orientation.
    Absolute quaternions are a poor action space -- most of the 4-sphere is unreachable junk and
    the useful region is a narrow band around palm-down -- so the policy steers away from
    flat-hold instead of naming an orientation from scratch."""

    arm_v_max: float = 0.5
    """Baseline arm mode: max commanded joint speed [rad/s] of the arm velocity field.

    The same quasi-static rate limit the hand already uses, applied to the arm. Without it the
    policy's joint target is applied as a step: the PD sees a large instantaneous error, the arm
    slews at whatever the effort limit allows, and the fastest way to earn the dense height reward
    is to swing through the object and launch it -- measured at iteration 100-200 of the first
    baseline run as 27 cm of lift with contact on 1 control step out of 855.
    """

    arm_slowdown_distance: float = 0.35
    """Joint error [rad] below which the arm velocity field ramps its speed down, mirroring
    ``hand_slowdown_distance``. Keeps the approach from arriving at full speed."""

    arm_joint_span: float | None = None
    """Baseline arm mode only: half-span [rad] of the commanded joint range, centred on ARM_Q_HOME.

    ``None`` maps the squashed action across each joint's FULL ``jnt_range``, which for this arm is
    roughly +-180 deg per joint -- so a zero action is a mid-range pose 1.07 m from the reset pose,
    well outside the workspace, and the policy must find the useful configurations inside a 7-D box
    that is almost entirely useless. A finite span instead centres the same absolute-joint-target
    command on the home pose the task already starts and holds at, which is where every useful
    configuration actually lives. Still absolute joint targets either way -- only the range differs.
    """

    arm_control: str = "impedance"
    """How the arm is driven.

    ``"impedance"`` / ``"velocity"``: the arm follows the phase controller (scripted REACH/LIFT
    toward the object and the retreat pose), and the policy influences it only through
    ``grasp_score`` and the optional palm residual.

    ``"joint_target"``: BASELINE. The phase controller does not drive the arm at all -- the policy
    commands all 7 arm joint positions directly, alongside the 20 hand joints, so the action is
    [arm(7), hand(20)] with no grasp score and no residual. Everything else is deliberately
    unchanged from the phase-controlled task -- same observations, same four rewards, same hand
    velocity field and PD, same home_goal_pos -- so a comparison isolates the controller.

    ``"joint_direct"``: BASELINE (model-based grasping, see ``mbg_exec``). The action is
    [arm joints(7), hand joints(20)] in RADIANS, unsquashed -- the same plant as ``"joint_target"``
    (arm velocity field -> joint PD + gravity feed-forward -> effort clamp), commanded in physical
    units because the commands come from inverse kinematics rather than from a network. This is
    what the model-based baseline actually uses: its grasps are arbitrary SE(3) palm poses, and
    the Cartesian law below cannot reliably reach those -- measured, it stalls tens of centimetres
    short while asking for 2.5 rad/s, because the greedy straight-line Cartesian path walks the
    redundant arm into a joint limit. Driving the IK solution in joint space avoids that without
    changing anything about the plant.

    ``"pose_target"``: BASELINE (model-based grasping). The action carries a
    Cartesian setpoint for the palm rather than a score: [grasp_site pos(3), quat wxyz(4),
    hand joints(20)], in metres and radians with NO squashing, since the commands come from an
    offline planner in physical units rather than from a network. The setpoint is handed to the
    same velocity-field + damped-pseudoinverse + PD arm law ``"velocity"`` uses, and the hand
    targets go through the same velocity field and servo, so the model-based baseline and the
    learned policy differ in what they command and in nothing else.
    """
    """Arm control law. ``"impedance"`` (default): the operational-space Cartesian spring/damper
    (``BatchedArmImpedance``). ``"velocity"``: a Jacobian-pseudoinverse velocity controller -- a
    Cartesian velocity field drives the grasp_site to the phase target, a *damped Jacobian
    pseudoinverse* (no QP -- fast, batched) maps that twist to arm joint velocities, and a
    joint-space PD (+ gravity feedforward) tracks ``q_des = q + qd*dt``. This is the sim-to-real
    deploy arm controller (the deploy adds joint-limit handling); training/play with it via
    ``--arm-velocity`` co-adapts the hand policy to the deploy arm."""
    vf_lin_speed: float = 0.35
    """Velocity field: far-field linear speed [m/s] (ramps to 0 within vf_lin_eps of the target).

    Was 1.0. Lowered with vf_lin_eps and the arm damping below as one tuned set, chosen in
    ``scripts/viser_baselines.py``. NOTE this is the speed of BOTH the approach and the lift --
    the field is shared by the two phases -- so it trades a gentler approach against a slower
    carry to the hold goal, and an episode length that was comfortable at 1.0 m/s is less so
    here."""
    vf_lin_eps: float = 0.1
    vf_ang_speed: float = 2.0
    vf_ang_eps: float = 0.25
    vel_damping: float = 0.02
    """Damped-pseudoinverse regularisation lambda^2: qd = J^T (J J^T + lambda^2 I)^-1 twist."""
    qd_limit: float = 2.5
    """Per-joint velocity clamp [rad/s] on the pseudoinverse solution."""
    twist_linear_limit: float = 0.35
    twist_angular_limit: float = 2.0
    """Maximum policy commanded palm linear [m/s] and angular [rad/s] velocities."""
    arm_kp: tuple[float, ...] | None = None
    arm_kd: tuple[float, ...] | None = None
    """Arm joint PD. None takes the SELECTED ARM's gains (``ARM.arm_kp`` / ``ARM.arm_kd``), which
    for the Rizon are the tuned pair (80/80/80/80/40/40/20 and 10/10/10/10/5/5/2.5 -- kp
    unchanged from the original, kd 2.5x its original 4/4/4/4/2/2/1, a more damped arm to go with
    the slower velocity field above). Tuned as a set; changing one without the others is not the
    configuration that was chosen.

    It has to default to None rather than to the Rizon's 7-vector: this is a cross-embodiment
    task now, and a 7-long tuple is not merely mistuned on the 6-DOF UR5e, it is a shape error
    that would broadcast into silence or a crash depending on where it lands."""

    arm_residual: bool = False
    """Whether the hand policy also outputs a 6-D palm residual: a 3-D position offset (dims 1:4)
    and a 3-D orientation offset (dims 4:7, euler XYZ) added to the arm phase target. With the
    REACH-pose randomisation above the hand is kept fully arm-agnostic, so this is off by default
    (action = [grasp_score, hand]). Set True *and* reach_randomize False (the --arm-residual train
    flag does both) for the residual-controlled arm, where the hand shapes both where and how the
    palm meets the object."""
    hand_mode: str = "joints"
    """Hand action space: "joints" (20 direct joint targets) or "pca" (5 PCA latent dims).

    "joints" gives full independent per-joint control and learns the grasp faster; "pca"
    commands coordinated grasp shapes on the fitted human-grasp manifold (5-DoF, more natural
    looking) but explores slower.
    """
    hand_pca_project: float = 0.0
    """(joints mode only) Blend the commanded 20-joint hand pose toward its PCA-manifold projection
    by this factor before sending it: ``cmd = (1-a)*cmd + a*reconstruct(cmd)``. 0 = off (raw
    joints), 1 = hard-project onto the manifold (exactly PCA-like shapes), 0.5 = halfway. Lets a
    joint-space policy be fine-tuned to produce natural PCA-like hand shapes while keeping its
    20-D action space."""
    hand_gate: bool = False
    """Arm-gated hand: the arm tells the hand when to grasp. When True, the hand policy controls the
    fingers only once the palm is within the phase's ``engage_radius`` of the object (the latched
    ``phase.engaged`` flag); until then the fingers are held at the default pre-grasp pose
    (``HAND_Q_GRASP_OPEN``). Fixes the "hand closes while standing still" failure: a proprioception
    policy has no notion of *when* to grasp, so the arm supplies that signal. The velocity field
    makes the open<->policy hand-off smooth. Off by default (unchanged behaviour); enable at play
    time to gate an already-trained policy, or at train time to learn with the gate."""
    reach_hand_pose: bool = True
    """Learned REACH-phase hand pose. When True (PCA mode only) the policy emits a *second*
    z_dim latent block: the fingers hold ``decode(reach_z)`` while the arm approaches, then
    switch to the grasp pose ``decode(grasp_z)`` once the palm engages the object -- replacing
    the fixed ``HAND_Q_GRASP_OPEN`` pre-grasp pose with one the policy chooses on the grasp
    manifold. Implies the arm-gate (a distinct reach vs grasp regime), so it works whether or
    not ``hand_gate`` is set. Adds z_dim (=5) action dims; no-ops in joints mode."""
    pca_latent_scale: float = 1.0
    """(PCA mode) Stretch the PCA latent range the action spans on each axis, about its centre.
    The action still maps [0, 1] -> the latent range, but the p01/p99 bounds are widened by this
    factor, so >1 lets the policy command more extreme grasp shapes (the decoded 20-joint pose is
    still clamped to the joint limits). 1.0 = the fitted 1st..99th-percentile range."""
    lift_height_offset: float = 0.0
    """Metres to raise the LIFT retreat / hold target (``home_goal_pos``) in world +z, so the arm
    lifts the object higher. The height, hold, and success rewards all follow ``home_goal_pos``,
    so they stay consistent. 0.0 = the FK home grasp_site height."""
    freeze_object_on_engage: bool = False
    """PERCEPTION ABLATION -- once the arm->hand gate engages, hold the ESTIMATED object position
    at whatever it was at that instant instead of tracking the object live.

    Models the thing that actually happens on hardware: the hand closing around the object is
    exactly what occludes it, so a pose estimator loses the target at the moment it matters and
    the arm has to servo on the last good estimate. With this off (the default) the REACH setpoint,
    the engage/release distance and the residual clamp all follow the object every control step.

    Frozen while engaged and live again if the gate releases -- the palm keeps moving, so
    ``dist_po = ||palm - frozen_object||`` still grows when the arm retreats and the Schmitt
    release still fires. That is what keeps drop-recovery reachable: an object knocked out of the
    hand is re-acquired rather than chased at a stale pose forever.

    Composes with ``object_pos_error``: the bias is applied first, so the frozen value is the
    frozen ESTIMATE, not the frozen truth."""
    object_pos_error: float = 0.0
    """SENSITIVITY HOOK -- half-range [m] of a per-episode isotropic bias on the ESTIMATED object
    position. 0.0 (default) is exact perception and changes nothing.

    Models a pose estimator that is biased, not noisy: one draw per episode, held for the whole
    grasp, matching the deploy path where GO GRASP latches the pose once. The physics object does
    NOT move -- only what the robot believes about it.

    The same bias is consumed at BOTH places the estimate is read, so a comparison across
    architectures perturbs one shared quantity rather than two different ones:
      * ``process_actions`` -- the scripted arm's reach target AND its engage gate (they share the
        one tensor), which is the whole dependency for the factorised policy;
      * ``palm_object_vector`` -- the ``palm_to_object`` observation, which is the sole channel
        carrying the estimate to the end-to-end joint-target baseline.
    """
    arm_residual_scale: float = 0.03
    """Max policy offset [m per axis] added to the arm phase target, ported from IsaacLab's
    ``palm_target_residual``. Gives the hand policy a bounded say in where the palm is driven so
    it can fine-tune the approach per object around the fixed tuned pose. Raw residual dims (1:4)
    are clamped to [-1, 1] then scaled by this (+-3 cm)."""
    arm_residual_rot_scale: float = math.radians(5.0)
    """Max policy palm-orientation offset [rad per axis, +-5 deg] added to the arm phase target
    (only when arm_residual). Raw dims (4:7) are clamped to [-1, 1], scaled by this, and applied
    as an euler-XYZ rotation in the palm/body frame, letting the hand tilt the palm per object."""
    hand_preshape: float = 0.0
    hand_v_max: float = 0.8
    """Max finger-joint closing speed [rad/s]: the fingers close quasi-statically so contact
    builds gently (less dynamic overshoot and fewer force spikes). Note the *steady* grip force
    -- and thus penetration -- is set by the hand actuator effort limit, not this; see
    ``HAND_EFFORT_LIMIT`` in assets/rizon4_hand.py. Ported from the IsaacLab hand velocity field."""
    hand_slowdown_distance: float = 0.35
    """Joint error [rad] below which the closing speed ramps down linearly to zero."""
    palm_flatten_euler: tuple[float, float, float] = (0.38, 0.0, 0.0)
    """World-frame XYZ rotation [rad] applied to the FK home palm attitude to hold the palm
    flatter (more level). The home pose from ARM_Q_HOME has the palm grasp axis ~63 deg off
    straight-down; a positive rotation about world X tips it toward horizontal (~85 deg off
    down, i.e. a level/flat palm). Tune in the viewer."""
    squash: str = "sigmoid"
    """How raw policy output maps to [0, 1]: "sigmoid" (gradient everywhere) or "clamp".

    "clamp" is retained only to replay run 1 faithfully; it has zero gradient outside the box
    and is what let run 1's grasp_score get stuck at a raw -14. New runs use "sigmoid".
    """

    def build(self, env) -> GraspAction:
        return GraspAction(self, env)


class GraspAction(ActionTerm):
    """Two scalars: grasp score drives the arm phase, hand closure drives the hand.

    Nothing routes between them. The arm's only input from the hand is the scalar score, which
    is the property that keeps the grasp reflex arm-agnostic.

    The setpoint is latched in ``process_actions`` at policy rate while the impedance law runs
    in ``apply_actions`` at physics rate, matching how the IsaacLab term split the two.
    """

    cfg: GraspActionCfg

    def __init__(self, cfg: GraspActionCfg, env) -> None:
        super().__init__(cfg, env)
        # A standalone compile of the robot spec, not env.sim.mj_model: EntityData arrays are
        # entity-local (39 robot bodies) while the scene model also holds the object and world
        # (41), so scene-global indices would silently misalign. The attached spec cannot be
        # recompiled in place, hence a fresh Entity from the same cfg -- identical ordering.
        import mujoco
        from mjlab.entity.entity import Entity as _Entity

        model = _Entity(_grasp_robot_cfg()).spec.compile()
        self._object = env.scene[cfg.object_name]
        self._num = env.num_envs

        # Home palm pose, computed once by FK at ARM_Q_HOME. This is the fixed hold pose the
        # arm returns to on LIFT and the fixed orientation held throughout the episode.
        #
        # It is NOT read from live state at reset. mjlab's _reset_idx runs the joint-reset
        # event and then action_manager.reset() with no sim.forward() between, so site_pos_w /
        # site_quat_w read in reset() are stale -- they reflect the *previous* episode's end
        # pose. Capturing from them gave a hold orientation 100+ deg off the home pose and
        # different every episode, which is why the hand orientation drifted in playback.
        _data = mujoco.MjData(model)
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            _data.qpos[model.jnt_qposadr[jid]] = ARM_Q_HOME[name]
        mujoco.mj_forward(model, _data)
        _gsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.site_name)
        import numpy as _np

        _home_quat = _np.zeros(4)
        mujoco.mju_mat2Quat(_home_quat, _data.site_xmat[_gsid])
        # Flatten the held palm attitude by a fixed world-frame rotation (left-multiply, so it
        # is applied in world, not palm, coordinates -- "flat" is relative to gravity). Composed
        # X then Y then Z from palm_flatten_euler; the default tips the palm toward palm-down.
        _off = _np.array([1.0, 0.0, 0.0, 0.0])
        for _ax, _ang in enumerate(cfg.palm_flatten_euler):
            if _ang != 0.0:
                _aq = _np.zeros(4)
                _aq[0] = _np.cos(_ang / 2.0)
                _aq[1 + _ax] = _np.sin(_ang / 2.0)
                _tmp = _np.zeros(4)
                mujoco.mju_mulQuat(_tmp, _aq, _off)
                _off = _tmp
        _flat_quat = _np.zeros(4)
        mujoco.mju_mulQuat(_flat_quat, _off, _home_quat)
        self._home_site_pos = torch.as_tensor(
            _data.site_xpos[_gsid], dtype=torch.float32, device=env.device
        ).clone()
        # Raise the LIFT retreat / hold target so the arm carries the object higher (the height,
        # hold, and success rewards all read home_goal_pos, so they follow it).
        self._home_site_pos[2] += float(cfg.lift_height_offset)
        self._home_site_quat = torch.as_tensor(_flat_quat, dtype=torch.float32, device=env.device)

        self.impedance = BatchedArmImpedance(
            model,
            ARM_JOINT_NAMES,
            cfg.site_name,
            env.num_envs,
            device=env.device,
            gains=cfg.impedance,
            body_offset=1,
        )
        self.phase = BatchedGraspPhase(cfg.phase, env.num_envs, device=env.device)
        # The phase counts control steps for the timed-lift baseline; give it the real
        # control period rather than letting it use its fallback.
        self.phase.control_dt = float(env.step_dt)

        ids, _ = self._entity.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        self._arm_joint_ids = ids
        ids, _ = self._entity.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        self._hand_joint_ids = ids

        # Pre-grasp / REACH pose held during approach (and at reset/init): the fixed splayed-open
        # constant HAND_Q_REACH_DEG (radians), clamped to joint limits below. Was HAND_Q_GRASP_OPEN.
        q_open = torch.deg2rad(
            torch.tensor(HAND_Q_REACH_DEG, dtype=torch.float32, device=env.device)
        )
        q_close = torch.tensor([HAND_Q_CLOSE[n] for n in HAND_JOINT_NAMES], device=env.device)
        self._q_open = q_open.unsqueeze(0)
        self._q_close = q_close.unsqueeze(0)

        # Per-joint position limits for the 20 hand joints, from the model. The policy now
        # commands each joint directly; its squashed [0, 1] output maps across [lo, hi] so the
        # full range is reachable. Read from jnt_range (soft factor is 1.0 for this hand).
        lo, hi = [], []
        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo.append(float(model.jnt_range[jid, 0]))
            hi.append(float(model.jnt_range[jid, 1]))
        self._q_lo = torch.tensor(lo, device=env.device).unsqueeze(0)
        self._q_hi = torch.tensor(hi, device=env.device).unsqueeze(0)
        # Clamp the fixed reach pose into the joint limits (in case a splay angle exceeds a range).
        self._q_open = self._q_open.clamp(self._q_lo, self._q_hi)

        # Velocity (Jacobian-pseudoinverse) arm control (arm_control="velocity"): reuses the
        # impedance controller's batched Jacobian + gravity + effort-limit; drives the SAME phase
        # target (set on self.impedance) via a velocity field -> damped pinv -> joint PD. These
        # params are held as mutable instance attrs so play_grasp can tune them live
        # (set_velocity_params); read fresh each apply_actions.
        self._vel = {
            "vf_lin_speed": float(cfg.vf_lin_speed),
            "vf_lin_eps": float(cfg.vf_lin_eps),
            "vf_ang_speed": float(cfg.vf_ang_speed),
            "vf_ang_eps": float(cfg.vf_ang_eps),
            "vel_damping": float(cfg.vel_damping),
            "qd_limit": float(cfg.qd_limit),
        }
        # Per-arm unless the cfg overrides them; see GraspActionCfg.arm_kp.
        _kp = cfg.arm_kp if cfg.arm_kp is not None else ARM.arm_kp
        _kd = cfg.arm_kd if cfg.arm_kd is not None else ARM.arm_kd
        if len(_kp) != len(ARM_JOINT_NAMES) or len(_kd) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"arm_kp/arm_kd have {len(_kp)}/{len(_kd)} entries but {ARM.label} has "
                f"{len(ARM_JOINT_NAMES)} joints."
            )
        self._arm_kp = torch.tensor(_kp, device=env.device).unsqueeze(0)
        self._arm_kd = torch.tensor(_kd, device=env.device).unsqueeze(0)
        # EXPLICIT-PD STABILITY, checked against the plant this term is about to drive.
        #
        # `tau = kp*(qd_des*dt) + kd*(qd_des - qd_meas) + grav` is applied as a torque, so its kd
        # term is integrated explicitly and is stable only while kd*dt < 2*I_eff -- unlike the
        # MJCF's own `damping`, which implicitfast handles implicitly and which is therefore
        # unconditionally stable. At a wrist joint I_eff is dominated by reflected rotor inertia,
        # so an arm model that ships none has almost nothing to divide by.
        #
        # Checked here, at init, because the failure is loud but anonymous: the Menagerie Gen3
        # (armature 0, damping 0, 0.0026 kg m^2 wrist) ran at kd*dt/I = 8, thrashed at 20-55 rad/s
        # against a 2.5 rad/s command clamp, tripped the instability termination every few steps
        # and scored a flat zero -- with nothing anywhere naming a gain or an inertia. `_data` is
        # already forward-kinematicked at the home pose a few lines above, so this costs one
        # mj_fullM.
        if self.cfg.arm_control in (
            "velocity",
            "pose_target",
            "pose_policy",
            "twist_policy",
            "joint_target",
            "joint_direct",
        ):
            from blindgraspreflex.assets.arms import PD_MARGIN_LIMIT, PHYSICS_DT

            _full_m = _np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, _data, _full_m)
            for _n, _k in zip(ARM_JOINT_NAMES, _kd, strict=True):
                _jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _n)
                _dof = model.jnt_dofadr[_jid]
                _inertia = float(_full_m[_dof, _dof]) + float(model.dof_damping[_dof]) * PHYSICS_DT
                _margin = float(_k) * PHYSICS_DT / max(_inertia, 1e-12)
                if _margin >= PD_MARGIN_LIMIT:
                    raise ValueError(
                        f"{ARM.label}: joint {_n!r} would run the explicit velocity PD at "
                        f"kd*dt/I = {_margin:.2f} (bound {PD_MARGIN_LIMIT}) -- effective inertia "
                        f"{_inertia:.5f} kg m^2 against kd {_k:.2f} at dt {PHYSICS_DT}. The arm "
                        "will thrash rather than track. Give the joint reflected rotor inertia "
                        "(ArmSpec.joint_armature) or lower arm_kd."
                    )

        # Hand action space: PCA latent or direct joints. PCA loads the fitted grasp basis and
        # exposes only z_dim (=5) action dims; the decoded 20-vector is clamped to joint limits.
        self._hand_mode = cfg.hand_mode
        if cfg.hand_mode == "pca":
            from blindgraspreflex.assets.hand_pca import load_hand_pca

            self._pca = load_hand_pca(device=env.device)
            self._num_hand = self._pca.z_dim
            # Widen the commandable latent range about its centre (decoded poses stay clamped to
            # joint limits), letting the policy reach more extreme grasp shapes.
            scale = float(cfg.pca_latent_scale)
            if scale != 1.0:
                center = 0.5 * (self._pca.p01 + self._pca.p99)
                half = 0.5 * (self._pca.p99 - self._pca.p01)
                self._pca.p01 = center - scale * half
                self._pca.p99 = center + scale * half
        else:
            self._pca = None
            self._num_hand = len(HAND_JOINT_NAMES)

        # (joints mode) optional projection of the commanded pose toward the PCA manifold, for
        # fine-tuning a joint-space policy to natural PCA-like hand shapes. Blend factor in [0, 1].
        self._pca_project = float(cfg.hand_pca_project)
        self._proj_pca = None
        if self._pca_project > 0.0 and self._pca is None:
            from blindgraspreflex.assets.hand_pca import load_hand_pca

            self._proj_pca = load_hand_pca(device=env.device)

        # Learned REACH-phase hand pose: a second hand block, decoded into the pose the fingers
        # hold while the arm approaches (before engagement), in place of the fixed HAND_Q_GRASP_OPEN.
        # Works in both modes -- a second 5-D PCA latent or a second 20-D joint block. Implies the
        # arm-gate.
        self._reach_hand = bool(cfg.reach_hand_pose)
        self._num_hand_out = self._num_hand * (2 if self._reach_hand else 1)

        # Action layout: [grasp_score(1), (palm pos residual(3) + rot residual(3) if
        # arm_residual), hand(5/20)]. With arm_residual False the hand policy is arm-agnostic --
        # it never touches the arm setpoint; the arm sees only the scalar grasp score, and the
        # REACH pose is randomised instead.
        self._use_residual = cfg.arm_residual
        self._residual_scale = cfg.arm_residual_scale
        self._residual_rot_scale = cfg.arm_residual_rot_scale
        # BASELINE arm mode: the policy commands the 7 arm joints directly. Limits come from the
        # model's jnt_range exactly as the hand's do, so a squashed [0,1] output spans the joint's
        # full reachable range. Read here (not lazily) because action_dim is queried at build time.
        # Physical-unit joint commands from an offline planner: same plant as "joint_target",
        # different source of the numbers, so it shares that mode's buffers below.
        self._joint_direct = self.cfg.arm_control == "joint_direct"
        self._arm_joint_target = self.cfg.arm_control in ("joint_target", "joint_direct")
        # Cartesian palm setpoint straight from an offline planner (the model-based baseline).
        self._pose_target = self.cfg.arm_control == "pose_target"
        # TASK-SPACE BASELINE: the policy commands a palm pose, the same tuned velocity law the
        # deploy stack runs converts it to torque. Shares pose_target's plumbing (set_target ->
        # _velocity_torque); differs in that the action is BOUNDED and squashed, because it comes
        # from a network rather than from a planner emitting physical units.
        self._pose_policy = self.cfg.arm_control == "pose_policy"
        self._twist_policy = self.cfg.arm_control == "twist_policy"
        if self._twist_policy and (
            cfg.hand_mode != "joints" or cfg.hand_pca_project != 0.0 or cfg.reach_hand_pose
        ):
            raise ValueError(
                "twist_policy requires 20 direct hand joints, no PCA projection, "
                "and no reach-pose action block"
            )
        if (self._pose_target or self._joint_direct) and self._hand_mode != "joints":
            raise ValueError(
                f"arm_control={self.cfg.arm_control!r} commands joint angles; use hand_mode='joints'"
            )
        # Per-episode object-position estimate bias (see GraspActionCfg.object_pos_error). Always
        # allocated so the read sites need no branch; identically zero unless the hook is engaged.
        self._obj_pos_bias = torch.zeros(env.num_envs, 3, device=env.device)
        # Latched object-position estimate for freeze_object_on_engage. `_valid` is the previous
        # step's engage state, so a rising edge is detected without a second pass over the phase.
        self._obj_pos_frozen = torch.zeros(env.num_envs, 3, device=env.device)
        self._obj_pos_frozen_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Whether this episode's bias has been drawn yet. mjlab runs reset EVENTS before
        # action_manager.reset(), so a reset event that consumes the estimate (
        # reset_pregrasp_approach) has to be able to draw it first and have reset() not redraw
        # over the top -- which would hand the event one bias and the policy a different one.
        self._bias_drawn = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Interactive override: when set to a (3,) or (num_envs, 3) tensor, the per-episode draw
        # returns THIS instead of a random one, so a GUI can hold the estimate error fixed and
        # have every consumer -- including the reset event, which runs before reset() -- agree on
        # it. None (the default) is the random draw the sweeps use.
        self._bias_override = None
        # Frozen-after-engage perception (REFLEXIVE_FREEZE_OBJ_AFTER_ENGAGE=1). Once the palm comes
        # within the phase's engage_radius of the object, the estimate every consumer reads locks
        # to the object's pose AT THAT MOMENT for the rest of the episode. This exists so the
        # end-to-end baseline can be TRAINED under the same information the factorised policy
        # effectively has -- that one needs the object's pose only at the instant of grasp, since
        # its arm then servos to a fixed goal. Blinding a policy that trained on live pose measures
        # the mismatch, not the architecture; training both under the freeze is the fair test.
        self._freeze_after_engage = bool(os.environ.get("REFLEXIVE_FREEZE_OBJ_AFTER_ENGAGE"))
        self._freeze_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._freeze_pos = torch.zeros(env.num_envs, 3, device=env.device)
        # A STANDING perception error an evaluation can install (believed = true + base). The
        # freeze composes with it rather than overwriting it: what gets latched is the BELIEVED
        # position, so "the estimate is wrong by 4 cm" and "the estimate stops updating" stack the
        # way they would on hardware instead of the second silently cancelling the first.
        self._obj_pos_bias_base = torch.zeros(env.num_envs, 3, device=env.device)
        if self._arm_joint_target:
            alo, ahi = [], []
            for name in ARM_JOINT_NAMES:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                alo.append(float(model.jnt_range[jid, 0]))
                ahi.append(float(model.jnt_range[jid, 1]))
            self._arm_lo = torch.tensor(alo, device=env.device).unsqueeze(0)
            self._arm_hi = torch.tensor(ahi, device=env.device).unsqueeze(0)
            if cfg.arm_joint_span is not None:
                # Centre the commandable range on the home pose, clipped to the model limits so a
                # wide span can never command outside jnt_range.
                home = torch.tensor(
                    [ARM_Q_HOME[n] for n in ARM_JOINT_NAMES], device=env.device
                ).unsqueeze(0)
                span = float(cfg.arm_joint_span)
                self._arm_lo = torch.maximum(home - span, self._arm_lo)
                self._arm_hi = torch.minimum(home + span, self._arm_hi)
            # Seeded at the home pose, NOT from live joint_pos: this runs before any reset event,
            # so live data is all zeros here and the arm would be commanded to the straight-up
            # configuration on the first step. ``reset`` re-seeds it every episode for the same
            # reason.
            self._arm_target = (
                torch.tensor([ARM_Q_HOME[n] for n in ARM_JOINT_NAMES], device=env.device)
                .unsqueeze(0)
                .repeat(env.num_envs, 1)
            )
            # Integrated command the PD actually tracks, rate-limited toward _arm_target. Same
            # structure as the hand's _q_cmd.
            self._arm_cmd = self._arm_target.clone()
            # Per-env flag set by a reset EVENT that has already seeded _arm_cmd itself (see
            # reset_pregrasp_approach). Events run before action_manager.reset, so without this the
            # home-seeding below silently discards the pose they just established.
            self._arm_cmd_seeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            self._arm_v_max = float(cfg.arm_v_max)
            self._arm_slowdown = max(float(cfg.arm_slowdown_distance), 1e-6)
        # Action dims before the hand block: the 7 arm joints in the baseline, else the grasp
        # score (+ 6 residual dims when the palm residual is on).
        self._n_pre = (
            len(ARM_JOINT_NAMES)
            if self._arm_joint_target
            else 7
            if self._pose_target  # palm pos(3) + quat(4)
            else 6
            if self._pose_policy or self._twist_policy  # palm pose or twist (6)
            else (7 if self._use_residual else 1)
        )
        if self._pose_policy:
            self._pose_centre = torch.tensor(
                cfg.pose_box_centre, dtype=torch.float32, device=env.device
            ).unsqueeze(0)
            self._pose_half = torch.tensor(
                cfg.pose_box_half, dtype=torch.float32, device=env.device
            ).unsqueeze(0)
            self._pose_rpy_span = float(cfg.pose_rpy_span)
        self._raw = torch.zeros(env.num_envs, self._n_pre + self._num_hand_out, device=env.device)
        if self._twist_policy:
            self._palm_twist = torch.zeros(env.num_envs, 6, device=env.device)

        # Contact-gate buffers: body ids (entity-local, for body_link_quat_w) and per-link
        # inward axis, in REWARD_CONTACT_LINKS order. The inside-of-hand gate keeps only
        # contacts whose normal points into the link's inward face, local +X for the palm face
        # and the phalanx pads (see reward_contact_frames).
        self._reward_body_ids, self._inward_local = reward_contact_frames(self._entity, env.device)
        # Hand velocity field. hand_target is the *commanded* pose the policy asked for; the
        # position target actually sent each physics step is rate-limited toward it so the
        # fingers close smoothly instead of snapping shut. Ported from IsaacLab's hand action:
        # per-joint speed = v_max * clamp(|err| / slowdown, 1), applied against measured q.
        self.hand_target = self._q_open.repeat(env.num_envs, 1)
        # Internal command pose, integrated open-loop toward hand_target at the capped speed.
        # It is the position target actually sent to the servo. Open-loop (not measured q) so
        # a finger that lags still gets a growing position error -- and therefore real torque
        # from the soft hand servo -- while the command itself stays rate-limited.
        self._q_cmd = self._q_open.repeat(env.num_envs, 1)
        self._v_max = cfg.hand_v_max
        self._slowdown = max(cfg.hand_slowdown_distance, 1e-6)
        # Per-ENV velocity-field parameters, so they can be domain-randomised per episode. Seeded
        # from the scalar cfg, so behaviour is identical until randomise_hand_vf() is called.
        # These matter for sim-to-real: the deployed low_level runs v_max 1.8 / slowdown 0.2 against
        # training's 0.8 / 0.35, which changes the finger-closing time constant by ~4x.
        self._v_max_env = torch.full((env.num_envs,), float(self._v_max), device=env.device)
        self._slowdown_env = torch.full((env.num_envs,), float(self._slowdown), device=env.device)
        # Per-env hand PD gains. In EXPLICIT mode these ARE the servo (this term computes tau);
        # in the default implicit mode they mirror the actuator's gains so tau can still be
        # reported. Per-env so gain DR writes here instead of into sim.model -- see
        # randomize_hand_pd, which replaces dr.pd_gains when explicit mode is on.
        from blindgraspreflex.assets.rizon4_hand import (
            HAND_EFFORT_LIMIT,
            HAND_EXPLICIT_PD,
            HAND_KD,
            HAND_KP,
        )

        self._explicit_pd = HAND_EXPLICIT_PD
        self._hand_kp_env = torch.tensor(HAND_KP, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        self._hand_kd_env = torch.tensor(HAND_KD, dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        self._hand_effort = float(HAND_EFFORT_LIMIT)
        # Last commanded hand torque (num_envs, 20), for the observation and for analysis.
        self._hand_tau = torch.zeros(env.num_envs, len(HAND_JOINT_NAMES), device=env.device)
        # --- online M9 contact metric -------------------------------------------------------
        # |tau| divided by the joint's OWN free-running torque (Coulomb friction), which is
        # independent of kp -- a softer joint carries a proportionally larger position error to
        # make the same torque. Mirrors blindgraspreflex/contact_metrics.py.
        #
        # OFF unless REFLEXIVE_M9_OBS=1 (build_env sets it for --student-obs m9). It is not free:
        # left always-on it cost every run a ring buffer and a periodic quantile for a signal
        # nothing was reading.
        #
        # The ring is sampled at the STRIDE rate (20 Hz), not the physics rate. At 6144 envs a
        # per-physics-step ring is 246 MB, its masked copy another 246 MB, and torch.quantile then
        # sorts 61M elements -- which OOM'd a 46 GB L40S. 40 samples over the 2 s window is ample
        # for a 20th percentile of a slowly varying quantity, and is 12x smaller.
        _nj = len(HAND_JOINT_NAMES)
        # REFLEXIVE_HAND_QCMD_DIRECT=1 makes the velocity field a pass-through.
        self._qcmd_passthrough = bool(os.environ.get("REFLEXIVE_HAND_QCMD_DIRECT"))
        self._m9_on = bool(os.environ.get("REFLEXIVE_M9_OBS"))
        self._m9_stride = max(1, int(round(0.05 / env.physics_dt)))
        self._m9_hist_n = max(8, int(round(2.0 / 0.05)))
        self._m9_alpha = min(1.0, env.physics_dt / 0.15)  # EMA for the offline 0.15 s boxcar
        # Measured-velocity threshold for "free running". Well under the velocity field's v_max
        # (0.8-1.8 rad/s) and well above encoder/solver noise.
        self._m9_min_qd = 0.05
        self._m9 = torch.zeros(env.num_envs, _nj, device=env.device)
        if self._m9_on:
            self._m9_buf = torch.zeros(self._m9_hist_n, env.num_envs, _nj, device=env.device)
            self._m9_mov = torch.zeros(
                self._m9_hist_n, env.num_envs, _nj, dtype=torch.bool, device=env.device
            )
            self._m9_base = torch.full((env.num_envs, _nj), float("nan"), device=env.device)
            self._m9_ptr = 0
            self._m9_count = 0
        # Optional PHYSICS-RATE trace (see start_trace). None on the normal path, so training and
        # inference pay nothing for it. Recording from the env.step loop instead only sees the
        # CONTROL rate (~12 Hz at decimation 21), which is too coarse to resolve the tau/qdot
        # relationship the deadband analysis needs -- hardware logs at 500 Hz+ and the comparison
        # would be rate-asymmetric.
        self._trace = None
        self._vf_dr = None  # (v_lo, v_hi, s_lo, s_hi) once enabled
        self._hand_dt = env.physics_dt
        self._site_id = self.impedance.site_id
        self._lift_captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def action_dim(self) -> int:
        # grasp + [3 residual] + hand block (5 PCA or 20 joints), x2 when reach_hand_pose adds a
        # second hand latent for the reach pose.
        return self._n_pre + self._num_hand_out

    @property
    def home_goal_pos(self) -> torch.Tensor:
        """Fixed end-effector goal position (the LIFT retreat pose). Shape (3,)."""
        return self._home_site_pos

    @property
    def grasp_score(self) -> torch.Tensor:
        """The squashed grasp score in [0, 1] the policy last output. Shape (num_envs,).

        This is the scalar the arm phase controller sees -- <0.5 reach, >0.5 lift -- computed
        from the raw action the same way ``process_actions`` does, so a viewer reads exactly
        what drives the phase latch.
        """
        if self._pose_target or self._joint_direct:
            # Dim 0 is a palm coordinate or an arm joint angle here, not a score. Reported as zero
            # so a viewer or an eval loop that logs grasp_score records "no score" rather than a
            # metre/radian value.
            return torch.zeros(self._num, device=self.device)
        raw = self._raw[:, 0]
        return torch.sigmoid(raw) if self.cfg.squash == "sigmoid" else raw.clamp(0.0, 1.0)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def _m9_update(self, qd_hand) -> None:
        """Advance the online M9 estimate by one physics step (no-op unless enabled)."""
        if not self._m9_on:
            return
        tau = self._hand_tau.abs()
        # "moving" = the JOINT IS ACTUALLY MOVING (measured), not merely commanded to.
        #
        # Gating on the command instead looks reasonable and is wrong: during contact the command
        # keeps advancing while the joint is stalled, so contact torque is admitted to the very
        # baseline it is supposed to be measured against. Under a sustained hold the 2 s window then
        # fills with contact samples, the p20 rises to the contact torque, and the metric decays
        # back to 1.0 exactly when it should read highest. Measured motion selects free-running
        # samples and excludes stalled ones, so the baseline freezes at its free-air value.
        self._m9_count += 1
        if self._m9_count % self._m9_stride == 0:
            moving = qd_hand.abs() > self._m9_min_qd
            self._m9_buf[self._m9_ptr] = tau
            self._m9_mov[self._m9_ptr] = moving
            self._m9_ptr = (self._m9_ptr + 1) % self._m9_hist_n
            n = min(self._m9_count // self._m9_stride, self._m9_hist_n)
            buf, mov = self._m9_buf[:n], self._m9_mov[:n]
            enough = mov.sum(dim=0) >= 8
            if bool(enough.any()):
                # Exact 20th percentile OVER THE VALID SAMPLES ONLY, per (env, joint).
                #
                # Padding the invalid slots with a large sentinel and calling torch.quantile does
                # NOT work: the quantile is then taken over valid+sentinel, so once free-running
                # samples fall below ~20% of the window the 20th percentile lands inside the
                # sentinel region and the baseline jumps to it. (numpy's nanquantile ignores NaNs
                # and has no such problem, which is why the offline metric was unaffected.)
                # Sorting puts +inf last, so index floor(q*(count-1)) is the right element whatever
                # the per-element valid count happens to be.
                masked = torch.where(mov, buf, torch.full_like(buf, float("inf")))
                srt, _ = masked.sort(dim=0)
                cnt = mov.sum(dim=0)
                idx = ((cnt - 1).clamp(min=0).float() * 0.2).long()
                q = srt.gather(0, idx.unsqueeze(0)).squeeze(0)
                self._m9_base = torch.where(enough & torch.isfinite(q), q, self._m9_base)
        ratio = torch.where(
            torch.isfinite(self._m9_base) & (self._m9_base > 1e-6),
            tau / self._m9_base.clamp(min=1e-6),
            torch.ones_like(tau),  # before a baseline exists, report "free running"
        ).clamp(0.0, 20.0)
        self._m9 = self._m9 + self._m9_alpha * (ratio - self._m9)

    def _m9_reset(self, env_ids) -> None:
        self._m9[env_ids] = 1.0
        if not self._m9_on:
            return
        self._m9_buf[:, env_ids] = 0.0
        self._m9_mov[:, env_ids] = False
        self._m9_base[env_ids] = float("nan")

    def start_trace(self, env_ids, max_steps: int) -> None:
        """Begin recording q/qd/q_cmd/hand_target/tau every PHYSICS step for ``env_ids``.

        Ring-free: recording simply stops once ``max_steps`` physics steps are captured, so a
        short buffer truncates rather than silently overwriting the beginning of the run.
        Subsample the envs -- 250 Hz x 20 joints x 5 signals adds up fast, and a handful of worlds
        is plenty to characterise a plant.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n, j = len(ids), len(HAND_JOINT_NAMES)
        self._trace = {
            "ids": ids,
            "cursor": 0,
            "max": int(max_steps),
            **{
                k: torch.zeros(int(max_steps), n, j, device=self.device)
                for k in ("q", "qd", "q_cmd", "hand_target", "tau_cmd")
            },
        }

    def _trace_write(self, q_hand, qd_hand) -> None:
        tr = self._trace
        i = tr["cursor"]
        if i >= tr["max"]:
            return
        ids = tr["ids"]
        tr["q"][i] = q_hand[ids]
        tr["qd"][i] = qd_hand[ids]
        tr["q_cmd"][i] = self._q_cmd[ids]
        tr["hand_target"][i] = self.hand_target[ids]
        tr["tau_cmd"][i] = self._hand_tau[ids]
        tr["cursor"] = i + 1

    def drain_trace(self) -> dict:
        """Return the captured physics-rate arrays as numpy and clear the buffer."""
        tr = self._trace
        n = tr["cursor"]
        out = {k: tr[k][:n].cpu().numpy() for k in ("q", "qd", "q_cmd", "hand_target", "tau_cmd")}
        out["env_ids"] = tr["ids"].cpu().numpy()
        self._trace = None
        return out

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        # Fixed home pose, not live state: see __init__. Broadcasting the same constant to
        # every reset env keeps the retreat position and the held orientation identical to the
        # default pose every episode. env_origins are zero under mujoco-warp (each env is its
        # own world), so no per-env position offset is needed.
        n = env_ids.numel()
        # Perception bias for the episodes starting now, drawn once each -- unless a reset event
        # already drew it (see resample_object_pos_bias), in which case that draw stands and is
        # the one every consumer sees.
        pending = env_ids[~self._bias_drawn[env_ids]]
        if pending.numel():
            self.resample_object_pos_bias(pending)
        self._bias_drawn[env_ids] = False
        # Drop any frozen estimate: a new episode starts with perception live again.
        self._obj_pos_frozen_valid[env_ids] = False
        site_pos = self._home_site_pos.unsqueeze(0).expand(n, 3)
        site_quat = self._home_site_quat.unsqueeze(0).expand(n, 4)
        self.phase.reset(env_ids, site_pos, site_quat)
        self.impedance.set_target(site_pos, site_quat, env_ids)
        self._raw[env_ids] = 0.0
        if self._twist_policy:
            self._palm_twist[env_ids] = 0.0
        self.hand_target[env_ids] = self._q_open
        self._q_cmd[env_ids] = self._q_open
        self._hand_tau[env_ids] = 0.0
        if self._arm_joint_target:
            # Re-seed the arm's rate-limited command at the pose the reset event just put the arm
            # in. Without this it carries over -- from the previous episode's final pose, or, on
            # the very first step, from the all-zero joint_pos this term read at construction
            # before any reset had run. The PD tracks _arm_cmd, not the action, so a stale command
            # is a step input to the plant: measured, 1.745 rad of error the instant the episode
            # begins and a peak of 7.6 rad/s while the action asked for the home pose it was
            # already in. The arm velocity field caps the COMMAND's rate, not the joint's, so it
            # does not bound this at all.
            #
            # Seeded from the constant rather than from live state for the same reason __init__
            # captures the home site pose from FK: _reset_idx runs the joint-reset event and then
            # action_manager.reset() with no sim.forward() between, so live reads here are not
            # guaranteed to reflect the reset that just happened. ARM_Q_HOME is what that event
            # restores (verified: max|q - home| = 0.0000 immediately after reset).
            #
            # ...but ONLY for envs a reset event has not already seeded. mjlab runs reset EVENTS
            # before action_manager.reset(), so reset_pregrasp_approach has already placed the arm
            # at its IK standoff pose and set _arm_cmd to match. Overwriting that with home
            # recreated the very defect this block exists to remove -- a large |_arm_cmd - q| at
            # reset, 1.0693 rad, now pulling the arm OFF the pre-grasp pose and back toward home
            # every episode. That cost a full overnight run: `hold` sat at exactly 0.0000 for 3521
            # iterations while the policy could still collect the reach reward it started next to.
            home = torch.tensor(
                [ARM_Q_HOME[n] for n in ARM_JOINT_NAMES], device=self.device
            ).unsqueeze(0)
            unseeded = env_ids[~self._arm_cmd_seeded[env_ids]]
            if unseeded.numel():
                self._arm_cmd[unseeded] = home
                self._arm_target[unseeded] = home
            # Consume the marks so the next episode re-seeds unless an event claims it again.
            self._arm_cmd_seeded[env_ids] = False
        self._m9_reset(env_ids)

    def resample_object_pos_bias(self, env_ids) -> None:
        """Draw the per-episode object-position estimate bias for ``env_ids``.

        Public because a reset EVENT may need the episode's bias before ``reset()`` runs -- mjlab
        applies reset events first. Marks the draw so ``reset()`` does not redraw over it; the
        mark is cleared in ``reset()`` for the next episode.

        Always writes, including the zeros at the 0.0 default, so lowering ``object_pos_error``
        back to 0 cannot leave a stale bias in place.
        """
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = int(env_ids.numel())
        if n == 0:
            return
        if self._bias_override is not None:
            ov = torch.as_tensor(
                self._bias_override, dtype=torch.float32, device=self.device
            ).reshape(-1, 3)
            # A PER-WORLD override (num_envs rows) is indexed by env_ids; a single (1,3) row
            # broadcasts. Without the first case a partial reset -- the normal case, since worlds
            # finish at different times -- would try to expand num_envs rows onto len(env_ids)
            # and raise, or silently apply the wrong world's offset.
            self._obj_pos_bias[env_ids] = (
                ov[env_ids] if ov.shape[0] == self._num else ov.expand(n, 3)
            )
        else:
            e = float(self.cfg.object_pos_error)
            self._obj_pos_bias[env_ids] = (
                (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * e
                if e
                else torch.zeros(n, 3, device=self.device)
            )
        self._bias_drawn[env_ids] = True
        if self._freeze_after_engage:
            # A new episode sees live pose again until its hand reaches the new object.
            self._freeze_latched[env_ids] = False

    def _decode_hand(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw hand action -> commanded 20-joint pose. Shared by every arm mode.

        Promoted from a closure inside ``process_actions`` so the baseline joint-target path
        decodes the hand identically rather than reimplementing it: the whole point of that
        baseline is that only the ARM differs.
        """
        a = torch.sigmoid(raw) if self.cfg.squash == "sigmoid" else raw.clamp(0.0, 1.0)
        if self._pca is not None:
            return self._pca.decode(self._pca.latent_from_alpha(a)).clamp(self._q_lo, self._q_hi)
        q = self._q_lo + a * (self._q_hi - self._q_lo)
        if self._proj_pca is not None:
            # Blend the joint target toward its PCA-manifold projection (natural grasp shape).
            recon = self._proj_pca.reconstruct(q).clamp(self._q_lo, self._q_hi)
            q = (1.0 - self._pca_project) * q + self._pca_project * recon
        return q

    def process_actions(self, actions: torch.Tensor) -> None:
        # Squash to [0, 1] with a sigmoid, not clamp. clamp has zero gradient outside the
        # box, so once the policy drifts a channel past the edge it is stuck there with no
        # signal to return -- run 1 drove grasp_score to a raw mean of -14 and, with action
        # std ~0.9, every sample clamped to 0, permanently disabling the LIFT phase. sigmoid
        # keeps gradient and exploration alive across the whole real line.
        actions = torch.nan_to_num(actions, nan=0.0)
        self._raw[:] = actions
        if self._twist_policy:
            limits = actions.new_tensor(
                [self.cfg.twist_linear_limit] * 3 + [self.cfg.twist_angular_limit] * 3
            )
            self._palm_twist = torch.tanh(actions[:, :6]) * limits
            self.hand_target = self._decode_hand(actions[:, 6 : 6 + self._num_hand])
            return
        if self._freeze_after_engage and self._bias_override is None:
            # Trigger on the TRUE palm-to-object distance. The phase's own `engaged` flag is
            # derived from the BIASED estimate and does not update at all when the policy drives
            # the arm (pose_policy), so it would latch at different times for different arm modes.
            _obj = self._object.data.root_link_pos_w
            _believed = _obj + self._obj_pos_bias_base
            _palm = self._entity.data.site_pos_w[:, self._site_id]
            # Trigger on the TRUE distance so the latch fires at the same moment however wrong the
            # estimate is; freeze the BELIEVED position, so a standing bias survives the freeze.
            _hit = (~self._freeze_latched) & (
                torch.linalg.norm(_palm - _obj, dim=-1) < self.phase.cfg.engage_radius
            )
            self._freeze_pos[_hit] = _believed[_hit]
            self._freeze_latched |= _hit
            self._obj_pos_bias[:] = torch.where(
                self._freeze_latched.unsqueeze(-1),
                self._freeze_pos - _obj,
                self._obj_pos_bias_base,
            )
        if self._pose_policy:
            # TASK-SPACE BASELINE: [palm pos(3), palm rpy offset(3), hand joints(20)].
            # tanh, not sigmoid: zero action must mean "centre of the box / flat-hold", so the
            # untrained policy starts pointing somewhere sane instead of at a corner. Position
            # spans the workspace box; attitude is an offset applied to the flat-hold home
            # attitude, so the palm stays in the palm-down band the grasp actually needs.
            pos = self._pose_centre + torch.tanh(actions[:, 0:3]) * self._pose_half
            rpy = torch.tanh(actions[:, 3:6]) * self._pose_rpy_span
            half = 0.5 * rpy
            cx, cy, cz = half[:, 0].cos(), half[:, 1].cos(), half[:, 2].cos()
            sx, sy, sz = half[:, 0].sin(), half[:, 1].sin(), half[:, 2].sin()
            dq = torch.stack(
                [
                    cx * cy * cz + sx * sy * sz,
                    sx * cy * cz - cx * sy * sz,
                    cx * sy * cz + sx * cy * sz,
                    cx * cy * sz - sx * sy * cz,
                ],
                dim=-1,
            )
            quat = quat_mul(dq, self._home_site_quat.unsqueeze(0).expand_as(dq))
            quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
            self.impedance.set_target(pos, quat)
            self.hand_target = self._decode_hand(
                actions[:, self._n_pre : self._n_pre + self._num_hand]
            )
            return
        if self._pose_target:
            # BASELINE (model-based): [grasp_site pos(3), quat wxyz(4), hand joints(20)], all in
            # physical units. No squash and no phase update -- the planner owns both the palm
            # trajectory and the finger targets, and the arm law is the same one arm_control
            # "velocity" runs. Joint targets are clamped to the model's limits, which is the only
            # thing standing between an out-of-range planner value and a silently saturated servo.
            quat = actions[:, 3:7]
            quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-6)
            self.impedance.set_target(actions[:, 0:3], quat)
            self.hand_target = actions[:, 7 : 7 + self._num_hand].clamp(self._q_lo, self._q_hi)
            return
        if self._joint_direct:
            # BASELINE (model-based): [arm(7), hand(20)] in radians, no squash. Clamped to the
            # model's own ranges, which is all that stands between an IK solution that grazes a
            # limit and a command outside it.
            self._arm_target = actions[:, : self._n_pre].clamp(self._arm_lo, self._arm_hi)
            self.hand_target = actions[:, self._n_pre : self._n_pre + self._num_hand].clamp(
                self._q_lo, self._q_hi
            )
            return
        if self._arm_joint_target:
            # BASELINE: [arm(7), hand(20)]. The arm block squashes to [0,1] and maps across each
            # joint's range, the same decode the hand block uses. No grasp score and no phase
            # update -- the policy owns the arm, so there is no REACH/LIFT latch to drive and no
            # hand gate to hold the fingers open (that gate exists to wait for the SCRIPTED arm).
            a_arm = (
                torch.sigmoid(actions[:, : self._n_pre])
                if self.cfg.squash == "sigmoid"
                else actions[:, : self._n_pre].clamp(0.0, 1.0)
            )
            self._arm_target = self._arm_lo + a_arm * (self._arm_hi - self._arm_lo)
            self.hand_target = self._decode_hand(
                actions[:, self._n_pre : self._n_pre + self._num_hand]
            )
            return
        # grasp score (dim 0): squashed to [0, 1] -- drives the arm phase latch.
        raw_score = actions[:, 0]
        score = (
            torch.sigmoid(raw_score) if self.cfg.squash == "sigmoid" else raw_score.clamp(0.0, 1.0)
        )
        # palm residual (only when arm_residual): position (dims 1:4) and orientation (dims 4:7),
        # each clamped to [-1, 1] (symmetric, so zero action = no offset -- NOT sigmoid, which
        # would bias the palm) then scaled to metres / radians. With the arm-agnostic hand policy
        # there is no residual and the arm ignores the hand.
        if self._use_residual:
            residual = actions[:, 1:4].clamp(-1.0, 1.0) * self._residual_scale
            rot_residual = actions[:, 4:7].clamp(-1.0, 1.0) * self._residual_rot_scale
        else:
            residual = None
            rot_residual = None
        # hand (dims _n_pre:): squashed to [0, 1]. PCA: [0,1] -> latent in [p01, p99] -> decode to
        # a coordinated 20-joint pose, clamped to limits. Joints: [0,1] mapped per joint to
        # [lo, hi]. Either way this is the *commanded* pose; the velocity field in apply_actions
        # rate-limits toward it, so the fingers close smoothly rather than snapping.
        _decode_hand = self._decode_hand

        # Hand block. The first _num_hand dims are the GRASP pose; when reach_hand_pose is on a
        # second _num_hand block is the REACH pose held until the arm engages the object.
        grasp_target = _decode_hand(actions[:, self._n_pre : self._n_pre + self._num_hand])
        if self._reach_hand:
            reach_target = _decode_hand(actions[:, self._n_pre + self._num_hand :])
        else:
            reach_target = self._q_open  # fixed pre-grasp pose (used by the hand_gate path)
        # The ESTIMATED object position, not the true one: the scripted arm servos to this and the
        # engage gate latches on it, so one bias covers the factorised policy's entire dependence
        # on perception. Zero unless the sensitivity hook is engaged.
        object_pos = self._object.data.root_link_pos_w + self._obj_pos_bias
        if self.cfg.freeze_object_on_engage:
            # self.phase.engaged is LAST step's latch (phase.update below is what advances it), so
            # this freezes on the first step at which the hand was already engaged -- the step the
            # fingers begin closing, which is the moment occlusion starts on hardware.
            eng = self.phase.engaged.bool()
            rising = eng & ~self._obj_pos_frozen_valid
            self._obj_pos_frozen = torch.where(
                rising.unsqueeze(-1), object_pos, self._obj_pos_frozen
            )
            # STICKY for the whole episode: latched on the FIRST engage and never re-latched, even
            # if the gate releases and the hand re-grasps. Cleared only at reset().
            #
            # The earlier version cleared the latch on release, so every re-grasp re-latched at the
            # object's CURRENT true position -- handing the arm a free, perfect perception update
            # at exactly the moment the policy was fumbling, which is the opposite of what
            # occlusion does. A re-grasp now aims at the same first estimate as the first attempt.
            #
            # The cost is deliberate: an object knocked well away is never re-acquired, so the arm
            # keeps reaching for where it last believed the object to be. That IS the hardware
            # behaviour being modelled -- a camera the hand is occluding does not recover just
            # because the fingers opened.
            self._obj_pos_frozen_valid = self._obj_pos_frozen_valid | eng
            object_pos = torch.where(
                self._obj_pos_frozen_valid.unsqueeze(-1), self._obj_pos_frozen, object_pos
            )
        palm_pos = self._entity.data.site_pos_w[:, self._site_id]
        target, target_quat = self.phase.update(score, object_pos, palm_pos, residual, rot_residual)
        # Arm-gated hand: until the arm brings the palm within engage_radius of the object, hold the
        # fingers at the reach pose (learned when reach_hand_pose, else the fixed pre-grasp pose)
        # instead of the grasp pose. phase.engaged is the latched proximity flag the phase just
        # updated; the velocity field smooths the switch. reach_hand_pose forces this gate on.
        if self._reach_hand or self.cfg.hand_gate:
            self.hand_target = torch.where(
                self.phase.engaged.unsqueeze(-1), grasp_target, reach_target
            )
        else:
            self.hand_target = grasp_target
        # REACH-target override (tune-reach GUI only): drive the palm to an externally-set pose
        # instead of the phase target, so the offset sliders can pose the real hand by hand.
        override_pos = getattr(self, "_reach_override_pos", None)
        if override_pos is not None:
            target, target_quat = override_pos, self._reach_override_quat
        self.impedance.set_target(target, target_quat)

    def set_velocity_params(self, **kw) -> None:
        """Live-update the velocity-control params (for play_grasp tuning). Recognised keys:
        vf_lin_speed, vf_lin_eps, vf_ang_speed, vf_ang_eps, vel_damping, qd_limit, arm_kp, arm_kd."""
        for k in (
            "vf_lin_speed",
            "vf_lin_eps",
            "vf_ang_speed",
            "vf_ang_eps",
            "vel_damping",
            "qd_limit",
        ):
            if k in kw and kw[k] is not None:
                self._vel[k] = float(kw[k])
        for k in ("arm_kp", "arm_kd"):
            if kw.get(k) is not None:
                t = torch.as_tensor(kw[k], dtype=torch.float32, device=self._arm_kp.device).reshape(
                    1, -1
                )
                setattr(self, f"_{k}", t)

    @staticmethod
    def _vf_speed(err_norm: torch.Tensor, v: float, eps: float) -> torch.Tensor:
        """Quadratic-decay velocity-field speed: ``v`` when far (err>eps), ramping to 0 within eps.
        Batched form of ``utils.velocity_fields.grasping.goto_x_des_with_v_quad_decay_from_eps``."""
        near = v * (1.0 - (err_norm - eps) ** 2 / (eps**2))
        speed = torch.where(err_norm > eps, torch.full_like(err_norm, float(v)), near)
        return torch.where(err_norm < 1e-6, torch.zeros_like(err_norm), speed)

    def _velocity_torque(self, data) -> torch.Tensor:
        """Jacobian-pseudoinverse velocity arm control (the deploy method), batched over envs.

        Drives the grasp_site to the phase target latched on ``self.impedance`` via a Cartesian
        velocity field -> damped Jacobian pseudoinverse (NO QP) -> joint PD + gravity feedforward.
        Reuses the impedance controller's batched Jacobian, gravity and effort limit; only the
        control law differs. Params are read from ``self._vel`` so play_grasp can tune them live.
        """
        p = self._vel
        site_pos = data.site_pos_w[:, self._site_id]  # (N, 3)
        site_quat = data.site_quat_w[:, self._site_id]  # (N, 4)
        J = self.impedance.jacobian.compute(
            data.body_link_pos_w, data.body_link_quat_w, site_pos
        )  # (N, 6, 7), rows [lin; ang]

        if self._twist_policy:
            twist = self._palm_twist.unsqueeze(-1)
        else:
            # Cartesian velocity field toward the phase target (position + orientation).
            pos_err = self.impedance.target_pos - site_pos
            e_pos = torch.linalg.norm(pos_err, dim=-1, keepdim=True).clamp(min=1e-9)
            v_des = self._vf_speed(e_pos, p["vf_lin_speed"], p["vf_lin_eps"]) * (pos_err / e_pos)
            rot_err = quat_to_rotvec(
                quat_mul(self.impedance.target_quat, quat_conjugate(site_quat))
            )
            e_rot = torch.linalg.norm(rot_err, dim=-1, keepdim=True).clamp(min=1e-9)
            w_des = self._vf_speed(e_rot, p["vf_ang_speed"], p["vf_ang_eps"]) * (rot_err / e_rot)
            twist = torch.cat([v_des, w_des], dim=-1).unsqueeze(-1)  # (N, 6, 1)

        # Damped Jacobian (right) pseudoinverse: qd = J^T (J J^T + lambda^2 I)^-1 twist. The 6x6
        # solve is the minimum-norm joint velocity for the desired twist -- fast, batched, no QP.
        JJt = J @ J.transpose(1, 2) + p["vel_damping"] * torch.eye(6, device=J.device).unsqueeze(0)
        qd = (J.transpose(1, 2) @ torch.linalg.solve(JJt, twist)).squeeze(-1)  # (N, 7)
        qd = torch.clamp(qd, -p["qd_limit"], p["qd_limit"])

        # Joint PD (+ gravity FF) tracking q_des = q + qd*dt -- the law the deploy plant applies.
        qd_meas = data.joint_vel[:, self._arm_joint_ids]
        grav = self.impedance.gravity.compute(
            data.body_link_pos_w, data.body_link_quat_w, data.body_com_pos_w
        )
        tau = self._arm_kp * (qd * self._hand_dt) + self._arm_kd * (qd - qd_meas) + grav
        return torch.clamp(tau, -self.impedance.effort_limit, self.impedance.effort_limit)

    def apply_actions(self) -> None:
        data = self._entity.data
        if self._arm_joint_target:
            # Same plant as _velocity_torque's final stage -- joint PD + gravity feed-forward,
            # same gains, same effort clamp -- with q_des coming from the policy instead of from
            # the Jacobian velocity field. Keeping the plant identical is what makes this a
            # controller comparison rather than a plant comparison.
            q_arm = data.joint_pos[:, self._arm_joint_ids]
            qd_arm = data.joint_vel[:, self._arm_joint_ids]
            # Arm velocity field: per-joint saturated proportional speed, integrated at the
            # physics dt -- identical in form to the hand's, so both plants rate-limit the same
            # way and the policy commands a POSE rather than an impulse.
            a_err = self._arm_target - self._arm_cmd
            a_speed = self._arm_v_max * (a_err.abs() / self._arm_slowdown).clamp(max=1.0)
            self._arm_cmd = self._arm_cmd + torch.sign(a_err) * a_speed * self._hand_dt
            grav = self.impedance.gravity.compute(
                data.body_link_pos_w, data.body_link_quat_w, data.body_com_pos_w
            )
            torque = torch.clamp(
                self._arm_kp * (self._arm_cmd - q_arm) - self._arm_kd * qd_arm + grav,
                -self.impedance.effort_limit,
                self.impedance.effort_limit,
            )
        elif self.cfg.arm_control in ("velocity", "pose_target", "pose_policy", "twist_policy"):
            torque = self._velocity_torque(data)
        else:
            torque = self.impedance.compute(
                data.body_link_pos_w,
                data.body_link_quat_w,
                data.site_pos_w[:, self._site_id],
                data.site_quat_w[:, self._site_id],
                data.joint_vel[:, self._arm_joint_ids],
                data.body_com_pos_w,
            )
        self._entity.set_joint_effort_target(torque, joint_ids=self._arm_joint_ids)

        # Velocity field on the internal command: per-joint saturated proportional speed,
        # v_max * clamp(|err| / slowdown, 1), integrated at the physics dt. A finger blocked by
        # the object stalls at its measured position while _q_cmd keeps advancing to the
        # commanded pose, so the position error -- and grip torque -- builds smoothly rather
        # than snapping shut.
        if self._qcmd_passthrough:
            # Diagnostic mode: q_cmd IS the commanded pose, no velocity field. Removes the
            # target->q_cmd stage entirely so a comparison isolates the servo and the plant.
            self._q_cmd = self.hand_target.clone()
        else:
            err = self.hand_target - self._q_cmd
            speed = self._v_max_env.unsqueeze(-1) * (
                err.abs() / self._slowdown_env.unsqueeze(-1)
            ).clamp(max=1.0)
            self._q_cmd = self._q_cmd + torch.sign(err) * speed * self._hand_dt
        # tau is computed either way so it is observable; only who APPLIES it differs.
        # Same expression as SimRobotis5FHandZ.apply_sim_control and the XM335 servo, with
        # qd_des = 0 and tau_ff = 0 (the hand gets no gravity feed-forward on either side).
        q_hand = self._entity.data.joint_pos[:, self._hand_joint_ids]
        qd_hand = self._entity.data.joint_vel[:, self._hand_joint_ids]
        self._hand_tau = (
            self._hand_kp_env * (self._q_cmd - q_hand) - self._hand_kd_env * qd_hand
        ).clamp(-self._hand_effort, self._hand_effort)
        if self._explicit_pd:
            self._entity.set_joint_effort_target(self._hand_tau, joint_ids=self._hand_joint_ids)
        else:
            self._entity.set_joint_position_target(self._q_cmd, joint_ids=self._hand_joint_ids)
        self._m9_update(qd_hand)
        if self._trace is not None:
            self._trace_write(q_hand, qd_hand)


##
# Shared per-env state.
##


def _reward_contact_weights() -> tuple:
    """Per-link weights for the contact-coverage reward (in REWARD_CONTACT_LINKS order).

    Default = the IsaacLab weights (CONTACT_WEIGHTS: thumb up, palm 2). If REFLEXIVE_EQUAL_CONTACTS
    is set, every link weighs 1.0 (equal) and REFLEXIVE_PALM_CONTACT_BONUS (default 0) is ADDED to
    the palm link -- e.g. bonus 5 -> palm 6, all others 1, so the reward equally credits contacting
    any link but pushes hardest for seating the object against the palm.
    """
    import os

    if not os.environ.get("REFLEXIVE_EQUAL_CONTACTS"):
        return CONTACT_WEIGHTS
    w = [1.0] * len(CONTACT_WEIGHTS)
    w[REWARD_CONTACT_LINKS.index("palm")] += float(
        os.environ.get("REFLEXIVE_PALM_CONTACT_BONUS", "0")
    )
    return tuple(w)


def _state(env) -> dict:
    """Lazily-created per-env buffers for the stateful rewards, hung off the env."""
    if not hasattr(env, "_grasp_state"):
        n, dev = env.num_envs, env.device
        env._grasp_state = {
            "contact_best": torch.full((n,), float("inf"), device=dev),
            "hold_streak": torch.zeros(n, dtype=torch.long, device=dev),
            "rest_z": torch.zeros(n, device=dev),
            "rest_ready": torch.zeros(n, dtype=torch.bool, device=dev),
            "rest_count": torch.zeros(n, dtype=torch.long, device=dev),
            "weights": torch.tensor(_reward_contact_weights(), device=dev),
        }
    return env._grasp_state


def reset_grasp_state(env, env_ids: torch.Tensor) -> None:
    """Event term (mode='reset'). Clears every improvement watermark and the lift latch.

    Without this the watermarks carry across episodes and the improvement rewards go silently
    dead after the first episode -- the arm would appear to stop being rewarded for progress.
    """
    s = _state(env)
    s["contact_best"][env_ids] = float("inf")
    s["hold_streak"][env_ids] = 0
    s["rest_ready"][env_ids] = False
    s["rest_count"][env_ids] = 0


def _reward_link_order(env, sensor_name: str) -> torch.Tensor:
    """Permutation from sensor primary order to REWARD_CONTACT_LINKS order.

    The sensor returns primaries in model order (palm first here), while CONTACT_WEIGHTS is
    written in the IsaacLab order (palm last). Applying the weights without this permutation
    would silently weight the thumb's 5/5/10 onto the palm and index links -- wrong, and
    invisible in any shape check.
    """
    key = f"_reward_perm_{sensor_name}"
    if not hasattr(env, key):
        names = list(env.scene[sensor_name].primary_names)
        perm = [names.index(n) for n in REWARD_CONTACT_LINKS]
        setattr(env, key, torch.tensor(perm, device=env.device, dtype=torch.long))
    return getattr(env, key)


def _contact_found(env, sensor_name: str) -> torch.Tensor:
    """(num_envs, num_links) bool of object<->link contact, from the contact sensor."""
    data = env.scene[sensor_name].data
    found = data.found
    assert found is not None, f"sensor {sensor_name!r} does not publish 'found'"
    if found.ndim == 3:  # (envs, primaries, slots)
        found = found.any(dim=-1)
    return found > 0


# Sign of the inside-of-hand gate. The mjlab sensor normal points primary->secondary
# (link->object), so a contact on a link's inside face has normal . inward > 0. Checked
# 2026-10-22: the gate agrees with which side of the link the contact point lies on for 97.6%
# of 2,667 hand-object contacts, against 2.7% for the earlier < 0, and a hand lowered back of
# the palm first was called inside on 99% of palm contacts by that earlier sign.
_INSIDE_ALIGN_MIN = 0.0


def _contact_inside(
    env, sensor_name: str = "hand_object", action_name: str = "grasp"
) -> torch.Tensor:
    """(num_envs, 16) bool: object<->link contact kept only if it loads the link's *inside* face.

    Rejects back-of-hand and knuckle contacts so the reward only credits contacts that could be
    part of a real grasp. Uses the sensor's world-frame contact ``normal`` and each reward
    link's inward axis (from the action term), in REWARD_CONTACT_LINKS order.
    """
    from blindgraspreflex.controllers.batched import quat_rotate

    term = env.action_manager.get_term(action_name)
    perm = _reward_link_order(env, sensor_name)
    data = env.scene[sensor_name].data
    found = _contact_found(env, sensor_name)[:, perm]  # (N, 16) bool
    normal = data.normal
    assert normal is not None, f"sensor {sensor_name!r} must publish 'normal' for the inside gate"
    if normal.ndim == 4:  # (envs, primaries, slots, 3)
        normal = normal[:, :, 0]
    normal = normal[:, perm]  # (N, 16, 3) world, link->object
    quat = env.scene["robot"].data.body_link_quat_w[:, term._reward_body_ids]  # (N, 16, 4)
    inward_w = quat_rotate(quat, term._inward_local.unsqueeze(0).expand_as(quat[..., :3]))
    aligns = (normal * inward_w).sum(dim=-1)
    return found & (aligns > _INSIDE_ALIGN_MIN)


def _min_step(best: torch.Tensor, error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    seeded = torch.isfinite(best)
    reward = torch.zeros_like(error)
    eligible = mask & seeded
    reward[eligible] = (best[eligible] - error[eligible]).clamp(min=0.0)
    improved = mask & ((~seeded) | (error < best))
    best[improved] = error[improved]
    return reward


##
# Observations.
##


_LINK_DIST_CACHE: dict[int, torch.Tensor] = {}


def hand_link_object_distance(env, object_name: str = "object") -> torch.Tensor:
    """Distance from every hand link to the object, one scalar per link -- the "vision" input.

    Replaces ``palm_to_object`` for the end-to-end baseline. Instead of being handed the object's
    POSE (3 numbers in the palm frame, which is a perception oracle), the policy reads a
    proximity-sensor array: ``len(REWARD_CONTACT_LINKS)`` = 16 distances, one per finger phalanx
    plus the palm, in that fixed order. The pattern across 16 links spread over the hand encodes
    the object's bearing implicitly, the way a real proximity/tactile array would, and nothing
    tells the policy where the object is in world coordinates.

    Distance is measured to the object's CENTRE, not its surface. True surface distance needs
    ``mj_geomDistance``, which is a CPU MuJoCo call against a CPU MjData -- unusable here, where
    physics runs batched in mujoco_warp on the GPU and this term is evaluated for every world at
    every control step. Centre distance is the documented fallback; it differs from surface
    distance by the object's extent along the link->centre ray, which is a per-object constant
    offset rather than a signal the policy can be misled by.
    """
    robot = env.scene["robot"]
    key = id(env)
    ids = _LINK_DIST_CACHE.get(key)
    if ids is None:
        # Reuse the project's own resolver rather than a second name lookup: it returns the ids in
        # REWARD_CONTACT_LINKS order, which is the order the contact observation and the contact
        # reward already use, so link i means the same link everywhere.
        ids, _ = reward_contact_frames(robot, env.device)
        _LINK_DIST_CACHE[key] = ids
    link_pos = robot.data.body_link_pos_w[:, ids]  # (N, L, 3)
    obj = env.scene[object_name].data.root_link_pos_w
    # Measure to the BELIEVED object, exactly as palm_to_object does. This term is the policy's
    # only channel to the object, so a perception perturbation that did not reach it would leak
    # ground truth and silently make the pose-error sweep measure nothing: the arm would be misled
    # while the distances stayed honest. Guarded because the term is also built during env
    # construction, before the action term exists.
    term = env.action_manager._terms.get("grasp") if hasattr(env, "action_manager") else None
    bias = getattr(term, "_obj_pos_bias", None)
    if bias is not None:
        obj = obj + bias
    d = torch.linalg.norm(link_pos - obj.unsqueeze(1), dim=-1)  # (N, L)
    # Sensor noise on the measurement itself (REFLEXIVE_LINK_DIST_NOISE = std in metres).
    # Independent per link and RESAMPLED EVERY STEP, which is what a noisy range sensor does --
    # distinct from a standing pose error, where every link would be wrong in a correlated way
    # because the object itself is believed to be somewhere else. Clamped at 0: a negative
    # distance is not a measurement a real sensor could return, and letting one through would
    # test the policy on an input it could never see.
    sigma = float(os.environ.get("REFLEXIVE_LINK_DIST_NOISE", "0") or 0)
    if sigma > 0:
        d = (d + torch.randn_like(d) * sigma).clamp_(min=0.0)
    return d


def contact_binary(env, sensor_name: str = "hand_object") -> torch.Tensor:
    return _contact_found(env, sensor_name).float()


# Fingertip = the distal (*_4) phalanx of each of the five fingers within REWARD_CONTACT_LINKS.
_FINGERTIP_CONTACT_IDX = tuple(i for i, n in enumerate(REWARD_CONTACT_LINKS) if n.endswith("4"))


# Positions of the 5 fingertip (*_4) links within REWARD_CONTACT_LINKS order.
_FINGERTIP_REWARD_IDX = list(_FINGERTIP_CONTACT_IDX)

# Sensor feeding the fingertip observation. Unlike the reward's object-gated ``hand_object``, this
# one has NO object-gate: it reports every contact on the hand links (object, self-collision,
# table), so the fingertip touch signal matches a real pad that fires on any contact.
_FINGERTIP_OBS_SENSOR = "hand_all"


def _fingertip_sense_spheres(env):
    """Cache the per-fingertip sensing-sphere ``(offsets (1,5,3), radii (1,5))`` tensors."""
    cached = getattr(env, "_fingertip_sense_spheres_cache", None)
    if cached is None:
        from blindgraspreflex.assets.rizon4_hand_v2 import (
            FINGERTIP_SENSE_OFFSET,
            FINGERTIP_SENSE_RADIUS,
        )

        off = torch.tensor(
            FINGERTIP_SENSE_OFFSET, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        rad = torch.tensor(
            FINGERTIP_SENSE_RADIUS, dtype=torch.float32, device=env.device
        ).unsqueeze(0)
        cached = (off, rad)
        env._fingertip_sense_spheres_cache = cached
    return cached


def fingertip_sphere_contact_binary(
    env, sensor_name: str = _FINGERTIP_OBS_SENSOR, action_name: str = "grasp"
) -> torch.Tensor:
    """5 fingertip binary touches -- exactly what the hardware fingertip pads report.

    A fingertip reports touch iff ANY contact on it -- with the object, another finger
    (self-collision), or the table, since the sensor has no object-gate -- has its contact point
    inside that fingertip's sensing sphere (centre = fingertip link frame + FINGERTIP_SENSE_OFFSET,
    radius FINGERTIP_SENSE_RADIUS, tuned in the grasp playground). Fingertips only; no palm /
    proximal-phalanx contacts. This is the deployable contact signal the actor observes (and, via
    ``fingertip_contact_binary``, the distillation student).
    """
    from blindgraspreflex.controllers.batched import quat_rotate

    data = env.scene[sensor_name].data
    found, pos = data.found, data.pos
    assert found is not None and pos is not None, (
        f"sensor {sensor_name!r} must publish 'found' and 'pos' for the fingertip sphere gate"
    )
    # Contact fields are flattened per (primary, slot), primary-major -> shape (N, P*num_slots).
    # Reshape back to (N, P, S[, 3]) so the reward-order perm indexes the primary (link) axis.
    n_prim = len(env.scene[sensor_name].primary_names)
    found = found.reshape(found.shape[0], n_prim, -1)  # (N, P, S)
    pos = pos.reshape(pos.shape[0], n_prim, -1, 3)  # (N, P, S, 3)

    perm = _reward_link_order(
        env, sensor_name
    )  # sensor primary order -> REWARD_CONTACT_LINKS order
    ft = _FINGERTIP_REWARD_IDX
    found_ft = found[:, perm][:, ft] > 0  # (N, 5, S)
    pos_ft = pos[:, perm][:, ft]  # (N, 5, S, 3) world-frame contact points

    term = env.action_manager.get_term(action_name)
    ft_body_ids = term._reward_body_ids[ft]  # entity-local body ids of the 5 fingertips
    link_pos = env.scene["robot"].data.body_link_pos_w[:, ft_body_ids]  # (N, 5, 3)
    link_quat = env.scene["robot"].data.body_link_quat_w[:, ft_body_ids]  # (N, 5, 4)

    offsets, radii = _fingertip_sense_spheres(env)  # (1, 5, 3), (1, 5)
    centers = link_pos + quat_rotate(link_quat, offsets.expand_as(link_pos))  # (N, 5, 3)
    dist = (pos_ft - centers.unsqueeze(2)).norm(dim=-1)  # (N, 5, S)
    in_sphere = found_ft & (dist <= radii.unsqueeze(-1))  # (N, 5, S)
    return in_sphere.any(dim=-1).float()  # (N, 5): any contact inside the fingertip's sphere


def fingertip_contact_binary(env, sensor_name: str = _FINGERTIP_OBS_SENSOR) -> torch.Tensor:
    """The 5 fingertip binary touches for the distillation *student* -- identical to the actor's
    ``fingertip_sphere_contact_binary`` (sphere-gated, all-contacts, correctly ordered), so the
    student sees exactly the teacher's contact signal.

    (Previously this returned an un-permuted, un-gated slice of ``contact_binary`` that, because
    the sensor's primary order differs from REWARD_CONTACT_LINKS order, selected the wrong five
    links.)
    """
    return fingertip_sphere_contact_binary(env, sensor_name)


def _grasp_site_id(env, site_name: str = "grasp_site") -> int:
    """Index of the palm site within the robot entity, cached on the env."""
    key = f"_grasp_site_{site_name}"
    if not hasattr(env, key):
        ids, _ = env.scene["robot"].find_sites([site_name])
        setattr(env, key, int(ids[0]))
    return getattr(env, key)


def palm_object_vector(
    env, site_name: str = "grasp_site", object_name: str = "object", action_name: str = "grasp"
):
    robot = env.scene["robot"]
    sid = _grasp_site_id(env, site_name)
    palm_pos = robot.data.site_pos_w[:, sid]
    palm_quat = robot.data.site_quat_w[:, sid]
    obj = env.scene[object_name].data.root_link_pos_w
    # Same per-episode estimate bias the scripted arm reads (GraspActionCfg.object_pos_error), so
    # the end-to-end baseline -- for which this term is the ONLY channel carrying the object
    # estimate -- is perturbed by exactly the quantity, not merely a comparable one. Guarded
    # because this term is also built during env construction, before the action term exists.
    term = env.action_manager._terms.get(action_name) if hasattr(env, "action_manager") else None
    bias = getattr(term, "_obj_pos_bias", None)
    if bias is not None:
        obj = obj + bias
    # ...and the same LATCH, under freeze_object_on_engage. Without this the freeze was partial:
    # the arm servoed to a stale estimate while this term -- the policy's only object channel --
    # kept reading the object's true position, so a "frozen" teacher still watched the object move.
    # Occlusion does not work that way: if the hand is blocking the camera, everything downstream
    # of perception is stale, the policy's own input included.
    #
    # `_obj_pos_frozen` is latched in apply_actions from (true + bias), so it is applied AFTER the
    # bias above rather than added to it. Only fires when the cfg flag is on, so non-freeze runs
    # (training included) are bit-identical to before.
    if term is not None and getattr(term.cfg, "freeze_object_on_engage", False):
        valid = getattr(term, "_obj_pos_frozen_valid", None)
        if valid is not None:
            obj = torch.where(valid.unsqueeze(-1), term._obj_pos_frozen, obj)
    return quat_rotate(quat_conjugate(palm_quat), obj - palm_pos)


def score_honesty_reward(
    env,
    action_name: str = "grasp",
    radius: float = 0.15,
    object_name: str = "object",
    false_high: float = 5.0,
    use_contacts: bool = True,
) -> torch.Tensor:
    """Reward grasp_score for being TRUE: high while the object is in the hand, low when it is not.

    Why this exists. grasp_score is only ever shaped indirectly -- it gates the REACH/LIFT latch, so
    the policy is rewarded for whatever score keeps the arm where the reward is. Nothing ever asks
    it to be *correct*. Measured consequence (2026-08-26): with the object removed from the hand the
    score RISES -- 0.916 -> 0.939 in the Manipulator-Software sim and 0.383 -> 0.581 in mjlab, with
    0/11 envs falling below score_low. A sweep over 20x hand stiffness, 4.5x finger speed and 4x
    slowdown never got the held-vs-gone AUC meaningfully above chance, so this cannot be tuned out
    of the plant: the score is uninformative because nothing ever trained it to be informative.

    ``in_hand`` uses the OBJECT-GATED contact sensor: is any hand link touching *the object*. Two
    weaker labels were tried first and both failed, for reasons worth recording:

      * the ungated ``hand_all`` sensor is useless here -- fingers closed on nothing press against
        each other and read as "touching" (measured: lit 79% while holding, 100% while EMPTY);
      * a palm->object distance test is too permissive. The palm sits only ~0.074 m from the object
        while grasping, so a can knocked loose but still sitting under the hand stays inside a
        0.15 m radius and never trips the penalty. With that label the reward plateaued at a
        healthy-looking 12.6 while the drop response did not improve at all (delta stayed NEGATIVE,
        AUC 0.44-0.47) -- the policy simply earned it by grasping well.

    ``false_high`` makes the term asymmetric. Symmetric shaping is dominated by the common case
    (holding), so its optimum is just "keep the score high" -- exactly the failure being fixed.
    Claiming a grasp you do not have has to cost more than under-claiming one you do.

    Privileged, TRAINING-ONLY: the observation is untouched, so the policy still has to infer the
    drop from proprioception and touch. Supervise the target, never the input.
    """
    term = env.action_manager.get_term(action_name)
    score = torch.sigmoid(term._raw[:, 0])
    if use_contacts:
        in_hand = (contact_binary(env, "hand_object").sum(-1) > 0).float()
    else:
        palm = env.scene["robot"].data.site_pos_w[:, _grasp_site_id(env)]
        obj = env.scene[object_name].data.root_link_pos_w
        in_hand = (torch.linalg.norm(obj - palm, dim=-1) < radius).float()
    return score * in_hand - false_high * score * (1.0 - in_hand)


def arm_joint_pos(env) -> torch.Tensor:
    """The 7 arm joint positions. Only meaningful when the policy commands the arm."""
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    return robot.data.joint_pos[:, ids]


def arm_tracking_error(env, action_name: str = "grasp") -> torch.Tensor:
    """``arm_cmd - q`` for the 7 arm joints -- the arm's analogue of ``hand_tracking_error``.

    With the arm rate-limited by a velocity field, the commanded pose and the measured pose differ
    while the arm is still travelling, and that difference is what the joint PD converts into
    torque. Without it the policy sees where the arm IS but not where it is going.
    """
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    term = env.action_manager.get_term(action_name)
    ref = getattr(term, "_arm_cmd", None)
    if ref is None:  # phase-controlled arm: no commanded joint pose exists
        return torch.zeros_like(robot.data.joint_pos[:, ids])
    return ref - robot.data.joint_pos[:, ids]


def hand_joint_pos(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    return robot.data.joint_pos[:, ids]


def hand_tracking_error(env, action_name: str = "grasp") -> torch.Tensor:
    """The 20-D error term in the policy observation.

    Default is ``hand_target - q``: the policy's COMMANDED pose minus the measured one. That
    superposes two different things --

        e = (hand_target - q_cmd)  +  (q_cmd - q)
              velocity-field lag       servo error, proportional to torque

    -- and only the second relates to grip force. The lag term decays with time constant
    slowdown/v_max (0.44 s in sim, 0.11 s on the deployed low_level), so during any command
    transient it dominates.

    Set REFLEXIVE_TRACKING_QCMD=1 to use ``q_cmd - q`` instead: the actuator's own error, hence a
    direct read on grip torque (tau = kp*(q_cmd - q) - kd*qdot) with no lag term mixed in.

    NOTE this changes the observation's meaning, so it needs its own teacher AND student, and a
    policy trained one way must not be run the other. On the deploy side q_cmd lives in the
    LOW-LEVEL node (``_q_cmd_hand``), not the hand-policy node, so deploying a q_cmd-trained policy
    additionally requires low_level to publish it -- which nothing does today.
    """
    import os

    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    term = env.action_manager.get_term(action_name)
    ref = term._q_cmd if os.environ.get("REFLEXIVE_TRACKING_QCMD") else term.hand_target
    return ref - robot.data.joint_pos[:, ids]


def hand_contact_metric(env, action_name: str = "grasp") -> torch.Tensor:
    """The 20-D online M9 contact metric: |tau| / the joint's own free-running torque.

    Selected over eight alternatives by the invariance study (see contact_metrics.py): across 12
    mjlab plant conditions plus the Manipulator-Software sim its contact value moves 1.41x, against
    19.8x for raw torque and 4.2x for the servo error, while scoring the best mean AUC of the nine.

    Unlike ``hand_tracking_error`` it is dimensionless and carries no stiffness or velocity scale,
    so the same number means the same contact on a soft deployed hand as on the stiff trained one.
    Deploying a policy on this requires the low-level node to publish q_cmd (it owns the velocity
    field and the gains); nothing does today.
    """
    return env.action_manager.get_term(action_name)._m9


def hand_torque(env, action_name: str = "grasp") -> torch.Tensor:
    """The 20-D commanded hand torque, ``kp*(q_cmd - q) - kd*qdot``, clamped to the effort limit.

    This is the same expression Manipulator-Software's sim writes into ``mj_data.ctrl``
    (``SimRobotis5FHandZ.apply_sim_control``) and the same one the XM335 servo runs on hardware,
    with ``qd_des = 0`` and ``tau_ff = 0``. Unlike a position-error observation it is invariant to
    the PD gains: under gain DR, or across the sim-to-real gain gap, the SAME torque means the same
    grip force, whereas the same position error does not.

    Available in both actuator modes -- in explicit mode this IS what gets applied, in the default
    implicit mode it mirrors what the <position> actuator computes internally.

    NOTE deploying a policy on this observation requires the LOW-LEVEL node to publish tau (it owns
    q_cmd, kp and kd, including the live ``_hand_kp_base * v[0]`` scaling); the hand-policy node
    cannot reconstruct it from what it currently subscribes to.
    """
    return env.action_manager.get_term(action_name)._hand_tau


def randomize_hand_pd(
    env, env_ids, kp_range=(0.1, 1.5), kd_range=(0.3, 1.5), action_name: str = "grasp"
) -> None:
    """Per-episode DR of the hand PD gains, for EXPLICIT-PD mode.

    The explicit-mode equivalent of ``mjlab.envs.mdp.dr.pd_gains``: with a <motor> actuator there
    is no kp in the model to randomize, the gains live in the action term. Ranges are SCALE factors
    on the nominal table, matching pd_gains' operation="scale" so the two modes randomize
    comparably. The deployed hand runs 0.083-0.25x nominal, so a range whose lower end sits above
    ~0.1 leaves the real robot outside the training distribution.
    """
    term = env.action_manager.get_term(action_name)
    n = len(env_ids)
    dev = term._hand_kp_env.device
    kp_s = torch.rand(n, 1, device=dev) * (kp_range[1] - kp_range[0]) + kp_range[0]
    kd_s = torch.rand(n, 1, device=dev) * (kd_range[1] - kd_range[0]) + kd_range[0]
    from blindgraspreflex.assets.rizon4_hand import HAND_KD, HAND_KP

    kp0 = torch.tensor(HAND_KP, dtype=torch.float32, device=dev)
    kd0 = torch.tensor(HAND_KD, dtype=torch.float32, device=dev)
    term._hand_kp_env[env_ids] = kp0 * kp_s
    term._hand_kd_env[env_ids] = kd0 * kd_s


def object_pos(env, object_name: str = "object") -> torch.Tensor:
    return env.scene[object_name].data.root_link_pos_w


def object_lin_vel(env, object_name: str = "object") -> torch.Tensor:
    return env.scene[object_name].data.root_link_lin_vel_w


def object_height_above_rest(env, object_name: str = "object") -> torch.Tensor:
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    return (z - s["rest_z"]).unsqueeze(-1)


##
# Rewards.
##


def palm_proximity_reward(
    env, sigma: float = 0.20, site_name: str = "grasp_site", object_name: str = "object"
) -> torch.Tensor:
    """Dense reward for bringing the grasp centre near the object: ``1 - tanh(d / sigma)``.

    Exists because the joint-target baseline has to LEARN the approach that the phase-controlled
    task scripts. Without it the only signal before first contact is ``height``, which pays for the
    object going up however that happens -- so the fastest thing to discover is a swing that
    launches the object, measured at 27 cm of lift with contact on 1 control step in 855.

    ``tanh`` rather than a raw negative distance: it is bounded in [0, 1], so the term cannot
    dominate the episode return no matter how the arm moves.

    ``sigma`` defaults to 0.20 m, not the 0.10 m that looks natural for a grasp: the reward has to
    carry gradient across the range the policy actually operates in, and episodes START at the
    0.25 m pre-grasp standoff. At sigma 0.10 the payout there is 0.013 and at 0.30 m it is 0.005 --
    tanh has saturated, so the term cannot tell "drifting away" from "closing in" exactly where the
    approach has to be learned. At sigma 0.20 the same points pay 0.152 and 0.095, and 5 cm pays
    0.755, which keeps a usable slope over the whole 0 - 0.3 m span.

    Paid densely rather than as an improvement watermark: the policy should be rewarded for STAYING
    near the object, not just for arriving once. Hovering is a real farming route, but a bounded
    per-step payout of at most 1.0 is small against ``height`` (lift x 100) and ``hold`` (150), so
    it shapes the approach without paying more than actually grasping does.
    """
    robot = env.scene["robot"]
    palm = robot.data.site_pos_w[:, _grasp_site_id(env, site_name)]
    obj = env.scene[object_name].data.root_link_pos_w
    d = torch.linalg.norm(obj - palm, dim=-1)
    return 1.0 - torch.tanh(d / max(float(sigma), 1e-6))


def contact_coverage_improvement(
    env, sensor_name: str = "hand_object", action_name: str = "grasp"
) -> torch.Tensor:
    """Improvement watermark on weighted inside-of-hand contact coverage (IsaacLab weights).

    Only contacts loading the *inside* of each hand link count (see :func:`_contact_inside`), so
    the reward credits forming a real grasp, not brushing the object with the back of the hand.
    Pays ``max(best_uncovered - uncovered, 0)`` and lowers the watermark, so it telescopes to
    total coverage gained over the episode and cannot be farmed by making and breaking contact.
    """
    s = _state(env)
    inside = _contact_inside(env, sensor_name, action_name).float()  # (N,16) in reward-link order
    uncovered = ((1.0 - inside) * s["weights"].unsqueeze(0)).sum(dim=1)
    return _min_step(s["contact_best"], uncovered, torch.ones_like(uncovered, dtype=torch.bool))


def object_height_reward(
    env, settle_steps: int = 5, max_lift: float = 0.6, object_name: str = "object"
) -> torch.Tensor:
    """Dense reward for how high the object is lifted above its settled rest height.

    Deliberately height, not distance-to-goal. A goal pose has a horizontal component, so a
    distance reward is gamed by sliding the object along the mat toward it without ever
    lifting. Height cannot be gamed that way -- sliding does not change z.

    Dense (paid every step, not a one-shot or an improvement watermark), so it rewards
    *holding* the object aloft, which favours a stable sustained grasp over a brief upward
    knock. Measured against the settled rest height so the object's initial drop onto the mat
    is not credited.

    Clamped at ``max_lift`` (default 0.6 m, safely above the ~0.46 m lift of a held-at-goal
    object) so an external-disturbance *fling* -- which can hurl a light object metres upward --
    cannot spike this unbounded dense term into the millions and blow up training. Legitimate
    lifting to the goal is never clipped.
    """
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    pending = ~s["rest_ready"]
    s["rest_count"][pending] += 1
    capture = pending & (s["rest_count"] >= settle_steps)
    s["rest_z"][capture] = z[capture]
    s["rest_ready"][capture] = True

    lift = (z - s["rest_z"]).clamp(min=0.0, max=max_lift)
    return lift * s["rest_ready"].float()


def object_goal_proximity_reward(
    env, sigma: float = 0.2, action_name: str = "grasp", object_name: str = "object"
) -> torch.Tensor:
    """Dense Gaussian reward on the object's distance to the FIXED end-effector goal pose.

    Additive shaping between ``object_height_reward`` (which only knows z) and
    ``hold_at_goal_reward`` (which pays a flat bonus once inside 0.15 m and nothing outside it).
    Height alone gives no gradient toward the goal's horizontal position, and the hold bonus gives
    no gradient at all until the object is already there; this fills the gap.

    SLIDE-PROOF, which is why the earlier distance-to-goal reward was removed and replaced by
    height (see object_height_reward's docstring: "a distance reward is gamed by sliding the
    object along the mat toward it without ever lifting"). Two things prevent that here. The goal
    is the ELEVATED retreat pose at z ~ 0.51 while the mat is at z ~ 0.01, so an object sliding on
    the mat can never get closer than ~0.50 m; and the Gaussian kernel at the default sigma pays
    exp(-(0.50/0.2)^2) ~ 0.002 there, i.e. nothing. The reward only becomes worth anything once
    the object is genuinely lifted and carried. The height reward is KEPT alongside it, so the
    lift gradient does not depend on this term at all.
    """
    goal = env.action_manager.get_term(action_name).home_goal_pos.reshape(1, 3)
    obj = env.scene[object_name].data.root_link_pos_w
    d = torch.linalg.norm(obj - goal, dim=-1)
    return torch.exp(-((d / max(float(sigma), 1e-6)) ** 2))


def hold_at_goal_reward(
    env, threshold: float = 0.15, action_name: str = "grasp", object_name: str = "object"
) -> torch.Tensor:
    """Bonus each step the object is held within ``threshold`` of the end-effector goal pose.

    Discourages the drop-and-re-grasp behaviour: under the dense height reward alone the policy
    can farm height by repeatedly lifting and dropping, since the height signal is the same
    whether the object is held or bouncing. This pays only while the object stays *at* the
    goal, so dropping it out of the threshold immediately forfeits the reward.

    Slide-proof, unlike the earlier distance-to-goal reward: the goal pose is the elevated LIFT
    retreat pose (z ~ 0.21 m), so an object on the mat is always >0.2 m away -- reaching the
    threshold requires actually lifting and carrying it there, not sliding along the ground.
    """
    goal = env.action_manager.get_term(action_name).home_goal_pos  # (3,)
    obj = env.scene[object_name].data.root_link_pos_w  # (N, 3)
    dist = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1)
    return (dist < threshold).float()


def _hold_steps(env, hold_time_s: float) -> int:
    return max(1, round(hold_time_s / env.step_dt))


def held_at_goal_success(
    env,
    threshold: float = 0.15,
    hold_time_s: float = 2.0,
    action_name: str = "grasp",
    object_name: str = "object",
) -> torch.Tensor:
    """Termination: object held within ``threshold`` of the goal for ``hold_time_s`` straight.

    Owns the consecutive-hold counter (runs before the reward manager each step), so
    :func:`success_bonus` just reads it. The streak resets to zero the moment the object
    leaves the threshold, so the hold must be continuous, not cumulative.
    """
    s = _state(env)
    goal = env.action_manager.get_term(action_name).home_goal_pos
    obj = env.scene[object_name].data.root_link_pos_w
    in_thr = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1) < threshold
    s["hold_streak"] = torch.where(in_thr, s["hold_streak"] + 1, torch.zeros_like(s["hold_streak"]))
    return s["hold_streak"] >= _hold_steps(env, hold_time_s)


def success_bonus(env, hold_time_s: float = 2.0) -> torch.Tensor:
    """One-shot bonus on the step the hold streak reaches the success duration.

    Reads the streak that :func:`held_at_goal_success` already advanced this step. It fires
    exactly once because the episode terminates on the same step. Sized (via the reward
    weight) to exceed the most hold reward obtainable by *avoiding* success -- otherwise the
    early termination would make the policy hold just short of the threshold on purpose.
    """
    s = _state(env)
    return (s["hold_streak"] >= _hold_steps(env, hold_time_s)).float()


def step_penalty(env) -> torch.Tensor:
    """Constant 1.0 per env each control step, applied with a NEGATIVE weight as a time cost.

    The reward manager scales by dt, so the effective per-step cost is ``|weight| * dt``. It bleeds
    reward while nothing else pays -- notably when the policy sits in a committed LIFT after a
    missed grasp (height and hold both pay 0 with no object aloft) -- so un-committing and
    re-approaching for another attempt within the episode is strictly better than idling.
    """
    return torch.ones(env.num_envs, device=env.device)


##
# Grasp-quality rewards (net contact wrench + grasp matrix).
#
# Both read a *dedicated* contact sensor ``hand_object_grasp`` (separate from the grasp-tuned
# ``hand_object`` sensor so nothing here can perturb the existing contact/coverage rewards). That
# sensor publishes, per hand-link<->object contact, the WORLD-frame contact force, the world
# contact point, and the world normal -- everything needed to (a) sum the net wrench the hand
# applies to the object and (b) assemble the grasp matrix.
##

_GRASP_SENSOR = "hand_object_grasp"


def _model_tensor(env, path: str) -> torch.Tensor:
    """Read a (possibly per-world) mujoco-warp model array as a torch tensor, expanded to B rows.

    The warp bridge exposes model arrays as torch-backed proxies whose leading dim is nworld for
    variant fields (per-env mass/inertia -- this project's objects differ per world) but may be 1
    for a shared field. Indexing materialises a real tensor; we broadcast the size-1 case to B.
    """
    obj = env.sim.model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    t = obj[:]  # TorchArray.__getitem__ -> torch.Tensor (shares memory)
    if t.shape[0] == 1 and env.num_envs > 1:
        t = t.expand(env.num_envs, *t.shape[1:])
    return t


def _object_mass_gravity(env, object_name: str = "object"):
    """Per-env object mass [B], world gravity vector [B,3], and weight magnitude m*|g| [B]."""
    bid = env.scene[object_name].data.indexing.root_body_id
    mass = _model_tensor(env, "body_mass")[:, bid]  # [B]
    g = _model_tensor(env, "opt.gravity")  # [B, 3]
    weight_mag = (mass * torch.linalg.norm(g, dim=-1)).clamp(min=1e-6)  # [B]
    return mass, g, weight_mag


def _char_length(env, object_name: str = "object") -> torch.Tensor:
    """Per-env characteristic object length [B] from the inertia: sqrt(tr(I)/m) ~ a size scale.

    Used to non-dimensionalise moment arms so the net-wrench torque term and the grasp matrix are
    comparable across the 78 YCB objects, which span an order of magnitude in size.
    """
    bid = env.scene[object_name].data.indexing.root_body_id
    inertia = _model_tensor(env, "body_inertia")[:, bid]  # [B, 3] principal diagonal
    mass = _model_tensor(env, "body_mass")[:, bid].clamp(min=1e-6)  # [B]
    return torch.sqrt((inertia.sum(dim=-1) / mass).clamp(min=1e-9))  # [B]


def _grasp_contacts_world(env, sensor_name: str = _GRASP_SENSOR):
    """(found, force, pos, normal) for every hand<->object contact, flattened to [B, M(, 3)] world.

    ``found`` [B, M] bool, ``force`` [B, M, 3] world-frame contact force, ``pos`` [B, M, 3] world
    contact point, ``normal`` [B, M, 3] world unit normal (hand-link -> object). M = 16 links x
    num_slots. Non-contact slots are False in ``found`` and (masked to) zero elsewhere.
    """
    data = env.scene[sensor_name].data
    b = env.num_envs
    found = (data.found > 0).reshape(b, -1)  # [B, M]
    force = torch.nan_to_num(data.force).reshape(b, -1, 3)  # [B, M, 3]
    pos = torch.nan_to_num(data.pos).reshape(b, -1, 3)  # [B, M, 3]
    normal = torch.nan_to_num(data.normal).reshape(b, -1, 3)  # [B, M, 3]
    return found, force, pos, normal


def _grasped_gate(env, found, lift_thresh: float, min_contacts: int, object_name: str = "object"):
    """True where the object is lifted off its rest height AND held by >= min_contacts contacts.

    Gates the grasp-quality rewards so they only credit a genuine in-air hold. Without the lift
    gate an object resting on the mat trivially satisfies "net wrench ~ 0" (the table, not the
    hand, balances gravity), and the grasp matrix would be scored on incidental brushing contacts.
    """
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    lifted = s["rest_ready"] & ((z - s["rest_z"]) > lift_thresh)
    return lifted & (found.sum(dim=1) >= min_contacts)


def _net_wrench(env, object_name: str = "object"):
    """Core of the net-wrench reward, shared with the viewer readout.

    Returns the world-frame net force + torque the hand contacts and gravity apply to the object
    about its COM, plus the weight magnitude and size scale used to non-dimensionalise them:
    ``(net_force[B,3], net_torque[B,3], weight_mag[B], length[B], found[B,M])``.
    """
    found, force, pos, _ = _grasp_contacts_world(env)
    mask = found.unsqueeze(-1).float()  # [B, M, 1]
    force = force * mask
    com = env.scene[object_name].data.root_com_pos_w.unsqueeze(1)  # [B, 1, 3]
    r = (pos - com) * mask  # [B, M, 3] moment arm from COM
    contact_force = force.sum(dim=1)  # [B, 3]
    contact_torque = torch.cross(r, force, dim=-1).sum(dim=1)  # [B, 3]
    mass, g, weight_mag = _object_mass_gravity(env)
    net_force = contact_force + mass.unsqueeze(-1) * g  # +m*g is the (downward) weight
    net_torque = contact_torque  # gravity acts at the COM -> no gravity torque
    return net_force, net_torque, weight_mag, _char_length(env), found


def net_wrench_components(env):
    """Viewer helper: ``(net_force[B,3], net_torque[B,3], force_resid[B], torque_resid[B])`` where
    the residuals are the weight-normalised imbalances the net-wrench reward drives to zero."""
    net_force, net_torque, weight_mag, length, _ = _net_wrench(env)
    force_resid = torch.linalg.norm(net_force, dim=-1) / weight_mag
    torque_resid = torch.linalg.norm(net_torque, dim=-1) / (weight_mag * length)
    return net_force, net_torque, force_resid, torque_resid


def net_wrench_reward(
    env,
    sharpness: float = 3.0,
    torque_weight: float = 0.3,
    lift_thresh: float = 0.03,
    min_contacts: int = 2,
) -> torch.Tensor:
    """Reward a *balanced* in-hand hold: the net wrench the hand + gravity apply to the object -> 0.

    Sums the measured world-frame contact wrench about the object COM and adds the gravity wrench
    (m*g at the COM, zero torque). At a stable static hold this net wrench is zero -- the hand's
    contact forces exactly bear the object's weight with no residual force/torque accelerating it.
    Normalised by the object weight (and a size scale for the torque term) so it is comparable
    across YCB objects, then mapped through exp(-sharpness * residual) to [0, 1].

    Note this is an *equilibrium* signal, not a force-closure one: it says "currently held steady,"
    which is why gravity is included (so it fires only when the HAND bears the weight, not when the
    object rests on the mat) and why it is complementary to :func:`grasp_matrix_quality_reward`,
    which scores robustness to disturbance.
    """
    net_force, net_torque, weight_mag, length, found = _net_wrench(env)
    residual = torch.linalg.norm(net_force, dim=-1) / weight_mag + torque_weight * (
        torch.linalg.norm(net_torque, dim=-1) / (weight_mag * length)
    )
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.exp(-sharpness * residual) * gate


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix of ``v`` [..., 3] -> [..., 3, 3] (so skew(v) @ w = v x w)."""
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zero, -vz, vy], dim=-1)
    row1 = torch.stack([vz, zero, -vx], dim=-1)
    row2 = torch.stack([-vy, vx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _frame_from_normal(n: torch.Tensor) -> torch.Tensor:
    """Orthonormal contact frame [..., 3, 3] with columns (n, t1, t2) from a unit normal n."""
    ex = torch.zeros_like(n)
    ex[..., 0] = 1.0
    ey = torch.zeros_like(n)
    ey[..., 1] = 1.0
    axis = torch.where((n[..., 0].abs() > 0.9).unsqueeze(-1), ey, ex)
    t1 = torch.cross(n, axis, dim=-1)
    t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([n, t1, t2], dim=-1)


def grasp_matrix_metrics(env, min_contacts: int = 3):
    """The three classical grasp-matrix (SVD) quality measures, per env. Each [B].

    Builds the point-contact-with-friction grasp matrix G in R^{6x3n} (Roa & Suarez 2015): each
    contact contributes a 6x3 block [[R],[skew(r/L) R]] mapping a contact-frame force to the object
    wrench about the COM, with moment arms r non-dimensionalised by the object size L. From the
    singular values sigma_1..6 of G it returns:

    * Q_MSV  = sigma_min      -- worst-case: the grasp's ability to resist a wrench in its weakest
      direction (0 if the contacts fail to span the 6-D wrench space).
    * Q_VEW  = prod(sigma_i)  -- volume of the wrench ellipsoid (average-case capability).
    * Q_GII  = sigma_min/sigma_max -- isotropy: how uniformly the grasp resists across directions.

    These are grasp-*matrix* metrics: cheap, batchable, and dominated by how well the contacts are
    distributed around the object. They ignore the friction-cone unilaterality that true
    force-closure (Ferrari-Canny epsilon / FRoGGeR min-weight) enforces -- see the module notes.
    """
    found, _, pos, normal = _grasp_contacts_world(env)
    b, m = found.shape
    mask = found.float()  # [B, M]
    com = env.scene["object"].data.root_com_pos_w.unsqueeze(1)  # [B, 1, 3]
    length = _char_length(env).clamp(min=1e-3).view(b, 1, 1)  # [B, 1, 1]
    r = (pos - com) / length  # [B, M, 3] non-dimensional moment arm
    n = normal / (normal.norm(dim=-1, keepdim=True) + 1e-9)  # [B, M, 3] unit
    rot = _frame_from_normal(n)  # [B, M, 3, 3]
    block = torch.cat([rot, torch.matmul(_skew(r), rot)], dim=2)  # [B, M, 6, 3]
    block = block * mask.view(b, m, 1, 1)  # zero non-contact columns (don't change sigma)
    gram = torch.einsum("bmij,bmkj->bik", block, block)  # G G^T, [B, 6, 6]
    # cusolver's batched eigvalsh intermittently fails to converge on an ill-conditioned or
    # eigenvalue-degenerate grasp matrix (a near-singular contact configuration -- e.g. a few
    # collinear contacts). Condition it with a tiny diagonal jitter (negligible vs the O(0.1-1)
    # eigenvalues that matter), and fall back to the robust CPU/double LAPACK path, then to zeros,
    # so one degenerate env can never crash a whole training run or the cross-eval report.
    gram = torch.nan_to_num(gram) + 1e-4 * torch.eye(6, device=gram.device, dtype=gram.dtype)
    try:
        eig = torch.linalg.eigvalsh(gram)
    except Exception:
        try:
            eig = torch.linalg.eigvalsh(gram.double().cpu()).to(
                device=gram.device, dtype=gram.dtype
            )
        except Exception:
            eig = torch.zeros(b, 6, device=gram.device, dtype=gram.dtype)
    eig = eig.clamp(min=0.0)  # [B, 6] ascending
    sigma = eig.sqrt()
    enough = found.sum(dim=1) >= min_contacts
    q_msv = sigma[:, 0] * enough
    q_vew = sigma.prod(dim=1) * enough
    q_gii = torch.where(
        sigma[:, -1] > 1e-9, sigma[:, 0] / sigma[:, -1], torch.zeros_like(sigma[:, 0])
    )
    q_gii = q_gii * enough
    return q_msv, q_vew, q_gii


def grasp_matrix_quality_reward(
    env, scale: float = 1.2, lift_thresh: float = 0.03, min_contacts: int = 3
) -> torch.Tensor:
    """Reward grasp-matrix quality Q_MSV (min singular value of G), bounded to [0, 1) via tanh.

    Rewards forming a grasp whose contacts are distributed so the hand can exert a wrench in every
    direction -- contacts clustered on one side (which a disturbance could peel off) score near
    zero. Gated on a genuine lifted, multi-contact hold. ``scale`` sets the Q_MSV at which the
    reward reaches tanh(1)~0.76; calibrated from measured ranges on held YCB objects.
    """
    q_msv, _, _ = grasp_matrix_metrics(env, min_contacts=min_contacts)
    found, *_ = _grasp_contacts_world(env)
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.tanh(q_msv / scale) * gate


def _hand_all_sensor_cfg() -> ContactSensorCfg:
    """No-object-gate hand contact sensor for the fingertip OBSERVATION.

    Reports EVERY contact on the hand links -- object, self-collision (finger<->finger), and the
    table -- like a real fingertip pad that fires on any touch, so the deployed contact signal is
    reproduced in sim. Same primary links + order as ``hand_object`` so the reward-order perm
    applies; only ``found`` + ``pos`` are needed (the fingertip sphere gate uses the world contact
    point). Kept separate from ``hand_object`` so the object-gated coverage reward is unchanged.
    """
    return ContactSensorCfg(
        name="hand_all",
        primary=ContactMatch(
            mode="body", pattern=f"^({'|'.join(REWARD_CONTACT_LINKS)})$", entity="robot"
        ),
        secondary=None,  # any contact with a primary counts -> object + self-collision + table
        fields=("found", "pos"),
        reduce="none",
        # Keep enough contacts per fingertip that an in-sphere touch is never dropped when the
        # link is busy (object + several finger/table contacts at once); a pad realistically has
        # far fewer than this many simultaneous contacts.
        num_slots=8,
    )


def reward_contact_frames(entity, device):
    """``(body_ids, inward_local)`` for :data:`REWARD_CONTACT_LINKS`, in that exact order.

    The inside-of-hand contact gate keeps only contacts whose normal points into the link's
    inward face: local +X for every reward link, the palm face and the finger pads. The fingers
    flex toward palm +X (the middle fingertip moves from x = 0.015 to 0.072 m when it bends), and
    the palm is 4.8 cm thick along X; its +Z runs from the wrist to the knuckles, which an
    earlier version of this table used for the palm. Both tensors are static geometry, not
    controller state,
    which is why any action term can supply them: :func:`_contact_inside` reads them off whatever
    term it is pointed at, so an alternative controller only has to expose these two attributes to
    reuse the contact reward.
    """
    ids, _ = entity.find_bodies(list(REWARD_CONTACT_LINKS), preserve_order=True)
    return (
        torch.tensor(ids, device=device, dtype=torch.long),
        torch.tensor([[1.0, 0.0, 0.0]] * len(REWARD_CONTACT_LINKS), device=device),
    )


def hand_object_sensor_cfg() -> ContactSensorCfg:
    """The ``hand_object`` sensor that :func:`contact_coverage_improvement` reads.

    Factored out of :func:`make_task` so another task can reuse the contact reward verbatim
    instead of re-deriving the sensor: the reward is only meaningful against a sensor with
    exactly this primary set and ordering, since ``_contact_inside`` indexes its rows by
    ``REWARD_CONTACT_LINKS`` position.
    """
    return ContactSensorCfg(
        name="hand_object",
        # Patterns are regexes matched against names *within* the entity, so they are
        # unprefixed. The parentheses matter: "^a|b|c$" parses as "^a" OR "b" OR "c$".
        primary=ContactMatch(
            mode="body", pattern=f"^({'|'.join(REWARD_CONTACT_LINKS)})$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="^object$", entity="object"),
        # "normal" (world frame, primary->secondary) feeds the inside-of-hand contact gate.
        # Object-gated: this sensor drives the object-coverage reward, so it must see the object
        # only. The fingertip *observation* uses the separate no-gate ``hand_all`` sensor instead.
        fields=("found", "force", "normal"),
        reduce="none",
        num_slots=1,
    )


def _grasp_metric_sensor_cfg() -> ContactSensorCfg:
    """Dedicated world-frame contact sensor for the grasp-quality rewards (separate from the
    grasp-tuned ``hand_object`` sensor). Publishes world force + contact point + normal per
    hand-link<->object contact; ``global_frame`` needs ``normal`` and ``tangent`` in fields."""
    return ContactSensorCfg(
        name=_GRASP_SENSOR,
        primary=ContactMatch(
            mode="body", pattern=f"^({'|'.join(REWARD_CONTACT_LINKS)})$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="^object$", entity="object"),
        fields=("found", "force", "normal", "tangent", "pos"),
        global_frame=True,
        reduce="none",
        num_slots=3,
    )


# The grasp-quality reward terms, keyed by the --grasp-reward name train_grasp.py exposes. Not in
# the default reward stack (the reward manager skips weight-0 terms, so they must be injected with
# a real weight); set_grasp_metric_reward puts the chosen one in with the given weight.
_GRASP_REWARD_FUNCS = {
    "wrench": net_wrench_reward,
    "matrix": grasp_matrix_quality_reward,
}


def set_grasp_metric_reward(env_cfg, method: str, weight: float) -> None:
    """Inject one grasp-quality reward term into an env cfg (used by train_grasp.py --grasp-reward).

    Adds it *on top of* the existing contact/height/hold/success stack: these metrics only produce
    signal once the hand already holds the object, so they refine grasp quality rather than replace
    the shaping that teaches grasping in the first place.
    """
    if method == "none":
        return
    func = _GRASP_REWARD_FUNCS[method]
    env_cfg.rewards[f"grasp_{method}"] = RewardTermCfg(func=func, params={}, weight=weight)


def success_held_to_end(
    env, threshold: float = 0.15, action_name: str = "grasp", object_name: str = "object"
) -> torch.Tensor:
    """One-shot success bonus on the episode's final step if the object is still held at the goal.

    The alternative to :func:`success_bonus` for a *hold-as-long-as-possible* task: instead of
    firing (and terminating) after a fixed 2 s hold -- which teaches the policy it can cash out
    early and then slip -- this pays only when the object is STILL within ``threshold`` of the goal
    at the moment the episode times out. Combined with dropping the early ``success`` termination
    and the dense per-step hold reward, the policy is rewarded for holding all the way to the end.

    Fires exactly once: ``episode_length_buf`` reaches ``max_episode_length`` only on the final
    step (it is incremented before rewards are computed), and the env resets immediately after.
    """
    goal = env.action_manager.get_term(action_name).home_goal_pos
    obj = env.scene[object_name].data.root_link_pos_w
    held = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1) < threshold
    at_end = env.episode_length_buf >= env.max_episode_length
    return (held & at_end).float()


def _hand_pca(env):
    """Lazily-loaded, env-cached HandPCA (the fitted 5-DoF human-grasp manifold)."""
    if not hasattr(env, "_hand_pca_cache"):
        from blindgraspreflex.assets.hand_pca import load_hand_pca

        env._hand_pca_cache = load_hand_pca(device=env.device)
    return env._hand_pca_cache


def hand_pca_manifold_reward(
    env, sharpness: float = 1.0, lift_thresh: float = 0.03, min_contacts: int = 3
) -> torch.Tensor:
    """Reward the hand joint pose lying ON the fitted PCA grasp manifold (natural human-grasp shapes).

    ``||q - reconstruct(q)||`` is the distance of the 20-joint pose off the PCA manifold (reconstruct
    = decode(encode(q)) projects onto it); ``exp(-sharpness * dist)`` is 1 on the manifold and decays
    off it. Gated on a real held grasp so it shapes the *grasp* pose, not the open reach pose. Used to
    fine-tune a joint-space policy toward PCA-like hand shapes without changing its action space.
    """
    pca = _hand_pca(env)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    q = robot.data.joint_pos[:, ids]
    dist = torch.linalg.norm(q - pca.reconstruct(q), dim=-1)
    found, *_ = _grasp_contacts_world(env)
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.exp(-sharpness * dist) * gate


##
# Task assembly.
##


def _hull_scaled(base_at_16: int, hulls: int) -> int:
    """Contact/constraint allocation for ``hulls`` hulls, from the value tuned at 16."""
    return int(base_at_16 * max(1.0, hulls / 16.0))


def make_task(
    objects: str | Sequence[str] = "v013_apple",
    num_envs: int = 4096,
    squash: str = "sigmoid",
    source: str = "ycb",
    max_collision_hulls: int | None = None,
) -> ManagerBasedRlEnvCfg:
    if max_collision_hulls is None:
        # Read at REGISTRATION time, like REFLEXIVE_OBJECT_SCALES, because the variant mesh table
        # is built when the task is registered and cannot be changed afterwards. 16 is the value
        # every number before this env var existed was produced with; raising it gives the solver
        # geometry that actually resembles the object (at 16 the mug loses its upper half and the
        # bowl becomes a ring of disconnected fragments -- see benchmark/README.md), at the cost of
        # geom count, which is what limits the variant scene.
        max_collision_hulls = int(os.environ.get("REFLEXIVE_MAX_HULLS", "16"))
    robot = _grasp_robot_cfg()
    # A single name spawns that object in every world; a list spawns a *per-world mesh variant*
    # (a different object per world, evenly allocated) so one policy trains across the set. The
    # object stays one free body named "object" either way, so nothing downstream changes.
    # `source` selects the prepared asset tree the names index into: "ycb" (v0xx_* names) or
    # "objaverse" (GraspXL UID names); both expose the identical single/variant entity configs.
    if source == "objaverse":
        from blindgraspreflex.assets.objaverse import (
            get_objaverse_object_cfg as single_cfg,
        )
        from blindgraspreflex.assets.objaverse import (
            get_objaverse_variant_object_cfg as variant_cfg,
        )
    elif source == "primitives":
        from blindgraspreflex.assets.primitives import (
            get_primitive_object_cfg as single_cfg,
        )
        from blindgraspreflex.assets.primitives import (
            get_primitive_variant_object_cfg as variant_cfg,
        )
    elif source == "gso":
        from blindgraspreflex.assets.gso import (
            get_gso_object_cfg as single_cfg,
        )
        from blindgraspreflex.assets.gso import (
            get_gso_variant_object_cfg as variant_cfg,
        )
    elif source == "graspxl":
        from blindgraspreflex.assets.graspxl import (
            get_graspxl_object_cfg as single_cfg,
        )
        from blindgraspreflex.assets.graspxl import (
            get_graspxl_variant_object_cfg as variant_cfg,
        )
    elif source == "ycb":
        single_cfg, variant_cfg = get_ycb_object_cfg, get_ycb_variant_object_cfg
    else:
        raise ValueError(
            f"Unknown object source {source!r} "
            "(expected 'ycb', 'objaverse', 'graspxl', 'primitives' or 'gso')."
        )
    names = (objects,) if isinstance(objects, str) else tuple(objects)
    if len(names) == 1:
        obj = single_cfg(
            names[0], pos=(*SPAWN_XY, SPAWN_Z), max_collision_hulls=max_collision_hulls
        )
    else:
        obj = variant_cfg(names, pos=(*SPAWN_XY, SPAWN_Z), max_collision_hulls=max_collision_hulls)

    def add_ground(spec):
        import mujoco

        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(3.0, 3.0, 0.1),
            pos=(0.0, 0.0, FLOOR_HEIGHT),
            condim=6,
            friction=(1.0, 5e-3, 1e-4),
        )
        spec.worldbody.add_geom(
            name="mat",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(*MAT_HALF_EXTENT, MAT_THICKNESS * 0.5),
            pos=(*MAT_CENTRE_XY, FLOOR_HEIGHT + MAT_THICKNESS * 0.5),
            condim=6,
            friction=(1.2, 5e-3, 1e-4),
            solref=(0.01, 1.0),
        )
        # The active arm mount's bolt plate / ceiling slab / wall panel. Visual only
        # (contype=0), so the scene's physics is bit-for-bit the same on every mount and a
        # difference in the numbers can only come from the base transform. No-op on the
        # default floor mount beyond a small plate under the base.
        from blindgraspreflex.assets import mounts

        mounts.add_fixture(spec)
        # Figure rendering only (REFLEXIVE_STUDIO_RENDER): a white skybox, key/fill/rim lights and
        # matte floor materials. A texture or a light cannot be added after compilation, so it has
        # to happen here; everything it adds is visual, and it is off unless the var is set.
        if mounts.studio_enabled():
            mounts.studio_scene(spec)

    contact = hand_object_sensor_cfg()

    scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=2.5,
        entities={"robot": robot, "object": obj},
        # Two contact sensors over the same hand<->object pairs: ``hand_object`` (grasp-tuned:
        # found/normal for the coverage reward + contact obs) and ``hand_object_grasp`` (world
        # force/point/normal for the net-wrench + grasp-matrix quality rewards). Kept separate so
        # the grasp-quality additions cannot perturb the existing, tuned contact pipeline.
        sensors=(contact, _grasp_metric_sensor_cfg(), _hand_all_sensor_cfg()),
        spec_fn=add_ground,
    )

    actor_terms = {
        # Hardware-realistic touch: only the 5 fingertips, and only when the contact point is
        # inside the fingertip's sensing sphere (see fingertip_sphere_contact_binary). This is
        # what the deployed hand actually senses, so the policy never learns to rely on palm /
        # proximal-phalanx contacts it won't have on hardware.
        "contacts": ObservationTermCfg(func=fingertip_sphere_contact_binary, params={}),
        # REFLEXIVE_LINK_DISTANCE_OBS=1 swaps the object-pose oracle for the per-link proximity
        # array. Done here, at task construction, so TRAINING and EVERY eval path pick it up from
        # the one switch -- plumbing it through each build_* function is how obs/flag mismatches
        # get introduced, and this project has lost several runs to exactly that.
        **(
            {"link_object_distance": ObservationTermCfg(func=hand_link_object_distance, params={})}
            if os.environ.get("REFLEXIVE_LINK_DISTANCE_OBS")
            else {"palm_to_object": ObservationTermCfg(func=palm_object_vector, params={})}
        ),
        "hand_joint_pos": ObservationTermCfg(func=hand_joint_pos, params={}),
        "hand_tracking": ObservationTermCfg(func=hand_tracking_error, params={}),
    }
    critic_terms = dict(actor_terms)
    # The critic is privileged and training-only, so it keeps the full 16-link contact vector for
    # a richer value signal (not deployed, so no hardware-sensing constraint).
    critic_terms["contacts"] = ObservationTermCfg(func=contact_binary, params={})
    critic_terms.update(
        {
            "object_pos": ObservationTermCfg(func=object_pos, params={}),
            "object_lin_vel": ObservationTermCfg(func=object_lin_vel, params={}),
            "object_height": ObservationTermCfg(func=object_height_above_rest, params={}),
        }
    )

    return ManagerBasedRlEnvCfg(
        decimation=20,
        episode_length_s=5.0,
        scene=scene,
        # Defaults overflow: 16 object hulls against 21 hand links plus the mat generates far
        # more contacts and constraint rows than the bundled tasks, and an nefc overflow
        # corrupts the solve rather than erroring.
        #
        # Both scale with the HULL BUDGET, because the contact count does: 400/1200 was sized for
        # 16 hulls, and at 64 the fixed value does not merely degrade -- mjwarp walks off the end
        # of the contact buffer and the process dies with "CUDA error: an illegal memory access".
        # Deriving them from the budget means raising REFLEXIVE_MAX_HULLS cannot silently corrupt
        # the solve, which is the failure this comment was already warning about.
        sim=SimulationCfg(
            nconmax=_hull_scaled(400, max_collision_hulls),
            njmax=_hull_scaled(1200, max_collision_hulls),
        ),
        actions={"grasp": GraspActionCfg(squash=squash)},
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True),
        },
        rewards={
            # Inside-of-hand contact coverage (IsaacLab weights: thumb tip 10, thumb mids 5,
            # palm 2, rest 1), weight 0.1 as in IsaacLab. Dense shaping toward wrapping fingers
            # onto the object -- object-agnostic, so it is what lets one policy learn to grasp a
            # diverse set rather than memorising a single shape's lift.
            "contact": RewardTermCfg(func=contact_coverage_improvement, params={}, weight=0.1),
            "height": RewardTermCfg(func=object_height_reward, params={}, weight=100.0),
            # Hold-at-goal bonus: rewards keeping the object at the elevated retreat pose,
            # which penalises drop-and-re-grasp (dropping leaves the threshold) and cannot be
            # slid for on the ground. weight 50 -> ~2/step held; height still gives the lift
            # gradient before the object first reaches the goal.
            "hold": RewardTermCfg(
                func=hold_at_goal_reward, params={"threshold": 0.15}, weight=50.0
            ),
            # Off unless --goal-proximity-weight is given, so every existing run is unaffected.
            "goal_proximity": RewardTermCfg(
                func=object_goal_proximity_reward, params={"sigma": 0.2}, weight=0.0
            ),
            "success": RewardTermCfg(
                func=success_bonus, params={"hold_time_s": 2.0}, weight=5000.0
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
            # Real termination (time_out=False -> GAE cuts the value bootstrap), so the policy
            # learns holding the object at the goal for 2 s is the terminal goal, not a state
            # to prolong.
            "success": TerminationTermCfg(
                func=held_at_goal_success,
                params={"threshold": 0.15, "hold_time_s": 2.0},
                time_out=False,
            ),
            # Safety reset for the intermittent contact/solver blow-ups: a world whose object has
            # gone non-finite or been flung fast/far is reset in place. time_out=True so GAE
            # treats it as a truncation (a sim glitch, not a policy failure to penalise). Because
            # the env recomputes observations *after* the reset, catching the divergence here --
            # ideally while the state is still finite -- keeps the NaN out of the returned obs.
            "instability": TerminationTermCfg(func=_unstable, time_out=True),
        },
        events={
            # Required for a fixed-base robot: without it every env's robot sits at the
            # world origin instead of its own env origin.
            "reset_robot_base": EventTermCfg(
                func=_reset_robot_base,
                mode="reset",
                params={},
            ),
            # Overriding `events` drops mjlab's default reset_scene_to_default, so without
            # this the arm resets to all-zero joints (pointing straight up, grasp_site at
            # z=1.25) instead of ARM_Q_HOME.
            "reset_robot_joints": EventTermCfg(func=_reset_robot_joints, mode="reset", params={}),
            "reset_object": EventTermCfg(
                func=_reset_object,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)}
                },
            ),
            "reset_state": EventTermCfg(func=reset_grasp_state, mode="reset", params={}),
        },
    )


def _time_out(env) -> torch.Tensor:
    from mjlab.envs.mdp.terminations import time_out

    return time_out(env)


def _unstable(env, max_speed: float = 15.0, max_dist: float = 2.0) -> torch.Tensor:
    """Fire on a diverged world so it gets reset before its blow-up poisons training.

    A contact/solver instability sends the object non-finite or flings it fast (``max_speed``
    m/s) or far (``max_dist`` m from its env origin). Caught here -- usually while the state is
    still finite -- the world resets and its observation is recomputed clean; the safety wrapper
    in ``train_grasp.py`` sanitises the one-step case where it goes straight to NaN.
    """
    data = env.scene["object"].data
    pos = data.root_link_pos_w
    vel = data.root_link_lin_vel_w
    bad = ~torch.isfinite(pos).all(dim=-1) | ~torch.isfinite(vel).all(dim=-1)
    bad = bad | (torch.linalg.norm(vel, dim=-1) > max_speed)
    bad = bad | (torch.linalg.norm(pos, dim=-1) > max_dist)
    return bad


def _reset_robot_base(env, env_ids) -> None:
    from mjlab.envs.mdp.events import reset_root_state_uniform

    reset_root_state_uniform(
        env, env_ids, pose_range={}, velocity_range={}, asset_cfg=SceneEntityCfg("robot")
    )


def _reset_robot_joints(env, env_ids) -> None:
    from mjlab.envs.mdp.events import reset_joints_by_offset

    reset_joints_by_offset(
        env,
        env_ids,
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
        asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    )


def _reset_object(env, env_ids, pose_range) -> None:
    from mjlab.envs.mdp.events import reset_root_state_uniform

    reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range={},
        asset_cfg=SceneEntityCfg("object"),
    )


def randomize_hand_vf(
    env, env_ids, v_range=(0.6, 2.0), slowdown_range=(0.15, 0.45), action_name: str = "grasp"
) -> None:
    """Per-episode DR of the hand VELOCITY FIELD (v_max, slowdown distance).

    These are controller parameters, not plant parameters, and they differ between sim and the
    robot: training runs v_max 0.8 / slowdown 0.35 while the deployed low_level runs 1.8 / 0.2.
    The ratio slowdown/v_max is the finger-closing time constant -- 0.44 s in sim vs 0.11 s on
    deploy, a 4x difference in how fast the commanded pose converges, which directly reshapes the
    hand_tracking observation the policy reads. Randomising it stops the policy depending on one
    calibration. Defaults deliberately span both the sim and deploy values.
    """
    term = env.action_manager.get_term(action_name)
    n = int(env_ids.numel()) if torch.is_tensor(env_ids) else len(env_ids)
    dev = term._v_max_env.device
    term._v_max_env[env_ids] = torch.rand(n, device=dev) * (v_range[1] - v_range[0]) + v_range[0]
    term._slowdown_env[env_ids] = (
        torch.rand(n, device=dev) * (slowdown_range[1] - slowdown_range[0]) + slowdown_range[0]
    ).clamp(min=1e-6)


def object_vanish(
    env,
    env_ids=None,  # None in mode="interval" (all envs)
    z_thresh: float = 0.15,
    prob: float = 0.02,
    away: float = 5.0,
    object_name: str = "object",
) -> None:
    """Randomly REMOVE a lifted object from the hand, mid-hold, and let the episode continue.

    This exists because two earlier attempts to make grasp_score notice a drop both failed, and the
    post-mortem pointed here. Adding disturbances plus a reward for the score being TRUE produced a
    perfectly healthy training curve (score_honesty ~+16) while the drop response did not move at
    all -- the score still ROSE when the object was removed (delta -0.06..-0.20, AUC 0.39-0.46
    across 8 checkpoints).

    The reason: "not holding" during training almost always means "has not grasped YET" (the REACH
    approach), where the score is naturally low. The honesty reward is fully satisfied by that
    approach-vs-hold asymmetry, so the policy learns score ~ f(phase) and is never forced to learn
    the transition that actually matters -- high score -> low score when the object leaves mid-hold.
    That state is rare even with velocity kicks, because the object usually stays in contact or is
    re-grasped immediately.

    So: make it common. Once the object is above ``z_thresh`` (genuinely lifted), with per-step
    probability ``prob``, teleport it ``away`` metres and leave it gone. The hand is then closed on
    nothing while the episode keeps running, which is exactly the state the drop test evaluates --
    and, paired with score_honesty, the state where a high score is punished hardest.
    """
    obj = env.scene[object_name]
    pos = obj.data.root_link_pos_w
    quat = obj.data.root_link_quat_w
    lifted = pos[:, 2] > z_thresh
    # only fire on envs that are holding it up and have not already been emptied
    fire = lifted & (torch.rand(pos.shape[0], device=pos.device) < prob)
    if not bool(fire.any()):
        return
    new_pos = pos.clone()
    new_pos[fire, 0] += away
    obj.write_root_link_pose_to_sim(torch.cat([new_pos, quat], dim=-1))
    # root velocity is 6-D (lin + ang); zero it so the removed object does not keep flying
    obj.write_root_link_velocity_to_sim(
        torch.zeros((pos.shape[0], 6), dtype=pos.dtype, device=pos.device)
    )


def object_high_wrench(
    env,
    env_ids=None,  # None in mode="step" (all envs)
    z_thresh: float = 0.2,
    max_force: float = 2.5,
    max_torque: float = 0.1,
    offset_range: float = 0.05,
    ref_mass: float = 0.15,
    object_name: str = "object",
) -> None:
    """Once the object is lifted above ``z_thresh`` (world z), apply a random 6-D wrench at a
    random point ON the object, to test the grasp's robustness to external disturbances.

    Per control step (mode="step"): for each env with the object above the threshold, sample a
    random force (magnitude <= ``max_force`` N *at* ``ref_mass``) applied at a random point within
    ``offset_range`` of the object COM (the offset induces a torque, ``offset_w x force``) plus a
    small independent random torque -- a full 6-D wrench written to ``xfrc_applied`` (world frame).
    Envs below the threshold are written zeros so stale forces clear (xfrc_applied persists).

    Mass-proportional: the wrench is scaled by ``mass / ref_mass`` (per-world, DR-aware) so it
    imparts a *bounded acceleration* (~``max_force / ref_mass``) regardless of the object's
    randomized mass and scale. A fixed-force wrench instead flings light DR'd objects (a 2.5 N
    push on a 30 g object is ~80 m/s^2), which is what previously spiked the height reward and
    blew up training. ``ref_mass`` (default 0.15 kg, mid of the 30-300 g mass-DR range) is the
    mass at which the requested ``max_force`` is delivered verbatim.
    """
    from mjlab.utils.lab_api.math import quat_apply, sample_uniform

    obj = env.scene[object_name]
    dev = env.device
    n = env.num_envs
    active = (obj.data.root_link_pos_w[:, 2] > z_thresh).float().unsqueeze(-1)  # (N, 1)

    # Per-world object mass (reflects mass/scale DR), scaled so the disturbance is a bounded
    # acceleration rather than a fixed force.
    mass = obj.data.model.body_mass[:, obj.data.indexing.root_body_id].to(dev)  # (N,)
    mass_scale = (mass / float(ref_mass)).clamp(min=0.0).unsqueeze(-1)  # (N, 1)

    # Random force, magnitude clamped to max_force (at ref_mass), then scaled by mass.
    force = sample_uniform(-max_force, max_force, (n, 3), dev)
    mag = torch.linalg.norm(force, dim=-1, keepdim=True).clamp(min=1e-6)
    force = force * (mag.clamp(max=max_force) / mag) * mass_scale
    # Random point on the object -> torque = offset_w x force (scales with force), plus a small
    # independent random torque (also mass-scaled to bound the angular disturbance similarly).
    offset_w = quat_apply(
        obj.data.body_com_quat_w[:, 0], sample_uniform(-offset_range, offset_range, (n, 3), dev)
    )
    torque = sample_uniform(-max_torque, max_torque, (n, 3), dev) * mass_scale + torch.cross(
        offset_w, force, dim=-1
    )

    obj.write_external_wrench_to_sim(
        (force * active).unsqueeze(1), (torque * active).unsqueeze(1), env_ids=None, body_ids=None
    )


##
# RL config + registration.
##


def grasp_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Stability fixes after run 3 peaked at iter ~200 then decayed. Diagnosis: the
            # entropy bonus inflated the policy std from 0.33 to 0.84 over training, which
            # made lifting unreliable and grew return variance (value loss 24 -> 327). These
            # target that directly:
            #   entropy_coef 0.01 -> 0.002   stop the std from inflating
            #   desired_kl   0.01 -> 0.008   gentler per-update policy drift
            #   normalize_advantage_per_mini_batch  decouple updates from reward scale
            entropy_coef=0.004,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=True,
        ),
        experiment_name="reflexive_grasp",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=2000,
    )


def register() -> None:
    from mjlab.tasks.registry import register_mjlab_task

    from blindgraspreflex.assets.ycb import EASY_YCB_OBJECT_LIST, YCB_OBJECT_LIST

    register_mjlab_task(
        task_id="Reflexive-Grasp-Apple",
        env_cfg=make_task("v013_apple", num_envs=4096),
        play_env_cfg=make_task("v013_apple", num_envs=16),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Diverse-object task: one policy over the EASY YCB set via per-world mesh variants (a
    # different object per world, evenly allocated). play uses one env per object so the
    # viewer shows the whole set at a glance.
    register_mjlab_task(
        task_id="Reflexive-Grasp-YCB",
        env_cfg=make_task(EASY_YCB_OBJECT_LIST, num_envs=4096),
        play_env_cfg=make_task(EASY_YCB_OBJECT_LIST, num_envs=len(EASY_YCB_OBJECT_LIST)),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Full YCB set (all 78 objects, including thin/degenerate ones a 5-finger hand can't grasp).
    # Same per-world-variant scaling: the compiled model is padded to the largest object, so
    # the scene is the same size as one object regardless of catalogue size.
    register_mjlab_task(
        task_id="Reflexive-Grasp-YCB-All",
        env_cfg=make_task(YCB_OBJECT_LIST, num_envs=4096),
        play_env_cfg=make_task(YCB_OBJECT_LIST, num_envs=len(YCB_OBJECT_LIST)),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Objaverse task: one policy over the GraspXL example objects via the same per-world mesh
    # variants. Registered only when the prepared tree exists (scripts/fetch_objaverse.py), so a
    # missing dataset leaves the YCB tasks untouched rather than failing registration.
    from blindgraspreflex.assets.objaverse import list_objaverse_objects

    objaverse_objects = list_objaverse_objects()
    if objaverse_objects:
        register_mjlab_task(
            task_id="Reflexive-Grasp-Objaverse",
            env_cfg=make_task(objaverse_objects, num_envs=4096, source="objaverse"),
            play_env_cfg=make_task(
                objaverse_objects, num_envs=len(objaverse_objects), source="objaverse"
            ),
            rl_cfg=grasp_ppo_cfg(),
        )

    # Procedural primitives: superquadric shapes (box <-> cylinder <-> ellipsoid <-> sphere) via
    # per-world variants. Fully generated, so always registered -- no dataset needed.
    from blindgraspreflex.assets.primitives import PRIMITIVE_OBJECT_LIST

    register_mjlab_task(
        task_id="Reflexive-Grasp-Primitives",
        env_cfg=make_task(PRIMITIVE_OBJECT_LIST, num_envs=4096, source="primitives"),
        play_env_cfg=make_task(
            PRIMITIVE_OBJECT_LIST, num_envs=len(PRIMITIVE_OBJECT_LIST), source="primitives"
        ),
        rl_cfg=grasp_ppo_cfg(),
    )

    # GSO task: a curated, size-normalised graspable subset of Google Scanned Objects via per-world
    # variants. Registered only when the prepared tree exists (scripts/fetch_gso.py).
    from blindgraspreflex.assets.gso import list_gso_objects

    gso_objects = list_gso_objects()
    if gso_objects:
        register_mjlab_task(
            task_id="Reflexive-Grasp-GSO",
            env_cfg=make_task(gso_objects, num_envs=4096, source="gso"),
            play_env_cfg=make_task(gso_objects, num_envs=len(gso_objects), source="gso"),
            rl_cfg=grasp_ppo_cfg(),
        )

    # GraspXL task: a sample of the Objaverse set RobustDexGrasp (arXiv 2504.05287) evaluates on,
    # already size-normalised into small/medium/large buckets upstream. Registered only when the
    # prepared tree exists (scripts/fetch_graspxl.py). The per-bucket task ids let eval reproduce
    # that paper's size-stratified table without re-fetching anything.
    from blindgraspreflex.assets.graspxl import list_graspxl_objects

    graspxl_objects = list_graspxl_objects()
    if graspxl_objects:
        register_mjlab_task(
            task_id="Reflexive-Grasp-GraspXL",
            env_cfg=make_task(graspxl_objects, num_envs=4096, source="graspxl"),
            play_env_cfg=make_task(
                graspxl_objects, num_envs=len(graspxl_objects), source="graspxl"
            ),
            rl_cfg=grasp_ppo_cfg(),
        )
        for _bucket in ("small", "medium", "large"):
            _objs = list_graspxl_objects(bucket=_bucket)
            if _objs:
                register_mjlab_task(
                    task_id=f"Reflexive-Grasp-GraspXL-{_bucket.capitalize()}",
                    env_cfg=make_task(_objs, num_envs=4096, source="graspxl"),
                    play_env_cfg=make_task(_objs, num_envs=len(_objs), source="graspxl"),
                    rl_cfg=grasp_ppo_cfg(),
                )

    from blindgraspreflex.tasks.grasping.clutter_task import register as register_clutter

    register_clutter()


##
# Pre-grasp approach initialisation (RobustDexGrasp-style), for the joint-target baseline.
##


def reset_pregrasp_approach(
    env,
    env_ids,
    standoff: float = 0.25,
    hand_open_frac: float = 0.1,
    action_name: str = "grasp",
    object_name: str = "object",
    approach_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
    init_vel: tuple[float, float, float] | None = None,
    aim_axis: str = "x",
    orientation: str = "home",
) -> None:
    """Start each episode with the palm a fixed standoff from the object, aimed at it.

    Ported from RobustDexGrasp's initialisation, which is the one part of that method worth
    keeping here: instead of beginning at a fixed home pose and having to discover the whole
    reach, the arm begins ``standoff`` metres from the object with the palm's approach axis
    pointing at it, and the fingers partially opened. The remaining problem is the last 25 cm and
    the grasp itself, which is what the baseline is supposed to be learning.

    This matters much more for the joint-target baseline than for the phase-controlled task: there
    the arm's approach is scripted, so the policy never had to learn it. Here it does, and without
    this the policy spends most of training discovering how to get near the object at all.

    The pose is chosen exactly as RDG chooses it -- candidate palm rolls spun about the approach
    axis, scored by the object's width along the closing axis, IK-solved and the narrowest feasible
    grasp taken -- reusing :mod:`rdg_ik` rather than reimplementing it.

    ``hand_open_frac`` interpolates from the open pre-grasp pose (0.0) toward the closed pose
    (1.0), so the fingers start partially curled as RDG's ``init_finger_pose`` does rather than
    flat open.

    Ordering note: this MUST run after the object reset, since the target frame is built from the
    object's post-reset pose. mjlab applies reset events in declaration order, so it is declared
    after ``reset_object``.
    """
    from blindgraspreflex.tasks.grasping.rdg_ik import (
        RDG_LENGTH_SCORE_COEFF,
        RDG_PROJECTION_CUTOFF,
        RDG_SAMPLE_NUM,
        sample_pregrasp_frames,
    )

    term = env.action_manager.get_term(action_name)
    ik = _pregrasp_ik(env, term)
    robot = env.scene["robot"]
    ids = env_ids
    n = len(ids)
    if n == 0:
        return

    # Flush the object placement written by the preceding reset event before reading it back.
    # mjlab applies reset events with NO intervening sim.forward, so data.root_link_pos_w still
    # holds the pre-write pose here -- reading it directly aimed every episode's IK at the origin
    # and parked the palm at [0, 0, standoff], 0.62 m from the object. rdg_reset_scene sidesteps
    # this by computing the object pose itself; this task's object reset samples internally, so
    # the pose has to be read, and therefore has to be forwarded first.
    env.sim.forward()
    # The BELIEVED object position, not the true one. This event places the arm 25 cm from the
    # object and aimed at it, so reading the true pose here would hand the policy a perfectly
    # aimed starting pose derived from ground truth and then corrupt only its observation --
    # measured, palm-to-TRUE-object was 0.2500 +- 0.0000 m under a 5 cm perception bias, i.e. an
    # exact aim every episode. The perception-sensitivity column would then understate this
    # policy's sensitivity, because the one thing the bias is supposed to move (where the arm
    # actually goes) was still being computed from the truth.
    #
    # Drawn HERE because mjlab runs reset events before action_manager.reset(): the term's own
    # reset would otherwise draw a different bias afterwards, and the arm would be aimed with one
    # estimate while the policy observed another.
    term.resample_object_pos_bias(ids)
    obj_pos = env.scene[object_name].data.root_link_pos_w[ids] + term._obj_pos_bias[ids]
    # Where the palm starts, relative to the object. ``to_view`` points from the object to the
    # standoff position, and the palm's +Z (its approach axis) points back down the same line, so
    # this direction sets BOTH where the hand starts and which way it faces.
    #
    # The default (0, 0, 1) is straight up, i.e. the palm 25 cm above the object with the fingers
    # pointing straight DOWN at it. That was hardcoded, and it is the only attitude the
    # joint-target baseline has ever been initialised in. A horizontal direction instead starts
    # the hand beside the object with the fingers horizontal -- the palm-parallel-to-the-table
    # attitude the factorised policy holds. Tune it in scripts/viser_pregrasp_init.py.
    to_view = (
        torch.as_tensor(approach_dir, dtype=torch.float32, device=obj_pos.device)
        .reshape(1, 3)
        .expand(n, 3)
        .clone()
    )
    to_view = to_view / torch.linalg.norm(to_view, dim=-1, keepdim=True)
    approach = -to_view
    target_pos = obj_pos + float(standoff) * to_view

    s = RDG_SAMPLE_NUM
    from blindgraspreflex.tasks.grasping.rdg_ik import _quat_from_matrix

    if orientation == "home":
        # THE FLAT-HOLD HOME ATTITUDE -- the pose the arm actually starts every episode in, the
        # one the factorised policy holds throughout, and the one visible at t=0 in
        # `play_grasp.py --tune-reach`. Measured: palm normal straight down (world z -1.00) with
        # the fingers horizontal (world z -0.09), i.e. the palm parallel to the table.
        #
        # Nothing is aimed at the object, so there is no roll to sample and no closing axis to
        # score -- the attitude is the same every episode and only the standoff position varies.
        # This is deliberately NOT the RDG behaviour below, which spins the wrist to aim an axis
        # down the approach line and therefore starts the hand at a different attitude for every
        # object pose.
        s = 1
        perp = None
        tq = term._home_site_quat.reshape(1, 1, 4).expand(n, 1, 4)
    else:
        # aim_axis="x" points the palm's NORMAL at the object (palm facing it); "z" points the
        # palm's finger axis at it, which is RDG's convention.
        rot, perp = sample_pregrasp_frames(approach, s, aim_axis=aim_axis)  # (n,S,3,3), (n,S,3)
        tq = _quat_from_matrix(rot.reshape(-1, 3, 3)).reshape(n, s, 4)
    tp = target_pos.unsqueeze(1).expand(n, s, 3)
    q0 = (
        robot.data.joint_pos[ids][:, term._arm_joint_ids]
        .unsqueeze(1)
        .expand(n, s, len(ARM_JOINT_NAMES))
    )
    q_ik, pos_err, rot_err = ik.solve(
        tp.reshape(-1, 3), tq.reshape(-1, 4), q0.reshape(-1, len(ARM_JOINT_NAMES))
    )
    q_ik = q_ik.reshape(n, s, -1)
    infeasible = (pos_err.reshape(n, s) > 0.02) | (rot_err.reshape(n, s) > 0.35)

    if perp is None:
        # Fixed attitude: one candidate, nothing to choose between. The IK residual is still
        # reported below, because an unreachable standoff must not pass silently.
        best = torch.zeros(n, dtype=torch.long, device=q_ik.device)
    else:
        # Score each candidate roll by the object's extent along that candidate's closing axis, so
        # the hand closes across the object's narrow dimension. Without a point cloud here, the
        # extent is approximated by the object's AABB projected on the closing axis.
        half = _object_half_extent(env, object_name)[ids]  # (n, 3)
        width = (perp.abs() * half.unsqueeze(1)).sum(-1) * 2.0  # (n, S)
        score = RDG_LENGTH_SCORE_COEFF * width
        score = score + 1e4 * (width >= RDG_PROJECTION_CUTOFF).float() + 1e4 * infeasible.float()
        best = score.argmin(dim=1)
    q_best = torch.gather(q_ik, 1, best.view(n, 1, 1).expand(n, 1, q_ik.shape[-1])).squeeze(1)

    # Partially opened fingers, between the open pre-grasp pose and the closed pose.
    f = float(hand_open_frac)
    hand_q = ((1.0 - f) * term._q_open + f * term._q_close).expand(n, -1)

    q_full = robot.data.joint_pos[ids].clone()
    q_full[:, term._arm_joint_ids] = q_best
    q_full[:, term._hand_joint_ids] = hand_q
    qd_full = torch.zeros_like(q_full)
    robot.write_joint_state_to_sim(q_full, qd_full, env_ids=ids)

    if init_vel is not None and any(abs(v) > 1e-9 for v in init_vel):
        # Start the palm already MOVING, rather than from rest. The commanded Cartesian velocity
        # is mapped to joint velocities with the same damped pseudoinverse the arm's own velocity
        # controller uses, so the initial state is one the arm law could itself have produced:
        #   qd = J^T (J J^T + lambda^2 I)^-1 [v; 0]
        # Forward first -- the Jacobian has to be evaluated at the configuration just written, not
        # at the previous episode's.
        env.sim.forward()
        data = term._entity.data
        site_pos = data.site_pos_w[:, term._site_id]
        J = term.impedance.jacobian.compute(data.body_link_pos_w, data.body_link_quat_w, site_pos)[
            ids
        ]  # (n, 6, 7), rows [lin; ang]
        v = torch.as_tensor(init_vel, dtype=torch.float32, device=q_full.device)
        twist = torch.zeros(n, 6, 1, device=q_full.device)
        twist[:, :3, 0] = v.reshape(1, 3).expand(n, 3)
        lam2 = float(term.cfg.vel_damping)
        JJt = J @ J.transpose(1, 2) + lam2 * torch.eye(6, device=J.device).unsqueeze(0)
        qd_arm = (J.transpose(1, 2) @ torch.linalg.solve(JJt, twist)).squeeze(-1)
        qd_arm = qd_arm.clamp(-term.cfg.qd_limit, term.cfg.qd_limit)
        qd_full[:, term._arm_joint_ids] = qd_arm
        robot.write_joint_state_to_sim(q_full, qd_full, env_ids=ids)
    # The hand servo integrates from wherever the fingers actually are; re-seat its internal
    # command so it does not spend the first steps chasing the previous episode's pose.
    term._q_cmd[ids] = hand_q
    term.hand_target[ids] = hand_q
    if getattr(term, "_arm_joint_target", False):
        # Both the commanded target and the integrated command, or the velocity field spends the
        # start of every episode travelling from the previous episode's pose.
        term._arm_target[ids] = q_best
        term._arm_cmd[ids] = q_best
        # Claim these envs so GraspAction.reset (which runs after every reset EVENT) leaves the
        # pre-grasp pose alone instead of resetting the command to the home pose.
        term._arm_cmd_seeded[ids] = True


def _pregrasp_ik(env, term):
    """Lazily build (and cache) the IK solver on the env -- compiling it per reset is wasteful."""
    ik = getattr(env, "_pregrasp_ik", None)
    if ik is None:
        from mjlab.entity.entity import Entity as _Entity

        from blindgraspreflex.tasks.grasping.rdg_ik import BatchedArmIK

        # A standalone compile of the robot spec, as GraspAction and RdgAction both do: a spec
        # attached by reference to the scene's parent spec cannot be compiled on its own.
        ik = BatchedArmIK(
            _Entity(_grasp_robot_cfg()).spec.compile(),
            ARM_JOINT_NAMES,
            "grasp_site",
            device=env.device,
        )
        env._pregrasp_ik = ik
    return ik


def _object_half_extent(env, object_name: str = "object") -> torch.Tensor:
    """Per-world object AABB half-extent (N, 3), cached. Approximates the point cloud's spread."""
    half = getattr(env, "_obj_half_extent", None)
    if half is None:
        import mujoco
        import numpy as np

        # Bounding radius of the object's geoms, spread over three axes. A crude stand-in for the
        # point cloud RDG scores against -- it only has to rank candidate wrist rolls, and every
        # candidate is scored against the same extent, so an isotropic approximation mostly
        # collapses the scoring to the feasibility term. Good enough to seed an approach; not a
        # shape feature.
        model = getattr(env.sim, "mj_model", None)
        r = 0.05
        if model is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
            if gs:
                r = float(np.array([model.geom_rbound[g] for g in gs]).max())
        half = torch.full((env.num_envs, 3), r / (3.0**0.5), device=env.device)
        env._obj_half_extent = half
    return half
