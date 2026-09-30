"""Batched torch ports of the arm stack, for use inside mjlab action terms.

The single-env NumPy classes in :mod:`~blindgraspreflex.controllers.arm_impedance` and
:mod:`~blindgraspreflex.controllers.grasp_phase` stay the reference implementation; these mirror
them tensor-for-tensor so the two can be cross-checked (see ``tests``-style parity script).

Two deliberate departures from the NumPy reference, both forced by what mjlab exposes:

* **Jacobian transpose, not operational-space control.** ``Lambda = (J M^-1 J^T)^-1`` needs a
  mass matrix and mjlab exposes neither ``qM`` nor a host-callable ``mul_m``. Measured on
  this arm the two are equivalent in free space (sub-mm step error either way) and ``J^T``
  actually saturates less, so the loss is confined to the palm's effective stiffness varying
  with arm configuration rather than being normalised flat.
* **Gravity compensation computed in the controller**, from body masses and centres of mass,
  since mjlab exposes no ``qfrc_bias``. Keeping it here rather than in the MJCF's per-body
  ``gravcomp`` means the same model serves a compensated and an uncompensated policy, and the
  term can be scaled or ablated without recompiling.

``joint_damping`` defaults to 0. The explicit damping torque IsaacLab used is bound by
``dt < 2 I / kd``; at the wrist that is 0.0025 s, so at the task's 0.005 s timestep it does
not blow up but silently degrades tracking to ~20 mm and ~17 deg. The MJCF's own ``damping``
is handled implicitly by the integrator and does the job without the stability cliff.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from blindgraspreflex.controllers.kinematics import ArmJacobian, GravityCompensation

__all__ = [
    "BatchedArmImpedance",
    "BatchedGraspPhase",
    "BatchedGraspPhaseCfg",
    "quat_conjugate",
    "quat_mul",
    "quat_rotate",
    "quat_to_rotvec",
]


def _parse_vec3(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse a "x,y,z" env-var override, falling back to ``default`` if unset or malformed."""
    if not raw:
        return default
    try:
        parts = tuple(float(v) for v in raw.split(","))
    except ValueError:
        return default
    return parts if len(parts) == 3 else default


##
# Quaternion helpers (wxyz, batched).
##


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., :1]
    xyz = q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def euler_xyz_to_quat(rpy: torch.Tensor) -> torch.Tensor:
    """Extrinsic-xyz Euler (roll, pitch, yaw) [rad] -> wxyz quat.

    Matches ``scipy.spatial.transform.Rotation.from_euler("xyz", rpy)`` (q = qz.qy.qx), so the
    REACH-pose randomisation composes exactly like the pose-tuning GUI the bounds were read from.
    """
    half = rpy * 0.5
    c, s = torch.cos(half), torch.sin(half)
    z = torch.zeros_like(c[..., :1])
    qx = torch.cat([c[..., 0:1], s[..., 0:1], z, z], dim=-1)
    qy = torch.cat([c[..., 1:2], z, s[..., 1:2], z], dim=-1)
    qz = torch.cat([c[..., 2:3], z, z, s[..., 2:3]], dim=-1)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation vector (axis * angle, radians) of a wxyz quaternion.

    Sign-normalised to the shortest path first: a quaternion and its negation are the same
    rotation, but naive extraction would return a ``2*pi - theta`` rotation the wrong way
    round, which reads as a huge orientation error and slams the controller into saturation.
    """
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    cos_half = q[..., :1].clamp(-1.0, 1.0)
    angle = 2.0 * torch.atan2(sin_half, cos_half)
    # angle/sin_half -> 2 as the rotation vanishes; the limit keeps it finite at identity.
    scale = torch.where(
        sin_half > eps, angle / sin_half.clamp(min=eps), torch.full_like(angle, 2.0)
    )
    return xyz * scale


##
# Arm impedance.
##


@dataclass
class BatchedImpedanceGains:
    stiffness: float = 300.0
    damping: float = 80.0
    orientation_stiffness: float = 8.0
    orientation_damping: float = 1.5
    joint_damping: float = 2.0  # joint-space damping, matches IsaacLab; steadies the arm
    gravity_comp: float = 1.0
    """Scale on the gravity-compensation torque. 1 cancels gravity, 0 disables it."""


class BatchedArmImpedance:
    """Cartesian spring/damper on a palm site, ``tau = J^T [F; M]``, batched over envs."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        site_name: str,
        num_envs: int,
        device: torch.device | str = "cpu",
        gains: BatchedImpedanceGains | None = None,
        prefix: str = "",
        body_offset: int = 0,
    ) -> None:
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.gains = gains if gains is not None else BatchedImpedanceGains()
        self.jacobian = ArmJacobian(
            model,
            joint_names,
            site_name,
            device=self.device,
            prefix=prefix,
            body_offset=body_offset,
        )
        self.gravity = GravityCompensation(
            model, joint_names, device=self.device, prefix=prefix, body_offset=body_offset
        )
        self.site_id = self.jacobian.site_id

        act_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n) for n in joint_names
        ]
        if any(a < 0 for a in act_ids):
            raise ValueError("Each arm joint needs a like-named actuator.")
        self.act_adr = np.array(act_ids)
        limited = model.actuator_ctrllimited[self.act_adr].astype(bool)
        effort = np.where(
            limited,
            np.abs(model.actuator_ctrlrange[self.act_adr, 1]),
            np.abs(model.actuator_forcerange[self.act_adr, 1]),
        )
        self.effort_limit = torch.as_tensor(effort, dtype=torch.float32, device=self.device)

        self.target_pos = torch.zeros(num_envs, 3, device=self.device)
        self.target_quat = torch.zeros(num_envs, 4, device=self.device)
        self.target_quat[:, 0] = 1.0

    def set_target(
        self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)
        if env_ids is None:
            self.target_pos[:] = pos
            self.target_quat[:] = quat
        else:
            self.target_pos[env_ids] = pos
            self.target_quat[env_ids] = quat

    def pose_error(
        self, site_pos_w: torch.Tensor, site_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_err = self.target_pos - site_pos_w
        rot_err = quat_to_rotvec(quat_mul(self.target_quat, quat_conjugate(site_quat_w)))
        return pos_err, rot_err

    def compute(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        site_pos_w: torch.Tensor,
        site_quat_w: torch.Tensor,
        joint_vel: torch.Tensor,
        body_com_pos_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Arm torques, (num_envs, num_arm_joints). ``joint_vel`` is the arm block only.

        ``body_com_pos_w`` is required when ``gains.gravity_comp`` is non-zero.
        """
        jac = self.jacobian.compute(body_pos_w, body_quat_w, site_pos_w)  # (N, 6, J)
        # Site twist from the Jacobian rather than mjlab's site_vel_w: guaranteed consistent
        # with the J^T mapping used a line later, so the damping term cannot disagree with
        # the stiffness term about what "velocity of this point" means.
        twist = torch.bmm(jac, joint_vel.unsqueeze(-1)).squeeze(-1)  # (N, 6)

        pos_err, rot_err = self.pose_error(site_pos_w, site_quat_w)
        g = self.gains
        wrench = torch.cat(
            [
                g.stiffness * pos_err - g.damping * twist[:, :3],
                g.orientation_stiffness * rot_err - g.orientation_damping * twist[:, 3:],
            ],
            dim=-1,
        )
        torque = torch.bmm(jac.transpose(1, 2), wrench.unsqueeze(-1)).squeeze(-1)
        if g.joint_damping:
            torque = torque - g.joint_damping * joint_vel
        if g.gravity_comp:
            if body_com_pos_w is None:
                raise ValueError("body_com_pos_w is required when gravity_comp is enabled.")
            torque = torque + g.gravity_comp * self.gravity.compute(
                body_pos_w, body_quat_w, body_com_pos_w
            )
        return torch.clamp(torque, -self.effort_limit, self.effort_limit)


##
# Grasp phase.
##


@dataclass
class BatchedGraspPhaseCfg:
    # Kept in lockstep with controllers.grasp_phase.GraspPhaseCfg. These are the values tuned
    # interactively in the playground; a silent divergence here means training runs against a
    # controller nobody has ever watched.
    pre_grasp_offset: tuple[float, float, float] = (0.0, 0.0, -0.01)
    offset_frame: str = "palm"
    # Fixed REACH orientation offset (roll, pitch, yaw) [rad], composed onto the flat-hold attitude
    # in the palm/body frame when reach_randomize is False -- the tuned grasp attitude the arm
    # presents to every object. Identity by default.
    reach_rpy_fixed: tuple[float, float, float] = (0.0, 0.0, 0.0)
    approach_height: float = 0.25
    descend_xy_tolerance: float = 0.04
    # 0.15 m matches the IsaacLab GraspPhaseSpringDamperControlAction leash. It has to be wider
    # than the policy's arm residual (+-0.05 m) so the residual actually moves the setpoint
    # instead of being clipped away against the palm.
    max_setpoint_radius: float = 0.15
    score_low: float = 0.45
    score_high: float = 0.55
    min_setpoint_z: float = 0.0
    # In REACH, the policy's arm residual is bounded so the commanded setpoint stays within this
    # radius [m] of the live object position -- the residual can nudge the palm around the object
    # to line up a grasp, but cannot walk the arm off the object. 0 disables the bound. Only
    # applies in REACH; LIFT drives to the fixed home retreat pose, which is meant to be far.
    reach_object_radius: float = 0.10
    # Domain randomisation of the REACH palm pose (the arm-agnostic alternative to a hand-driven
    # arm residual): each episode the REACH target = object_centre + a sampled world-frame offset,
    # oriented as (flat hold attitude) x (sampled roll/pitch/yaw). Bounds are per-axis (min, max);
    # position in metres, orientation in radians. When ``reach_randomize`` is False this is off and
    # REACH uses the fixed ``pre_grasp_offset`` (plus any residual).
    reach_randomize: bool = False
    reach_pos_bounds: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    reach_rpy_bounds: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    # Arm->hand engagement gate (used only when GraspActionCfg.hand_gate is on). The arm tells the
    # hand when to grasp: the hand policy controls the fingers only once the palm is within
    # engage_radius [m] of the object; farther out the fingers are held at a default pre-grasp
    # pose. Schmitt-latched (release_radius > engage_radius) so it does not chatter at the boundary.
    # Engagement holds through LIFT (the object rides with the palm, so the distance stays small)
    # and releases if the object is dropped (distance grows), letting the hand reopen to re-grasp.
    engage_radius: float = 0.10
    release_radius: float = 0.15
    # Release the hand on the POLICY'S OWN grasp_score falling, not on the palm drifting away from
    # the believed object. The distance test assumes "the object rides with the palm, so the
    # distance stays small" through LIFT -- true only while the estimate tracks the object. Freeze
    # or corrupt the estimate and lifting itself walks the palm away from the believed point, trips
    # release_radius, and the gate commands the fingers OPEN mid-lift: measured on the apple, palm
    # at 0.239 m against a belief frozen at 0.005 m, object dropped from 0.181 m, every time.
    # Score-based release keeps drop-detection (grasp_score collapses within ONE control step of
    # losing the object, 36/36 crossings of score_low in the dislodge study) without any dependence
    # on object pose. Env-var toggle rather than a default change: the shipped policies were
    # trained under the distance gate.
    release_on_score: bool = bool(os.environ.get("REFLEXIVE_RELEASE_ON_SCORE"))

    # TIMED-LIFT BASELINE (REFLEXIVE_TIMED_LIFT = seconds, 0 = off). Replaces the whole
    # grasp_score feedback loop with a clock: the hand engages on proximity as usual, then the arm
    # lifts a fixed time later and never comes back. Engagement is made STICKY in this mode, so
    # there is no release and no re-grasp -- which is the point. It isolates how much of the
    # method's success comes from CLOSING the loop on grasp quality rather than from the
    # scripted approach plus a competent hand.
    timed_lift_s: float = float(os.environ.get("REFLEXIVE_TIMED_LIFT", "0") or 0)
    # Hold the REACH palm attitude through LIFT instead of snapping back to the flat hold.
    #
    # By default the attitude target is the reach attitude while reaching and the flat hold once
    # the score latches LIFT -- so at the instant the fingers have just closed, the wrist rotates
    # back to level while holding the object. At the tuned attitude that rotation is 11 deg and
    # invisible. Under the palm-attitude sweep (orientations/) it is the whole swept angle, and a
    # grasp taken at roll 80 deg is then asked to survive an 80 deg wrist rotation under load --
    # which measures the rotation-back, not the grasp. With this set the commanded attitude is the
    # same from approach through grasp to carry, which is the question that experiment is asking.
    #
    # Env-var toggle rather than a default change: every existing number, and the deploy stack,
    # were produced with the snap-back, and the LIFT attitude is also what the hold reward and the
    # success criterion were tuned around.
    hold_attitude_through_lift: bool = bool(os.environ.get("REFLEXIVE_HOLD_REACH_ATTITUDE"))


class BatchedGraspPhase:
    """Schmitt-triggered REACH/LIFT setpoint generator, batched over envs.

    Phase and descent are both *latched* per env, so this carries state that must be reset
    on episode boundaries -- see :meth:`reset`.
    """

    REACH = 0
    LIFT = 1

    def __init__(
        self,
        cfg: BatchedGraspPhaseCfg,
        num_envs: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        # Apply the policy's POSITION residual in the palm frame (rotated by the reach attitude)
        # instead of the world frame. Off by default -- the deployed world-frame policies must not
        # change under them. Set REFLEXIVE_PALM_FRAME_RESIDUAL before the env is built.
        import os

        self._palm_frame_residual = bool(os.environ.get("REFLEXIVE_PALM_FRAME_RESIDUAL"))
        # LAYERED reach-pose domain randomisation. Distinct from cfg.reach_randomize, which is the
        # arm-agnostic regime that REPLACES the tuned pose and drops the hand's residual entirely.
        # This instead perturbs the tuned pose per episode -- object + tuned offset + sampled offset,
        # attitude (flat hold) x (tuned rpy) x (sampled rpy) -- and keeps the residual on top, so the
        # hand sees a different approach pose every episode and must grasp robustly regardless.
        # Off by default; set REFLEXIVE_REACH_DR before the env is built (inherited by distill/eval).
        self._reach_dr = bool(os.environ.get("REFLEXIVE_REACH_DR"))
        dr_pos = _parse_vec3(os.environ.get("REFLEXIVE_REACH_DR_POS"), (0.02, 0.02, 0.015))
        dr_rpy_deg = _parse_vec3(os.environ.get("REFLEXIVE_REACH_DR_RPY"), (15.0, 10.0, 20.0))
        self._dr_pos_mag = torch.tensor(dr_pos, dtype=torch.float32, device=self.device)
        self._dr_rpy_mag = torch.tensor(
            [math.radians(v) for v in dr_rpy_deg], dtype=torch.float32, device=self.device
        )
        if self._reach_dr:
            # The sampled offset moves the palm further from the object, and two fixed radii would
            # silently eat the DR if left alone: engage_radius (the hand gate never fires, so the
            # fingers stay at the default pre-grasp pose and the episode is an automatic failure that
            # teaches nothing) and reach_object_radius (the setpoint is projected back, clipping the
            # randomisation away). Widen both by the worst-case sampled displacement so the full DR
            # range is actually realised. max(...) so an already-wider config is never narrowed.
            span = float(torch.linalg.norm(self._dr_pos_mag))
            cfg.engage_radius = max(cfg.engage_radius, cfg.engage_radius + span)
            cfg.release_radius = max(cfg.release_radius, cfg.release_radius + span)
            cfg.reach_object_radius = max(cfg.reach_object_radius, cfg.reach_object_radius + span)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.descending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # Latched arm->hand engagement flag (see cfg.engage_radius). Read by the action term to gate
        # the hand policy; cleared on reset so every episode starts disengaged (fingers at default).
        self.engaged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # Control steps a world has spent engaged, for cfg.timed_lift_s. control_dt is set
        # by the action term from the env's own step_dt; the default is only a fallback.
        self.engaged_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.control_dt = 0.084
        self.lift_pos = torch.zeros(num_envs, 3, device=self.device)
        self.lift_quat = torch.zeros(num_envs, 4, device=self.device)
        self.lift_quat[:, 0] = 1.0
        # Optional external override of the LIFT-phase target position (x, y, z), e.g. the viewer's
        # "move the grasped object" sliders. ``None`` -> use the fixed home retreat pose. Broadcasts
        # over envs (shape (3,) or (num_envs, 3)); orientation stays the home hold attitude.
        self.lift_override_pos: torch.Tensor | None = None
        # Final (pre-clamp) target, set each update; here so it exists before the first call.
        self.base_target = torch.zeros(num_envs, 3, device=self.device)
        self.offset = torch.as_tensor(cfg.pre_grasp_offset, dtype=torch.float32, device=self.device)
        # Fixed tuned REACH attitude offset (1, 4), broadcasts over envs; identity unless set.
        self.reach_quat_fixed = euler_xyz_to_quat(
            torch.tensor([cfg.reach_rpy_fixed], dtype=torch.float32, device=self.device)
        )
        # PALM-ATTITUDE SWEEP. REFLEXIVE_REACH_RPY="roll,pitch,yaw" in DEGREES rotates the palm the
        # arm presents to the object, composed on the RIGHT of the tuned attitude and so applied in
        # the palm's own frame -- "roll the hand about its approach axis", not "rotate it in the
        # world". An OFFSET rather than an absolute pose, so "0,0,0" is exactly the tuned attitude
        # every existing number was produced at and the sweep has a real reference point at 0.
        #
        # Only the REACH attitude moves. LIFT keeps the plain home hold attitude, so the carry and
        # the success criterion are the same at every angle and what is being varied is how the
        # hand is presented to the object, nothing else. Read here, like REFLEXIVE_REACH_DR above,
        # so it must be set before the env is built.
        self.reach_rpy_sweep = _parse_vec3(os.environ.get("REFLEXIVE_REACH_RPY"), (0.0, 0.0, 0.0))
        if any(self.reach_rpy_sweep):
            sweep = euler_xyz_to_quat(
                torch.tensor(
                    [[math.radians(v) for v in self.reach_rpy_sweep]],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.reach_quat_fixed = quat_mul(self.reach_quat_fixed, sweep)

        # Per-episode REACH-pose randomisation offsets (identity until sampled).
        self.reach_pos_offset = torch.zeros(num_envs, 3, device=self.device)
        self.reach_quat_offset = torch.zeros(num_envs, 4, device=self.device)
        self.reach_quat_offset[:, 0] = 1.0
        self._pos_lo = torch.tensor([b[0] for b in cfg.reach_pos_bounds], device=self.device)
        self._pos_hi = torch.tensor([b[1] for b in cfg.reach_pos_bounds], device=self.device)
        self._rpy_lo = torch.tensor([b[0] for b in cfg.reach_rpy_bounds], device=self.device)
        self._rpy_hi = torch.tensor([b[1] for b in cfg.reach_rpy_bounds], device=self.device)

        # Per-episode LAYERED DR perturbations (see _reach_dr above). Separate buffers from the
        # reach_randomize ones so the two regimes can never quietly read each other's samples.
        self.dr_pos_offset = torch.zeros(num_envs, 3, device=self.device)
        self.dr_quat_offset = torch.zeros(num_envs, 4, device=self.device)
        self.dr_quat_offset[:, 0] = 1.0

    def sample_reach(self, env_ids: torch.Tensor) -> None:
        """Draw fresh REACH position + orientation offsets for ``env_ids`` from the bounds."""
        n = int(env_ids.numel()) if torch.is_tensor(env_ids) else len(env_ids)
        if self.cfg.reach_randomize:
            u = torch.rand(n, 3, device=self.device)
            self.reach_pos_offset[env_ids] = self._pos_lo + u * (self._pos_hi - self._pos_lo)
            v = torch.rand(n, 3, device=self.device)
            rpy = self._rpy_lo + v * (self._rpy_hi - self._rpy_lo)
            self.reach_quat_offset[env_ids] = euler_xyz_to_quat(rpy)
        if self._reach_dr:
            # Symmetric +-magnitude perturbations of the tuned pose, redrawn every episode.
            u = torch.rand(n, 3, device=self.device) * 2.0 - 1.0
            self.dr_pos_offset[env_ids] = u * self._dr_pos_mag
            v = torch.rand(n, 3, device=self.device) * 2.0 - 1.0
            self.dr_quat_offset[env_ids] = euler_xyz_to_quat(v * self._dr_rpy_mag)

    def reset(
        self,
        env_ids: torch.Tensor,
        lift_pos: torch.Tensor,
        lift_quat: torch.Tensor,
    ) -> None:
        self.phase[env_ids] = self.REACH
        self.descending[env_ids] = False
        self.engaged[env_ids] = False
        self.engaged_steps[env_ids] = 0
        self.lift_pos[env_ids] = lift_pos
        self.lift_quat[env_ids] = lift_quat
        self.sample_reach(env_ids)

    def update(
        self,
        grasp_score: torch.Tensor,
        object_pos_w: torch.Tensor,
        palm_pos_w: torch.Tensor,
        target_residual: torch.Tensor | None = None,
        rot_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the latches and return (target position, target quaternion).

        ``target_residual`` is the policy's per-env XYZ offset (already scaled to metres) added
        to the phase base target before the leash clamp; ``rot_residual`` is the policy's per-env
        palm-orientation offset (euler XYZ, radians, already scaled) applied as a bounded rotation
        in the palm/body frame. Together they give the hand policy a bounded say in both where and
        how the palm is presented -- ported from and extending IsaacLab's ``palm_target_residual``.
        """
        cfg = self.cfg
        score = grasp_score.clamp(0.0, 1.0).reshape(-1)

        # Arm->hand engagement (Schmitt-latched): is the palm close enough to the object to grasp?
        # Engage below engage_radius, release above release_radius, else hold. The action term reads
        # this to decide whether the hand policy or the default pre-grasp pose drives the fingers.
        dist_po = torch.linalg.norm(palm_pos_w - object_pos_w, dim=-1)
        # Engagement still UNLOCKS on proximity (the fingers wait at the pre-grasp pose until the
        # arm has brought them to the object). Only the RELEASE test changes: score-based release
        # cannot be tripped by the palm moving away from the object, which is what a lift is.
        if cfg.timed_lift_s > 0.0:
            # Sticky: engage on proximity, never release. No re-grasp by construction.
            self.engaged = self.engaged | (dist_po < cfg.engage_radius)
        else:
            release = (
                (score <= cfg.score_low) if cfg.release_on_score else (dist_po > cfg.release_radius)
            )
            self.engaged = torch.where(
                dist_po < cfg.engage_radius,
                torch.ones_like(self.engaged),
                torch.where(release, torch.zeros_like(self.engaged), self.engaged),
            )

        if cfg.timed_lift_s > 0.0:
            # grasp_score is ignored entirely: LIFT fires a fixed time after engagement and is
            # never given back. One-way, so a dip in grasp quality cannot trigger a re-approach.
            self.engaged_steps = self.engaged_steps + self.engaged.long()
            due = self.engaged_steps.float() * float(self.control_dt) >= cfg.timed_lift_s
            self.phase = torch.where(
                self.engaged & due, torch.full_like(self.phase, self.LIFT), self.phase
            )
        else:
            is_reach = self.phase == self.REACH
            self.phase = torch.where(
                is_reach & (score >= cfg.score_high),
                torch.full_like(self.phase, self.LIFT),
                torch.where(
                    (~is_reach) & (score <= cfg.score_low),
                    torch.full_like(self.phase, self.REACH),
                    self.phase,
                ),
            )
        reach = self.phase == self.REACH
        # LIFT target position: the fixed home retreat, unless externally overridden (viewer).
        lift_pos = self.lift_pos if self.lift_override_pos is None else self.lift_override_pos

        if cfg.reach_randomize:
            # Domain-randomised REACH pose (arm-agnostic training): the palm is presented at
            # object + sampled world offset, oriented (flat hold) x (sampled rpy). No residual and
            # no object-radius clamp -- the per-episode sampled bounds already constrain the pose.
            # LIFT is the fixed home retreat pose/attitude, unchanged.
            reach_orient = quat_mul(self.lift_quat, self.reach_quat_offset)
            target_quat = (
                reach_orient
                if cfg.hold_attitude_through_lift
                else torch.where(reach.unsqueeze(-1), reach_orient, self.lift_quat)
            )
            target = torch.where(
                reach.unsqueeze(-1), object_pos_w + self.reach_pos_offset, lift_pos
            )
        else:
            # Fixed tuned REACH pose (reach_randomize off): the palm is presented at a fixed
            # offset + attitude the tuner found, with the policy's *bounded* pos/rot residual
            # composed on top -- REACH only. LIFT keeps the plain home retreat pose/attitude, so
            # the residual never disturbs the hold.
            reach_quat = quat_mul(self.lift_quat, self.reach_quat_fixed)  # flat hold x tuned rpy
            if self._reach_dr:
                # Per-episode attitude perturbation, composed BEFORE the residual so the policy's
                # rot residual stays a bounded correction *relative to the pose it was handed*.
                reach_quat = quat_mul(reach_quat, self.dr_quat_offset)
            if rot_residual is not None:
                reach_quat = quat_mul(reach_quat, euler_xyz_to_quat(rot_residual))
            # Note the residual is folded into reach_quat above, so holding it through LIFT holds
            # the residual too. That is deliberate: the point is that the palm does not rotate at
            # the moment of the grasp, and the residual is part of the attitude the grasp was
            # taken at.
            target_quat = (
                reach_quat
                if cfg.hold_attitude_through_lift
                else torch.where(reach.unsqueeze(-1), reach_quat, self.lift_quat)
            )
            offset = self.offset.expand(self.num_envs, 3)
            if cfg.offset_frame == "palm":
                offset = quat_rotate(reach_quat, offset)
            # REACH goes straight to the grasp point (object + offset). The hover-above-then-descend
            # approach was removed: with the object tracked live, a target above the object center
            # lets a *grasped* object be carried upward while still in REACH -- lifting without ever
            # committing the grasp score to LIFT. Going directly to the object removes that path.
            reach_target = object_pos_w + offset
            # Per-episode position perturbation, in the same frame as the position residual below
            # (palm frame when REFLEXIVE_PALM_FRAME_RESIDUAL is set, world otherwise) so DR and
            # residual always speak the same coordinates.
            if self._reach_dr:
                dr_pos = self.dr_pos_offset
                if self._palm_frame_residual:
                    dr_pos = quat_rotate(reach_quat, dr_pos)
                reach_target = reach_target + dr_pos
            # Policy arm position residual: a bounded offset the hand adds so it can fine-tune where
            # the palm meets each object. World frame by default; if REFLEXIVE_PALM_FRAME_RESIDUAL is
            # set, rotate it into the palm frame (by the full reach attitude, incl. the rot residual),
            # so a residual axis means the same thing relative to the tilted palm regardless of pose.
            if target_residual is not None:
                if self._palm_frame_residual:
                    target_residual = quat_rotate(reach_quat, target_residual)
                reach_target = reach_target + target_residual
            # Bound the REACH setpoint to a radius around the live object, so the residual can only
            # reposition the palm *around* the object, never walk the arm off it.
            if cfg.reach_object_radius > 0.0:
                delta_obj = reach_target - object_pos_w
                dist_obj = torch.linalg.norm(delta_obj, dim=-1, keepdim=True)
                scale_obj = (cfg.reach_object_radius / dist_obj.clamp(min=1e-9)).clamp(max=1.0)
                reach_target = object_pos_w + delta_obj * scale_obj
            target = torch.where(reach.unsqueeze(-1), reach_target, lift_pos)
        # The *final* target the arm drives the palm to (before the leash clamp below). Exposed
        # for visualisation.
        self.base_target = target.clone()

        if cfg.max_setpoint_radius > 0.0:
            # Carrot-on-a-stick: the impedance controller only ever chases a setpoint at most
            # max_setpoint_radius ahead of the current palm, re-advanced each step. This caps
            # the spring force (~stiffness * radius) and turns the big LIFT jump into a smooth
            # bounded-rate pursuit toward base_target rather than a lurch.
            delta = target - palm_pos_w
            distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
            scale = (cfg.max_setpoint_radius / distance.clamp(min=1e-9)).clamp(max=1.0)
            target = palm_pos_w + delta * scale
        target = torch.cat([target[:, :2], target[:, 2:3].clamp(min=cfg.min_setpoint_z)], dim=-1)
        return target, target_quat
