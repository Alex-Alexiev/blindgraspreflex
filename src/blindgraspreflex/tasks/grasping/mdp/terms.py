"""Observation and reward terms for the reflexive grasping task.

Ported from the IsaacLab ``brl_arm_robotis_5F_hand`` task. Written as pure tensor functions
plus small stateful helpers so each can be unit-tested without standing up an environment.

Two link sets, deliberately different
-------------------------------------
* **Observation** uses 21 links -- palm plus all 20 finger links -- which is simply every
  contactable body on the hand.
* **Reward** uses 16 -- palm plus phalanges 2/3/4 of each finger, in the exact order the
  IsaacLab ``contact_weights`` tuple assumes. The phalanx-1 abduction knuckles are dropped
  because the inward-force gate is not well defined on them: their local +X sits within
  9 deg of perpendicular to the grasp direction, so the gate would be near-random there.

The gate itself only counts a contact whose force pushes into the *inside* of the link. For
the phalanges that is local +X (verified: +X dotted with the direction to the grasp centre is
0.27-0.98 across the hand). For the palm it is also local +X, the palm face: the fingers flex
toward it, and the palm's +Z runs from the wrist to the knuckles.

Improvement rewards
-------------------
``contact_coverage`` and ``target_height_error`` are both *improvement* rewards: they pay out
``max(best_so_far - current, 0)`` and then lower the watermark, so over an episode they
telescope to total improvement and cannot be farmed by oscillating. That makes them stateful,
and the state must be reset per episode -- see :class:`StatefulMinBuffer`.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

__all__ = [
    "HandContacts",
    "ObjectRestHeight",
    "StatefulMinBuffer",
    "StatefulOnceBuffer",
    "contact_coverage_improvement",
    "hand_joint_tracking_error",
    "lifted_reward",
    "palm_to_object_vector",
    "target_height_error_improvement",
]

# Order assumed by CONTACT_WEIGHTS below. Must not be reordered: the IsaacLab reward pads or
# truncates a mismatched weight tuple silently rather than raising.
REWARD_CONTACT_LINKS = (
    "finger_r_link_1_thumb2",
    "finger_r_link_1_thumb3",
    "finger_r_link_1_thumb4",
    "finger_r_link_2_index2",
    "finger_r_link_2_index3",
    "finger_r_link_2_index4",
    "finger_r_link_3_middle2",
    "finger_r_link_3_middle3",
    "finger_r_link_3_middle4",
    "finger_r_link_4_ring2",
    "finger_r_link_4_ring3",
    "finger_r_link_4_ring4",
    "finger_r_link_5_little2",
    "finger_r_link_5_little3",
    "finger_r_link_5_little4",
    "palm",
)

CONTACT_WEIGHTS = (
    5.0,
    5.0,
    10.0,  # thumb: weighted up, an opposing thumb is what makes a grasp
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,  # palm
)


##
# Contacts.
##


class HandContacts:
    """Extracts per-link binary contact against the object, with the inward-force gate.

    Loops over ``data.contact`` on the host. That is fine for the verification harness but is
    not what a batched env should use -- mjlab's ``ContactSensorCfg`` does this on device and
    should replace it in the task.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        object_body_name: str = "object",
        force_threshold: float = 0.0,
        use_inward_gate: bool = True,
    ) -> None:
        self.force_threshold = force_threshold
        self.use_inward_gate = use_inward_gate
        self.object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)

        obs_links = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if name == "palm" or name.startswith("finger_r_link_"):
                obs_links.append((i, name))
        self.obs_body_ids = [i for i, _ in obs_links]
        self.obs_names = [n for _, n in obs_links]
        self.obs_index = {i: k for k, i in enumerate(self.obs_body_ids)}

        self.reward_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in REWARD_CONTACT_LINKS
        ]
        if any(b < 0 for b in self.reward_body_ids):
            raise ValueError("Reward contact links missing from the model.")
        self.reward_index = {i: k for k, i in enumerate(self.reward_body_ids)}
        # Every reward link's inward direction is local +X: the palm face and the finger pads.
        self.inward_local = np.array([[1.0, 0.0, 0.0]] * len(REWARD_CONTACT_LINKS))

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs_binary (21,), reward_binary (16,)) for one env."""
        obs = np.zeros(len(self.obs_body_ids), dtype=bool)
        rew = np.zeros(len(self.reward_body_ids), dtype=bool)
        wrench = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if (b1 == self.object_body) == (b2 == self.object_body):
                continue  # not an object<->link contact
            link = b2 if b1 == self.object_body else b1
            if link not in self.obs_index and link not in self.reward_index:
                continue
            mujoco.mj_contactForce(model, data, i, wrench)
            normal = abs(float(wrench[0]))
            if normal <= self.force_threshold:
                continue
            if link in self.obs_index:
                obs[self.obs_index[link]] = True
            if link in self.reward_index:
                k = self.reward_index[link]
                if self.use_inward_gate:
                    # contact frame row 0 is the normal, pointing from geom1 to geom2, so
                    # normal_w points object->link. On the link's inside face it opposes inward.
                    frame = con.frame.reshape(3, 3)
                    normal_w = frame[0] * (1.0 if b1 == self.object_body else -1.0)
                    inward_w = data.xmat[link].reshape(3, 3) @ self.inward_local[k]
                    if float(np.dot(normal_w, inward_w)) >= 0.0:
                        continue
                rew[k] = True
        return obs, rew


##
# Stateful buffers.
##


class StatefulMinBuffer:
    """Best-so-far watermark for improvement rewards.

    ``+inf`` means "not yet seeded". The first eligible step only seeds and pays nothing;
    thereafter the payout is ``max(best - current, 0)`` and the watermark drops. A masked-out
    step neither pays nor updates -- the watermark *freezes* rather than resetting, which is
    what makes the gate a true eligibility condition and not a reward multiplier.
    """

    def __init__(self, num_envs: int, device: torch.device | str = "cpu") -> None:
        self.best = torch.full((num_envs,), float("inf"), device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.best[env_ids] = float("inf")

    def step(self, error: torch.Tensor, update_mask: torch.Tensor | None = None) -> torch.Tensor:
        if update_mask is None:
            update_mask = torch.ones_like(error, dtype=torch.bool)
        seeded = torch.isfinite(self.best)
        reward = torch.zeros_like(error)
        eligible = update_mask & seeded
        reward[eligible] = (self.best[eligible] - error[eligible]).clamp(min=0.0)
        improved = update_mask & ((~seeded) | (error < self.best))
        self.best[improved] = error[improved]
        return reward


class StatefulOnceBuffer:
    """One-shot latch: pays 1 the first time a condition holds in an episode, then never again."""

    def __init__(self, num_envs: int, device: torch.device | str = "cpu") -> None:
        self.fired = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.fired[env_ids] = False

    def step(self, condition: torch.Tensor) -> torch.Tensor:
        first = condition & (~self.fired)
        self.fired[first] = True
        return first.float()


class ObjectRestHeight:
    """Captures each object's settled height once per episode, after a delay.

    The height reward is measured against where the object actually came to rest, not its
    spawn height, because it is dropped from above and would otherwise be credited for the
    fall. Until capture the reward is gated off entirely, so a still-falling object cannot
    seed the watermark with a meaningless value.
    """

    def __init__(
        self, num_envs: int, settle_steps: int, device: torch.device | str = "cpu"
    ) -> None:
        self.settle_steps = max(1, settle_steps)
        self.height = torch.zeros(num_envs, device=device)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.ready[env_ids] = False
        self.counter[env_ids] = 0

    def step(self, object_z: torch.Tensor) -> None:
        pending = ~self.ready
        self.counter[pending] += 1
        capture = pending & (self.counter >= self.settle_steps)
        self.height[capture] = object_z[capture]
        self.ready[capture] = True


##
# Rewards.
##


def contact_coverage_improvement(
    contact_binary: torch.Tensor,
    buffer: StatefulMinBuffer,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Pay for newly covering hand links, weighted. Error is the weighted *uncovered* count."""
    uncovered = ((1.0 - contact_binary.float()) * weights.unsqueeze(0)).sum(dim=1)
    return buffer.step(uncovered)


def target_height_error_improvement(
    object_z: torch.Tensor,
    rest: ObjectRestHeight,
    buffer: StatefulMinBuffer,
    target_height: float,
    contact_binary: torch.Tensor | None = None,
    min_contact_links: int = 0,
) -> torch.Tensor:
    """Pay for new progress toward ``target_height`` above the settled rest height."""
    lifted = (object_z - rest.height).clamp(min=0.0)
    error = (target_height - lifted).clamp(min=0.0)
    update_mask = rest.ready.clone()
    if min_contact_links > 0:
        if contact_binary is None:
            raise ValueError("contact_binary required when min_contact_links > 0.")
        update_mask &= contact_binary.sum(dim=1) >= min_contact_links
    return buffer.step(error, update_mask)


def lifted_reward(
    object_z: torch.Tensor,
    rest: ObjectRestHeight,
    buffer: StatefulOnceBuffer,
    z_lift: float,
    relative_to_rest: bool = False,
) -> torch.Tensor:
    """One-shot bonus the first time the object exceeds ``z_lift``.

    ``relative_to_rest=False`` reproduces IsaacLab, where the threshold is an absolute
    env-local height. Note that measures from the floor, not the tabletop, and has no contact
    gate -- an object merely knocked upward latches the bonus.
    """
    height = (object_z - rest.height) if relative_to_rest else object_z
    return buffer.step(height > z_lift)


##
# Observations.
##


def palm_to_object_vector(
    object_pos_w: torch.Tensor, palm_pos_w: torch.Tensor, palm_quat_w: torch.Tensor
) -> torch.Tensor:
    """Object position relative to the palm, expressed in the palm frame.

    Palm frame rather than world so the observation is invariant to where in the workspace
    the arm happens to be -- the policy sees the same numbers for the same relative geometry.
    """
    from blindgraspreflex.controllers.batched import quat_conjugate, quat_rotate

    return quat_rotate(quat_conjugate(palm_quat_w), object_pos_w - palm_pos_w)


def hand_joint_tracking_error(joint_pos: torch.Tensor, joint_target: torch.Tensor) -> torch.Tensor:
    """Commanded minus achieved joint angle -- how far the hand is from its own setpoint.

    Non-zero mainly where a finger is blocked by the object, so it doubles as a soft contact
    signal that degrades gracefully where the binary contacts are all-or-nothing.
    """
    return joint_target - joint_pos
