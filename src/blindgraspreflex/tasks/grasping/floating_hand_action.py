"""Direct task-space palm velocity and joint-space finger control, without an arm."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from blindgraspreflex.assets.floating_hand import HAND_Q_START, get_floating_hand_cfg
from blindgraspreflex.assets.rizon4_hand import (
    HAND_EFFORT_LIMIT,
    HAND_EXPLICIT_PD,
    HAND_JOINT_NAMES,
    HAND_KD,
    HAND_KP,
)
from blindgraspreflex.controllers.batched import quat_mul, quat_rotate
from blindgraspreflex.tasks.grasping import mjlab_task as base

PALM_SITE_OFFSET = (0.0, 0.0, 0.1)


@dataclass(kw_only=True)
class FloatingHandActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    linear_limit: float = 0.35
    angular_limit: float = 2.0
    hand_v_max: float = 0.8
    hand_slowdown_distance: float = 0.35
    bounded: bool = False
    """Actions arrive already clipped to [-1, 1] (rl_games): map them linearly onto the twist
    limits and joint ranges. The default tanh/sigmoid squash would leave the outer quarter of the
    twist range and of each joint range unreachable."""
    home_goal_pos: tuple[float, float, float] = (0.522, 0.127, 0.513)
    """Lift/hold goal that mjlab_task's hold and success terms read: the arm task's LIFT retreat
    pose, 0.51 m above the mat next to the target's spawn point."""

    def build(self, env) -> FloatingHandAction:
        return FloatingHandAction(self, env)


class FloatingHandAction(ActionTerm):
    """Integrate a grasp-site twist and command all 20 real finger joints."""

    cfg: FloatingHandActionCfg

    def __init__(self, cfg: FloatingHandActionCfg, env) -> None:
        super().__init__(cfg, env)
        self._num = env.num_envs
        self._dt = float(env.physics_dt)
        self._origins = env.scene.env_origins
        self._site_offset = torch.tensor(PALM_SITE_OFFSET, device=env.device).expand(
            self._num, -1
        )
        ids, _ = self._entity.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        self._hand_joint_ids = ids
        self._reward_body_ids, self._inward_local = base.reward_contact_frames(
            self._entity, env.device
        )

        from mjlab.entity.entity import Entity

        model = Entity(get_floating_hand_cfg()).spec.compile()
        limits = []
        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            limits.append(model.jnt_range[jid].copy())
        ranges = torch.as_tensor(np.stack(limits), dtype=torch.float32, device=env.device)
        self._q_lo = ranges[:, 0].unsqueeze(0)
        self._q_hi = ranges[:, 1].unsqueeze(0)
        q_start = torch.tensor(
            [HAND_Q_START[name] for name in HAND_JOINT_NAMES], device=env.device
        ).unsqueeze(0)
        self._q_start = q_start.clamp(self._q_lo, self._q_hi)

        self._raw = torch.zeros(self._num, 6 + len(HAND_JOINT_NAMES), device=env.device)
        self._twist = torch.zeros(self._num, 6, device=env.device)
        self._site_pos = torch.zeros(self._num, 3, device=env.device)
        self._site_quat = torch.zeros(self._num, 4, device=env.device)
        self.hand_target = self._q_start.expand(self._num, -1).clone()
        self._q_cmd = self.hand_target.clone()
        self._hand_kp = torch.tensor(HAND_KP, device=env.device).unsqueeze(0)
        self._hand_kd = torch.tensor(HAND_KD, device=env.device).unsqueeze(0)
        self._home_goal_pos = torch.tensor(cfg.home_goal_pos, device=env.device)

    @property
    def action_dim(self) -> int:
        return 6 + len(HAND_JOINT_NAMES)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    @property
    def home_goal_pos(self) -> torch.Tensor:
        """Fixed lift/hold goal position. Shape (3,)."""
        return self._home_goal_pos

    def reset(self, env_ids) -> None:
        ids = torch.arange(self._num, device=self._site_pos.device) if env_ids is None else env_ids
        root = self._entity.data.default_root_state[ids]
        quat = root[:, 3:7]
        offset = quat_rotate(quat, self._site_offset[ids])
        self._site_pos[ids] = root[:, :3] + self._origins[ids] + offset
        self._site_quat[ids] = quat
        self.hand_target[ids] = self._q_start
        self._q_cmd[ids] = self._q_start
        self._twist[ids] = 0.0
        self._raw[ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = torch.nan_to_num(actions, nan=0.0)
        limits = actions.new_tensor([self.cfg.linear_limit] * 3 + [self.cfg.angular_limit] * 3)
        if self.cfg.bounded:
            bounded = self._raw.clamp(-1.0, 1.0)
            self._twist[:] = bounded[:, :6] * limits
            alpha = (bounded[:, 6:] + 1.0) / 2.0
        else:
            self._twist[:] = torch.tanh(self._raw[:, :6]) * limits
            alpha = torch.sigmoid(self._raw[:, 6:])
        self.hand_target[:] = self._q_lo + alpha * (self._q_hi - self._q_lo)

    def apply_actions(self) -> None:
        linear, angular = self._twist[:, :3], self._twist[:, 3:]
        self._site_pos += linear * self._dt
        bounds_lo = self._origins + linear.new_tensor((0.05, -0.45, 0.06))
        bounds_hi = self._origins + linear.new_tensor((1.05, 0.45, 0.75))
        self._site_pos[:] = torch.minimum(torch.maximum(self._site_pos, bounds_lo), bounds_hi)

        angle = torch.linalg.norm(angular, dim=-1, keepdim=True) * self._dt
        axis = angular / torch.linalg.norm(angular, dim=-1, keepdim=True).clamp_min(1e-8)
        dq = torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)
        self._site_quat[:] = quat_mul(dq, self._site_quat)
        self._site_quat[:] /= torch.linalg.norm(
            self._site_quat, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        offset_w = quat_rotate(self._site_quat, self._site_offset)
        root_pose = torch.cat([self._site_pos - offset_w, self._site_quat], dim=-1)
        root_linear = linear - torch.linalg.cross(angular, offset_w)
        self._entity.write_root_link_pose_to_sim(root_pose)
        self._entity.write_root_link_velocity_to_sim(torch.cat([root_linear, angular], dim=-1))

        err = self.hand_target - self._q_cmd
        speed = self.cfg.hand_v_max * (err.abs() / self.cfg.hand_slowdown_distance).clamp(max=1.0)
        self._q_cmd += torch.sign(err) * speed * self._dt
        if HAND_EXPLICIT_PD:
            data = self._entity.data
            q = data.joint_pos[:, self._hand_joint_ids]
            qd = data.joint_vel[:, self._hand_joint_ids]
            effort = (self._hand_kp * (self._q_cmd - q) - self._hand_kd * qd).clamp(
                -HAND_EFFORT_LIMIT, HAND_EFFORT_LIMIT
            )
            self._entity.set_joint_effort_target(effort, joint_ids=self._hand_joint_ids)
        else:
            self._entity.set_joint_position_target(self._q_cmd, joint_ids=self._hand_joint_ids)
