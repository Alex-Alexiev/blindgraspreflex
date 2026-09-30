"""Canonical reflexive reach/lift state machine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def euler_xyz_to_quaternion(rpy: torch.Tensor) -> torch.Tensor:
    half = 0.5 * rpy
    cosine, sine = torch.cos(half), torch.sin(half)
    zeros = torch.zeros_like(cosine[..., :1])
    qx = torch.cat((cosine[..., 0:1], sine[..., 0:1], zeros, zeros), dim=-1)
    qy = torch.cat((cosine[..., 1:2], zeros, sine[..., 1:2], zeros), dim=-1)
    qz = torch.cat((cosine[..., 2:3], zeros, zeros, sine[..., 2:3]), dim=-1)
    return quaternion_multiply(qz, quaternion_multiply(qy, qx))


@dataclass(frozen=True)
class ReflexPhaseConfig:
    """Robot-independent constants used by the canonical policy."""

    pregrasp_offset: tuple[float, float, float] = (0.03, 0.015, 0.065)
    reach_rpy_degrees: tuple[float, float, float] = (0.0, 11.0, 0.0)
    reach_position_randomization: tuple[float, float, float] = (0.05, 0.05, 0.05)
    reach_rpy_randomization_degrees: tuple[float, float, float] = (10.0, 10.0, 10.0)
    score_low: float = 0.45
    score_high: float = 0.55
    engage_radius: float = 0.10
    release_radius: float = 0.15
    reach_object_radius: float = 0.15
    max_setpoint_radius: float = 0.15
    minimum_setpoint_z: float = -0.03


class ReflexPhaseController:
    """Close the arm loop around the hand policy's scalar grasp score.

    The arm approaches the perceived object pose. A high score latches the fixed lift target; a
    low score returns to reach for another attempt. A separate proximity latch prevents the hand
    policy from closing before the palm reaches the object.
    """

    REACH = 0
    LIFT = 1

    def __init__(
        self,
        num_envs: int,
        config: ReflexPhaseConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or ReflexPhaseConfig()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.hand_engaged = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.lift_position = torch.zeros(num_envs, 3, device=self.device)
        self.lift_orientation = torch.zeros(num_envs, 4, device=self.device)
        self.lift_orientation[:, 0] = 1.0
        self.position_noise = torch.zeros(num_envs, 3, device=self.device)
        self.orientation_noise = torch.zeros(num_envs, 4, device=self.device)
        self.orientation_noise[:, 0] = 1.0

        self._offset = torch.tensor(
            self.config.pregrasp_offset, dtype=torch.float32, device=self.device
        )
        fixed_rpy = torch.tensor(
            [[math.radians(value) for value in self.config.reach_rpy_degrees]],
            dtype=torch.float32,
            device=self.device,
        )
        self._fixed_orientation = euler_xyz_to_quaternion(fixed_rpy)
        self._position_magnitude = torch.tensor(
            self.config.reach_position_randomization,
            dtype=torch.float32,
            device=self.device,
        )
        self._rpy_magnitude = torch.tensor(
            [math.radians(value) for value in self.config.reach_rpy_randomization_degrees],
            dtype=torch.float32,
            device=self.device,
        )
        randomization_span = float(torch.linalg.norm(self._position_magnitude))
        self._engage_radius = self.config.engage_radius + randomization_span
        self._release_radius = self.config.release_radius + randomization_span
        self._reach_object_radius = self.config.reach_object_radius + randomization_span

    def reset(
        self,
        env_ids: torch.Tensor,
        lift_position: torch.Tensor,
        lift_orientation: torch.Tensor,
    ) -> None:
        count = int(env_ids.numel())
        self.phase[env_ids] = self.REACH
        self.hand_engaged[env_ids] = False
        self.lift_position[env_ids] = lift_position
        self.lift_orientation[env_ids] = lift_orientation
        position_uniform = 2.0 * torch.rand(count, 3, device=self.device) - 1.0
        rpy_uniform = 2.0 * torch.rand(count, 3, device=self.device) - 1.0
        self.position_noise[env_ids] = position_uniform * self._position_magnitude
        self.orientation_noise[env_ids] = euler_xyz_to_quaternion(
            rpy_uniform * self._rpy_magnitude
        )

    def update(
        self,
        grasp_score: torch.Tensor,
        object_position: torch.Tensor,
        palm_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = grasp_score.clamp(0.0, 1.0).reshape(-1)
        palm_distance = torch.linalg.norm(palm_position - object_position, dim=-1)
        self.hand_engaged = torch.where(
            palm_distance < self._engage_radius,
            torch.ones_like(self.hand_engaged),
            torch.where(
                palm_distance > self._release_radius,
                torch.zeros_like(self.hand_engaged),
                self.hand_engaged,
            ),
        )
        reaching = self.phase == self.REACH
        self.phase = torch.where(
            reaching & (score >= self.config.score_high),
            torch.full_like(self.phase, self.LIFT),
            torch.where(
                (~reaching) & (score <= self.config.score_low),
                torch.full_like(self.phase, self.REACH),
                self.phase,
            ),
        )
        reaching = self.phase == self.REACH
        reach_position = object_position + self._offset + self.position_noise
        object_delta = reach_position - object_position
        object_distance = torch.linalg.norm(object_delta, dim=-1, keepdim=True)
        object_scale = (self._reach_object_radius / object_distance.clamp_min(1e-9)).clamp_max(1.0)
        reach_position = object_position + object_delta * object_scale
        reach_orientation = quaternion_multiply(
            quaternion_multiply(self.lift_orientation, self._fixed_orientation),
            self.orientation_noise,
        )
        target_position = torch.where(
            reaching.unsqueeze(-1), reach_position, self.lift_position
        )
        target_orientation = torch.where(
            reaching.unsqueeze(-1), reach_orientation, self.lift_orientation
        )
        target_delta = target_position - palm_position
        target_distance = torch.linalg.norm(target_delta, dim=-1, keepdim=True)
        target_scale = (
            self.config.max_setpoint_radius / target_distance.clamp_min(1e-9)
        ).clamp_max(1.0)
        target_position = palm_position + target_delta * target_scale
        target_position[:, 2].clamp_(min=self.config.minimum_setpoint_z)
        return target_position, target_orientation

