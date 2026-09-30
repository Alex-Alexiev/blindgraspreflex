"""Canonical proprioceptive student observation."""

from __future__ import annotations

import torch


class QCommandHistory:
    """Frame stack of `[q, q_cmd - q]`, oldest frame first.

    `q_cmd` is the command actually presented to the joint servo after velocity limiting. It is
    deliberately not the policy's more distant hand target: the servo error is proportional to
    the effort applied by a position actuator and therefore exposes blocked-finger information.
    """

    def __init__(
        self,
        num_envs: int,
        hand_dof: int,
        history_length: int = 5,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_envs <= 0 or hand_dof <= 0 or history_length <= 0:
            raise ValueError("num_envs, hand_dof, and history_length must be positive")
        self.num_envs = num_envs
        self.hand_dof = hand_dof
        self.history_length = history_length
        self.device = torch.device(device)
        self._history = torch.zeros(
            num_envs,
            history_length,
            2 * hand_dof,
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def observation_size(self) -> int:
        return self.history_length * 2 * self.hand_dof

    def reset(
        self,
        joint_position: torch.Tensor,
        joint_command: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        frame = self._frame(joint_position, joint_command)
        if env_ids is None:
            if frame.shape[0] != self.num_envs:
                raise ValueError("full reset requires one frame per environment")
            self._history[:] = frame.unsqueeze(1)
            return
        self._history[env_ids] = frame.unsqueeze(1)

    def update(self, joint_position: torch.Tensor, joint_command: torch.Tensor) -> torch.Tensor:
        frame = self._frame(joint_position, joint_command)
        if frame.shape[0] != self.num_envs:
            raise ValueError("update requires one frame per environment")
        self._history = torch.roll(self._history, shifts=-1, dims=1)
        self._history[:, -1] = frame
        return self.observation

    @property
    def observation(self) -> torch.Tensor:
        return self._history.reshape(self.num_envs, self.observation_size)

    def _frame(self, joint_position: torch.Tensor, joint_command: torch.Tensor) -> torch.Tensor:
        expected = (joint_position.shape[0], self.hand_dof)
        if joint_position.shape != expected or joint_command.shape != expected:
            raise ValueError(f"expected joint tensors shaped (N, {self.hand_dof})")
        return torch.cat((joint_position, joint_command - joint_position), dim=-1)

