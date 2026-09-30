"""mjlab wrapper used by both training stages."""

from __future__ import annotations

import torch
from mjlab.rl import RslRlVecEnvWrapper


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Keep one unstable physics world from poisoning an entire vectorized update."""

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        observations = {
            key: torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            for key, value in observations.items()
        }
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        return observations, rewards, dones, extras
