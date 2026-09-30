"""Projection of hand commands toward a fixed grasp PCA manifold."""

from __future__ import annotations

import torch

from blindgraspreflex.embodiments.pca import HandPCA


class HandPCAProjector:
    """Blend a joint command with its nearest point on a linear PCA manifold."""

    def __init__(
        self,
        pca: HandPCA,
        alpha: float = 0.5,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = float(alpha)
        self.components = torch.as_tensor(pca.components, dtype=torch.float32, device=device)
        self.mean = torch.as_tensor(pca.mean, dtype=torch.float32, device=device)

    @property
    def hand_dof(self) -> int:
        return int(self.components.shape[1])

    def __call__(self, command: torch.Tensor) -> torch.Tensor:
        if command.shape[-1] != self.hand_dof:
            raise ValueError(f"command has {command.shape[-1]} joints; PCA expects {self.hand_dof}")
        latent = (command - self.mean) @ self.components.T
        projected = latent @ self.components + self.mean
        return torch.lerp(command, projected, self.alpha)
