"""PCA latent -> hand joint pose decoder, ported from the IsaacLab hand action.

The policy commands a low-dimensional latent (z_dim=5) instead of all 20 finger joints; the
PCA basis expands it back to a full joint pose lying on the manifold of real grasps the model
was fit to. That turns a 20-DoF exploration problem into a 5-DoF one over coordinated hand
shapes, which is the whole point of using it.

Model provenance: fit in Dexterous-Hand-Retargeting on 300 retargeted grasp poses, bundled
here as ``hand_pca_model/{pca_model,latent_stats}.npz``. The manifest's ``robot_joint_names``
match :data:`blindgraspreflex.assets.HAND_JOINT_NAMES` in order, so the decoded 20-vector maps
straight onto the hand joints with no reordering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from blindgraspreflex import BLINDGRASPREFLEX_SRC_PATH
from blindgraspreflex.assets.rizon4_hand import HAND_JOINT_NAMES

_PCA_DIR = BLINDGRASPREFLEX_SRC_PATH / "embodiments" / "robotis_5f" / "pca"

__all__ = ["HandPCA", "load_hand_pca"]


@dataclass
class HandPCA:
    """PCA basis plus percentile latent bounds, as torch tensors on a chosen device.

    ``components`` is (z_dim, 20), ``mean`` is (20,); ``p01``/``p99`` are the per-latent 1st and
    99th percentiles from the fitting set, used to map a squashed [0, 1] action onto the range
    of latents the model actually saw (IsaacLab's ``latent_input_mode="percentile"``).
    """

    components: torch.Tensor  # (z_dim, 20)
    mean: torch.Tensor  # (20,)
    p01: torch.Tensor  # (z_dim,)
    p99: torch.Tensor  # (z_dim,)
    z_dim: int

    def latent_from_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """Map alpha in [0, 1] per latent dim to a latent code in [p01, p99]. alpha: (N, z_dim)."""
        return self.p01.unsqueeze(0) + alpha * (self.p99 - self.p01).unsqueeze(0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents (N, z_dim) to hand joint positions (N, 20): ``z @ components + mean``."""
        return z @ self.components + self.mean.unsqueeze(0)

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        """Project hand poses (N, 20) onto the latent basis: ``(q - mean) @ components^T``."""
        return (q - self.mean.unsqueeze(0)) @ self.components.t()

    def reconstruct(self, q: torch.Tensor) -> torch.Tensor:
        """Encode then decode -- the closest point to ``q`` on the PCA manifold. (N, 20)."""
        return self.decode(self.encode(q))


def load_hand_pca(device: torch.device | str = "cpu", pca_dir=None) -> HandPCA:
    """Load the bundled PCA model, verifying its joint order matches the hand's."""
    pca_dir = _PCA_DIR if pca_dir is None else pca_dir
    manifest = json.loads((pca_dir / "manifest.json").read_text())
    if tuple(manifest["robot_joint_names"]) != HAND_JOINT_NAMES:
        raise ValueError(
            "PCA model joint order does not match HAND_JOINT_NAMES; decoded poses would be "
            "scrambled. Re-fit the PCA or remap columns."
        )
    model = np.load(pca_dir / "pca_model.npz")
    stats = np.load(pca_dir / "latent_stats.npz")

    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)

    return HandPCA(
        components=t(model["components"]),
        mean=t(model["mean"]),
        p01=t(stats["p01"]),
        p99=t(stats["p99"]),
        z_dim=int(model["z_dim"]),
    )
