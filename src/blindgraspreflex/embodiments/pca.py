"""Runtime loader for fixed hand PCA vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HandPCA:
    components: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def latent_size(self) -> int:
        return int(self.components.shape[0])

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return latent @ self.components + self.mean

    def reconstruct(self, joint_position: np.ndarray) -> np.ndarray:
        latent = (joint_position - self.mean) @ self.components.T
        return self.decode(latent)


def load_hand_pca(directory: Path, joint_names: tuple[str, ...]) -> HandPCA:
    manifest = json.loads((directory / "manifest.json").read_text())
    model = np.load(directory / "pca_model.npz")
    stats = np.load(directory / "latent_stats.npz")
    expected = tuple(manifest["robot_joint_names"])
    if expected != joint_names:
        raise ValueError("PCA joint order does not match the embodiment hand joint order")
    components = np.asarray(model["components"], dtype=np.float32)
    mean = np.asarray(model["mean"], dtype=np.float32)
    lower = np.asarray(stats["p01"], dtype=np.float32)
    upper = np.asarray(stats["p99"], dtype=np.float32)
    if components.shape[1] != len(joint_names):
        raise ValueError("PCA component width does not match the embodiment hand DOF")
    return HandPCA(components=components, mean=mean, lower=lower, upper=upper)

