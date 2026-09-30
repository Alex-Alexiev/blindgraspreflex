"""Robot/hand contract consumed by the training environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArmSpec:
    joint_names: tuple[str, ...]
    home_position: tuple[float, ...]
    palm_site: str
    max_joint_velocity: float = 0.5
    slowdown_distance: float = 0.35

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.joint_names:
            errors.append("arm must declare at least one joint")
        if len(self.home_position) != len(self.joint_names):
            errors.append("arm home_position length must match joint_names")
        if not self.palm_site:
            errors.append("arm palm_site cannot be empty")
        return errors


@dataclass(frozen=True)
class HandSpec:
    joint_names: tuple[str, ...]
    open_position: tuple[float, ...]
    reach_position: tuple[float, ...]
    contact_bodies: tuple[str, ...]
    fingertip_bodies: tuple[str, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort_limit: tuple[float, ...]
    pca_directory: Path | None = None

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.joint_names:
            errors.append("hand must declare at least one joint")
        for name, values in (
            ("open_position", self.open_position),
            ("reach_position", self.reach_position),
            ("kp", self.kp),
            ("kd", self.kd),
            ("effort_limit", self.effort_limit),
        ):
            if len(values) != self.dof:
                errors.append(f"hand {name} length must match joint_names")
        if not self.contact_bodies:
            errors.append("hand must declare contact_bodies for privileged teacher observations")
        if self.pca_directory is not None:
            for filename in ("manifest.json", "pca_model.npz", "latent_stats.npz"):
                if not (self.pca_directory / filename).is_file():
                    errors.append(f"hand PCA is missing {filename}")
        return errors


@dataclass(frozen=True)
class EmbodimentSpec:
    """A complete arm-hand assembly with names resolved in one MuJoCo model."""

    key: str
    mjcf_path: Path
    arm: ArmSpec
    hand: HandSpec

    def validate(self, *, require_mjcf: bool = True) -> list[str]:
        errors = self.arm.validate() + self.hand.validate()
        names = self.arm.joint_names + self.hand.joint_names
        if len(set(names)) != len(names):
            errors.append("arm and hand joint names must be unique")
        if require_mjcf and not self.mjcf_path.is_file():
            errors.append(f"MJCF does not exist: {self.mjcf_path}")
        return errors

    def require_valid(self, *, require_mjcf: bool = True) -> None:
        errors = self.validate(require_mjcf=require_mjcf)
        if errors:
            raise ValueError("Invalid embodiment:\n- " + "\n- ".join(errors))

