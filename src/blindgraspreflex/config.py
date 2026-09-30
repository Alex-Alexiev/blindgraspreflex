"""Typed configuration for the canonical q_cmd training recipe."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhysicsConfig:
    timestep: float = 0.004
    decimation: int = 21

    @property
    def control_timestep(self) -> float:
        return self.timestep * self.decimation


@dataclass(frozen=True)
class DomainRandomizationConfig:
    object_mass_range: tuple[float, float] = (0.03, 0.20)
    object_scale_range: tuple[float, float] = (0.7, 1.0)
    reach_position: tuple[float, float, float] = (0.05, 0.05, 0.05)
    reach_rpy_degrees: tuple[float, float, float] = (10.0, 10.0, 10.0)
    disturbance_linear_velocity: float = 0.35
    disturbance_angular_velocity: float = 0.9
    hand_velocity_range: tuple[float, float] = (0.6, 2.0)
    hand_slowdown_range: tuple[float, float] = (0.15, 0.45)
    hand_kp_scale_range: tuple[float, float] = (0.1, 1.5)
    hand_kd_scale_range: tuple[float, float] = (0.3, 1.5)


@dataclass(frozen=True)
class TeacherConfig:
    architecture: str = "gru"
    hidden_size: int = 256
    iterations: int = 3500
    privileged_contacts: bool = True


@dataclass(frozen=True)
class StudentConfig:
    architecture: str = "mlp"
    history_length: int = 5
    observation: str = "q_cmd"
    iterations: int = 3550


@dataclass(frozen=True)
class MethodConfig:
    """Settings which define the method rather than a particular robot."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    num_envs: int = 6144
    episode_length_seconds: float = 10.0
    hand_pca_projection: float = 0.5
    lift_height: float = 0.3
    hold_reward_weight: float = 150.0
    use_arm_velocity_controller: bool = True
    use_hand_gate: bool = True
    learned_reach_hand_pose: bool = False

    def student_observation_size(self, hand_dof: int) -> int:
        """Five histories of `[q, q_cmd - q]` for the selected hand."""
        if hand_dof <= 0:
            raise ValueError("hand_dof must be positive")
        return self.student.history_length * 2 * hand_dof


CANONICAL_METHOD = MethodConfig()
