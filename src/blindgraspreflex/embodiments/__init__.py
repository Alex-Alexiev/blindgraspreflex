"""Embodiment plugin API."""

from blindgraspreflex.embodiments.registry import available, load
from blindgraspreflex.embodiments.spec import ArmSpec, EmbodimentSpec, HandSpec

__all__ = ["ArmSpec", "EmbodimentSpec", "HandSpec", "available", "load"]

