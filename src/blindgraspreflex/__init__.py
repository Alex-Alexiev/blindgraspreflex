"""Blind Grasp Reflex training package."""

import os
from pathlib import Path

from blindgraspreflex.config import CANONICAL_METHOD, MethodConfig

BLINDGRASPREFLEX_SRC_PATH = Path(__file__).parent
BLINDGRASPREFLEX_ROOT_PATH = Path(
    os.environ.get("BLINDGRASPREFLEX_ROOT", BLINDGRASPREFLEX_SRC_PATH.parents[1])
)
MANIPULATOR_SOFTWARE_PATH = Path(
    os.environ.get(
        "BLINDGRASPREFLEX_MANIPULATOR_SOFTWARE",
        BLINDGRASPREFLEX_ROOT_PATH / "third_party" / "manipulator-software",
    )
)

__all__ = [
    "BLINDGRASPREFLEX_ROOT_PATH",
    "BLINDGRASPREFLEX_SRC_PATH",
    "CANONICAL_METHOD",
    "MANIPULATOR_SOFTWARE_PATH",
    "MethodConfig",
]
