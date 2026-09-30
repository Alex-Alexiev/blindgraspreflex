"""Built-in metadata for the Flexiv Rizon 4 with Robotis 5F hand."""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

from blindgraspreflex.embodiments.spec import ArmSpec, EmbodimentSpec, HandSpec

FINGERS = ("1_thumb", "2_index", "3_middle", "4_ring", "5_little")
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINT_NAMES = tuple(
    f"a{8 + 4 * finger_index + joint_index:02d}_finger_r_joint_{finger}{joint_index + 1}"
    for finger_index, finger in enumerate(FINGERS)
    for joint_index in range(4)
)

ARM_HOME = (-0.4363323, -0.6981317, 0.6108652, 1.7453293, 0.8726646, 0.5235988, 1.0471976)
HAND_OPEN = (0.0,) * 20
HAND_REACH = tuple(
    value * 3.141592653589793 / 180.0
    for value in (-8, 0, 0, 0, -28, 0, 0, 0, -5, 0, 0, 0, 5, 0, 0, 0, 28, 0, 0, 0)
)
HAND_KP = (6.0, 5.0, 3.0, 2.0) * 5
HAND_KD = (0.25, 0.20, 0.12, 0.08) * 5
HAND_EFFORT = (1.0,) * 20

CONTACT_BODIES = tuple(
    [f"finger_r_link_{finger}{level}" for finger in FINGERS for level in (2, 3, 4)] + ["palm"]
)
FINGERTIP_BODIES = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip")


def make_embodiment() -> EmbodimentSpec:
    """Return the built-in embodiment.

    Robot geometry is intentionally an external input until its redistribution license is
    recorded. This keeps the public training package independent of the hardware/deployment repo.
    """
    default_mjcf = Path("assets/robotis_5f/Rizon4_hand_robot.xml")
    mjcf_path = Path(os.environ.get("BLINDGRASPREFLEX_ROBOTIS_5F_MJCF", default_mjcf)).expanduser()
    pca_directory = Path(str(files(__package__).joinpath("pca")))
    return EmbodimentSpec(
        key="robotis_5f",
        mjcf_path=mjcf_path,
        arm=ArmSpec(
            joint_names=ARM_JOINT_NAMES,
            home_position=ARM_HOME,
            palm_site="grasp_site",
        ),
        hand=HandSpec(
            joint_names=HAND_JOINT_NAMES,
            open_position=HAND_OPEN,
            reach_position=HAND_REACH,
            contact_bodies=CONTACT_BODIES,
            fingertip_bodies=FINGERTIP_BODIES,
            kp=HAND_KP,
            kd=HAND_KD,
            effort_limit=HAND_EFFORT,
            pca_directory=pca_directory,
        ),
    )


__all__ = ["make_embodiment"]
