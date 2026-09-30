"""Robot entity configs."""

from blindgraspreflex.assets.rizon4_hand import (
    ARM_JOINT_NAMES,
    ARM_Q_HOME,
    FINGERTIP_BODY_NAMES,
    FINGERTIP_SITE_NAMES,
    GRASP_SITE_NAME,
    HAND_ACTION_SCALE,
    HAND_JOINT_NAMES,
    HAND_Q_CLOSE,
    HAND_Q_GRASP_OPEN,
    HAND_Q_OPEN,
    RIZON4_HAND_XML,
    get_rizon4_hand_robot_cfg,
)
from blindgraspreflex.assets.rizon4_hand_v2 import (
    RIZON4_HAND_V2_XML,
    get_rizon4_hand_v2_robot_cfg,
)

__all__ = [
    "ARM_JOINT_NAMES",
    "ARM_Q_HOME",
    "FINGERTIP_BODY_NAMES",
    "FINGERTIP_SITE_NAMES",
    "GRASP_SITE_NAME",
    "HAND_ACTION_SCALE",
    "HAND_JOINT_NAMES",
    "HAND_Q_CLOSE",
    "HAND_Q_GRASP_OPEN",
    "HAND_Q_OPEN",
    "RIZON4_HAND_XML",
    "get_rizon4_hand_robot_cfg",
    "RIZON4_HAND_V2_XML",
    "get_rizon4_hand_v2_robot_cfg",
]
