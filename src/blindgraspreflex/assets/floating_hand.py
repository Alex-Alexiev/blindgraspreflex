"""Arm-free RH-5 hand for learning a transferable palm task-space policy."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from blindgraspreflex.assets.rizon4_hand import (
    _CONTACT_EXCLUDES,
    COLLISION,
    GRASP_SITE_NAME,
    HAND_ACTUATORS,
    HAND_JOINT_NAMES,
    HAND_Q_CLOSE,
    HAND_Q_OPEN,
    _name_geoms,
)
from blindgraspreflex.assets.rizon4_hand_v2 import (
    RIZON4_HAND_V2_XML,
    _override_hand_frictionloss,
)

PALM_FACE_SITE_NAME = "palm_face"
PALM_FACE_POS = (0.019, -0.004, 0.115)
"""Point on the palm face at the knuckles, where the fingers start, in the palm frame. The face is
the palm's +X side (the fingers close toward it); +Z runs from the wrist (z = 0) to the finger
bases (z ~ 0.09-0.10) and the knuckle edge of the palm (z ~ 0.12)."""

START_QUAT = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
"""Start orientation: the palm face (+X) points straight down, the fingers (+Z) along world +x."""
START_PALM_FACE = (0.55, 0.15, 0.27)
"""Start position of the palm-face point: 27 cm straight above the target's spawn point
(mjlab_task.SPAWN_XY; the target rests with its centre at about z = 0)."""
HAND_CURL = 0.25
"""How far the start pose bends each flexion joint, as a fraction of HAND_Q_OPEN -> HAND_Q_CLOSE."""
# Flexion joints: phalanges 2-4 of the four fingers and the thumb's last two. The *1 joints spread
# the fingers and thumb1/thumb2 rotate the thumb; those stay open.
_FLEXION = tuple(
    n for n in HAND_JOINT_NAMES if n[-1] in "34" or (n[-1] == "2" and "thumb" not in n)
)
HAND_Q_START = {
    n: HAND_Q_OPEN[n] + (HAND_CURL * (HAND_Q_CLOSE[n] - HAND_Q_OPEN[n]) if n in _FLEXION else 0.0)
    for n in HAND_JOINT_NAMES
}
"""Start joint pose: open, with the fingers curled in a little."""


def get_spec() -> mujoco.MjSpec:
    """Keep the original palm and finger subtree, rooted at a free joint."""
    root = ET.parse(RIZON4_HAND_V2_XML).getroot()
    world = root.find("worldbody")
    assert world is not None
    palm = next(body for body in world.iter("body") if body.get("name") == "palm")
    palm.set("pos", "0 0 0")
    palm.attrib.pop("quat", None)
    palm.insert(0, ET.Element("freejoint", name="palm_free"))
    world.clear()
    world.append(palm)

    asset = root.find("asset")
    assert asset is not None
    used_meshes = {geom.get("mesh") for geom in palm.iter("geom")}
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") not in used_meshes:
            asset.remove(mesh)
        elif path := mesh.get("file"):
            mesh.set("file", str(RIZON4_HAND_V2_XML.parent / path))
    root.set("model", "floating_rh5_hand")

    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    _name_geoms(spec)
    spec.body("palm").add_site(name=GRASP_SITE_NAME, pos=(0.0, 0.0, 0.1))
    spec.body("palm").add_site(name=PALM_FACE_SITE_NAME, pos=PALM_FACE_POS)
    _override_hand_frictionloss(spec)
    for body1, body2 in _CONTACT_EXCLUDES:
        spec.add_exclude(name=f"exclude_{body1}_{body2}", bodyname1=body1, bodyname2=body2)
    return spec


def _start_root_pos() -> tuple[float, float, float]:
    """Root position that puts the palm-face point at START_PALM_FACE in START_QUAT."""
    offset = np.zeros(3)
    mujoco.mju_rotVecQuat(offset, np.asarray(PALM_FACE_POS), np.asarray(START_QUAT))
    return tuple(float(v) for v in np.asarray(START_PALM_FACE) - offset)


def get_floating_hand_cfg() -> EntityCfg:
    """The palm face starts above the target, facing it, with the fingers curled in a little."""
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=_start_root_pos(),
            rot=START_QUAT,
            joint_pos=HAND_Q_START,
            joint_vel={".*": 0.0},
        ),
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=HAND_ACTUATORS,
            soft_joint_pos_limit_factor=1.0,
        ),
    )
