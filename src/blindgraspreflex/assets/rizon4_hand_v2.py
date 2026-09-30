"""Flexiv Rizon 4 + Robotis RH-5 five-finger hand -- **v2 (new) hand geometry**.

Parallel to :mod:`blindgraspreflex.assets.rizon4_hand`, for the updated hand MJCF
(``Rizon4_new_hand_robot.xml``: new link geometry, joint origins, and joint limits). The
original module is deliberately left untouched so the old hand stays fully runnable; this
module reuses everything that is genuinely shared between the two hands and only redefines
what the new geometry changes.

Differences from the old hand
-----------------------------
* MJCF is ``Rizon4_new_hand_robot.xml``.
* The new MJCF has **no separate ``*_tip`` frames**. The distal phalanx (``...link_N_*4``)
  *is* the fingertip and already carries a site at the tip, so ``FINGERTIP_BODY_NAMES`` /
  ``FINGERTIP_SITE_NAMES`` point at those bodies/sites and :func:`get_spec` only adds the
  palm ``grasp_site`` (the fingertip sites already exist).

Everything else -- joint names (a08..a27, joint1..7), actuators + gains, collision
surfaces, task poses, arm home -- is identical across the two hands and is imported from
:mod:`blindgraspreflex.assets.rizon4_hand` so the two stay in sync.
"""

from __future__ import annotations

import json
import os

import mujoco
from mjlab.entity import EntityCfg

from blindgraspreflex import BLINDGRASPREFLEX_ROOT_PATH, MANIPULATOR_SOFTWARE_PATH

# Shared, hand-invariant pieces (joint names, actuators, gains, collision, poses, home).
from blindgraspreflex.assets.rizon4_hand import (
    _CONTACT_EXCLUDES,
    ARM_JOINT_NAMES,
    ARM_Q_HOME,
    ARTICULATION,
    COLLISION,
    FINGERS,
    GRASP_SITE_NAME,
    HAND_ACTION_SCALE,
    HAND_JOINT_NAMES,
    HAND_Q_CLOSE,
    HAND_Q_GRASP_OPEN,
    HAND_Q_OPEN,
    HOME_KEYFRAME,
    _name_geoms,
)

##
# MJCF (new hand).
##

RIZON4_HAND_V2_XML = (
    MANIPULATOR_SOFTWARE_PATH / "assets" / "scene" / "flexiv_arm" / "Rizon4_new_hand_robot.xml"
)
assert RIZON4_HAND_V2_XML.exists(), (
    f"New-hand MJCF not found at {RIZON4_HAND_V2_XML}. It lives in the manipulator-software "
    "submodule (branch reflexive-grasp-deploy); run: git submodule update --init --recursive"
)

##
# Fingertips.
##

# The new MJCF folds the fingertip into the distal phalanx body (``...link_N_*4``) and
# already declares a site there at the tip, so the fingertip body and site names coincide.
FINGERTIP_BODY_NAMES = (
    "finger_r_link_1_thumb4",
    "finger_r_link_2_index4",
    "finger_r_link_3_middle4",
    "finger_r_link_4_ring4",
    "finger_r_link_5_little4",
)
FINGERTIP_SITE_NAMES = FINGERTIP_BODY_NAMES  # reuse the sites the MJCF already carries

##
# Spec editing.
##


def _add_sites(spec: mujoco.MjSpec) -> None:
    """Add only the palm ``grasp_site``.

    Unlike the old hand, the new MJCF already carries fingertip sites on the distal
    phalanges (see ``FINGERTIP_SITE_NAMES``), so they are not re-added here. ``grasp_site``
    reproduces the palm grasp point -- 10 cm along the palm's local +z toward the fingers --
    matching the old hand and the deploy MJCF.
    """
    spec.body("palm").add_site(name=GRASP_SITE_NAME, pos=(0.0, 0.0, 0.1))


def _override_hand_frictionloss(spec: mujoco.MjSpec) -> None:
    """Optionally force the hand joints' frictionloss, for sim-to-sim experiments.

    The MJCF ships 0.05 Nm on every hand joint, which is a deadband of frictionloss/kp -- 0.008 rad
    at the proximal gains, 0.025 rad at the distal ones. REFLEXIVE_HAND_FRICTIONLOSS=0 removes it.
    The MS stack reads the SAME MJCF and has a matching override keyed to the same variable, so the
    two stacks can be changed together and stay comparable.
    """
    raw = os.environ.get("REFLEXIVE_HAND_FRICTIONLOSS", "")
    if raw == "":
        return
    val = float(raw)
    names = set(HAND_JOINT_NAMES)
    for j in spec.joints:
        if j.name in names:
            j.frictionloss = val


def get_spec() -> mujoco.MjSpec:
    """Load the new-hand MJCF and apply the edits mjlab needs. Returns a fresh spec."""
    spec = mujoco.MjSpec.from_file(str(RIZON4_HAND_V2_XML))

    _name_geoms(spec)
    _add_sites(spec)
    _override_hand_frictionloss(spec)

    for body1, body2 in _CONTACT_EXCLUDES:
        spec.add_exclude(name=f"exclude_{body1}_{body2}", bodyname1=body1, bodyname2=body2)

    return spec


##
# Final config.
##


def get_rizon4_hand_v2_robot_cfg() -> EntityCfg:
    """Rizon 4 + RH-5 v2 hand: 7 torque arm joints, 20 position hand joints (new geometry)."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


##
# Default / reach pose.
##

# The hand's reach/idle default pose -- used when the grasp policy is NOT driving the fingers.
# This is the reference pose tuned in the retargeting viser and saved to
# ``default_reference_pose.json`` (indexed in HAND_JOINT_NAMES order, a08..a27), i.e. the same
# pose the retargeting optimiser regularises toward, so the sim's reach pose and the retargeting
# agree.
_DEFAULT_REFERENCE_POSE_JSON = (
    BLINDGRASPREFLEX_ROOT_PATH
    / "third_party"
    / "hand-retargeting"
    / "assets"
    / "robots"
    / "robotis_5f_hand_v2"
    / "default_reference_pose.json"
)


def get_default_reference_pose() -> dict[str, float]:
    """Load the retargeting-tuned default/reach pose as ``{joint_name: angle}``.

    Falls back to the fully-open pose (``HAND_Q_OPEN``) if the retargeting submodule / JSON is
    unavailable, so the sim still runs without it.
    """
    try:
        vals = json.loads(_DEFAULT_REFERENCE_POSE_JSON.read_text())["reference_pose"]
        return {n: float(v) for n, v in zip(HAND_JOINT_NAMES, vals, strict=True)}
    except Exception:
        return dict(HAND_Q_OPEN)


# Materialised at import for convenience (safe: falls back to HAND_Q_OPEN).
HAND_Q_DEFAULT = get_default_reference_pose()

##
# Fingertip contact-sensing spheres (hardware pad model).
##

# On hardware each fingertip senses contact only on a small pad, not across the whole link. A
# sim contact therefore counts as a fingertip touch only if its contact point lies inside a
# per-fingertip sphere: centre = fingertip link frame + FINGERTIP_SENSE_OFFSET (link-frame
# metres), radius = FINGERTIP_SENSE_RADIUS. Both training (the fingertip contact observation) and
# the grasp playground read these, so they stay in sync. Order matches ``FINGERTIP_BODY_NAMES``
# (thumb, index, middle, ring, little). Defaults are the fingertip sites the MJCF carries; tune
# in the playground's "Fingertip sensing spheres" and paste the values back here.
FINGERTIP_SENSE_OFFSET = (
    (0.0, 0.04, 0.009),  # thumb4
    (0.01, 0.0, 0.02425),  # index4
    (0.01, 0.0, 0.02425),  # middle4
    (0.01, 0.0, 0.02425),  # ring4
    (0.01, 0.0, 0.02425),  # little4
)
FINGERTIP_SENSE_RADIUS = (0.012, 0.012, 0.012, 0.012, 0.012)


__all__ = [
    "RIZON4_HAND_V2_XML",
    "FINGERTIP_BODY_NAMES",
    "FINGERTIP_SITE_NAMES",
    "GRASP_SITE_NAME",
    "get_rizon4_hand_v2_robot_cfg",
    "get_default_reference_pose",
    "HAND_Q_DEFAULT",
    "FINGERTIP_SENSE_OFFSET",
    "FINGERTIP_SENSE_RADIUS",
    # re-exported shared constants, so a v2 consumer can import from one place
    "ARM_JOINT_NAMES",
    "ARM_Q_HOME",
    "HAND_JOINT_NAMES",
    "HAND_Q_OPEN",
    "HAND_Q_GRASP_OPEN",
    "HAND_Q_CLOSE",
    "HAND_ACTION_SCALE",
    "FINGERS",
]
