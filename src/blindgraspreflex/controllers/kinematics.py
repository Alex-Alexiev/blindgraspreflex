"""Batched analytic Jacobians in torch, for use inside mjlab action terms.

mjlab's ``EntityData`` exposes body and site poses but no Jacobian, and mujoco-warp's ``jac``
is a device-side warp kernel with no host-callable binding. A task-space controller therefore
has to build its own. For a serial chain of revolute joints that is cheap and exact: each
column is fixed by the joint's world axis and the lever arm from its anchor to the controlled
point, so nothing is approximated here -- this reproduces ``mj_jacSite`` to float precision.

Only hinge joints are handled, which is all the Rizon 4 arm has. Slide or ball joints in the
controlled chain would need extra column types.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

__all__ = ["ArmJacobian"]


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by wxyz ``quat``. Both (..., 3)/(..., 4), broadcast over leading dims."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


class ArmJacobian:
    """Precomputes the chain layout once, then evaluates batched Jacobians per step.

    The static description (which body each joint sits on, its axis and anchor in that body's
    frame) is read from the compiled model at construction; only poses change per step.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        site_name: str,
        device: torch.device | str = "cpu",
        prefix: str = "",
        body_offset: int = 0,
    ) -> None:
        # mjlab namespaces every element by entity name ("robot/joint1"), so a task-level
        # controller must prefix its lookups; the standalone MuJoCo path uses prefix="".
        # body_offset=1 shifts model body indices into mjlab EntityData indices, which omit
        # the world body -- entity arrays are one shorter and off by one throughout.
        self.device = torch.device(device)
        self.joint_names = tuple(joint_names)
        joint_names = tuple(prefix + n for n in joint_names)
        site_name = prefix + site_name

        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in joint_ids):
            missing = [n for n, j in zip(joint_names, joint_ids, strict=True) if j < 0]
            raise ValueError(f"Joints not found: {missing}")
        for j, name in zip(joint_ids, joint_names, strict=True):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(
                    f"ArmJacobian only supports hinge joints; {name!r} is "
                    f"{mujoco.mjtJoint(model.jnt_type[j]).name}."
                )

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"No site named {site_name!r}.")

        # Body carrying each joint. Its pose maps the joint's local axis/anchor into world.
        self.joint_body_ids = torch.as_tensor(
            [model.jnt_bodyid[j] - body_offset for j in joint_ids],
            dtype=torch.long,
            device=self.device,
        )
        self.axis_local = torch.as_tensor(
            np.array([model.jnt_axis[j] for j in joint_ids]),
            dtype=torch.float32,
            device=self.device,
        )
        self.anchor_local = torch.as_tensor(
            np.array([model.jnt_pos[j] for j in joint_ids]), dtype=torch.float32, device=self.device
        )
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
        self.num_joints = len(joint_ids)

    def compute(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        site_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return the (num_envs, 6, num_joints) site Jacobian in world axes.

        Args:
            body_pos_w: (num_envs, nbody, 3) body origins.
            body_quat_w: (num_envs, nbody, 4) body orientations, wxyz.
            site_pos_w: (num_envs, 3) position of the controlled site.

        Rows 0:3 are translational, 3:6 rotational, matching ``mj_jacSite``'s jacp/jacr split.
        """
        joint_pos = body_pos_w[:, self.joint_body_ids, :]  # (N, J, 3)
        joint_quat = body_quat_w[:, self.joint_body_ids, :]  # (N, J, 4)

        axis_w = _quat_rotate(joint_quat, self.axis_local.expand_as(joint_pos))
        anchor_w = joint_pos + _quat_rotate(joint_quat, self.anchor_local.expand_as(joint_pos))

        # Revolute column: omega = axis, v = axis x (p_site - anchor).
        lever = site_pos_w.unsqueeze(1) - anchor_w  # (N, J, 3)
        jac_pos = torch.cross(axis_w, lever, dim=-1)  # (N, J, 3)
        return torch.cat([jac_pos.transpose(1, 2), axis_w.transpose(1, 2)], dim=1)  # (N, 6, J)


class GravityCompensation:
    """Batched gravity-compensation torques for a serial revolute chain.

    Computed here rather than via MuJoCo's per-body ``gravcomp`` attribute so that
    compensation is a property of the *controller*, not of the model: the same MJCF then
    serves a gravity-compensated policy and an uncompensated one, and the term can be scaled
    or ablated without recompiling.

    mjlab exposes no ``qfrc_bias``, but the gravity term needs no dynamics -- only which
    masses hang off each joint and where their centres of mass are::

        tau_i = sum_{j in subtree(i)} m_j * g . (axis_i x (com_j - anchor_i))

    Note this is the *gravity* term alone; the Coriolis/centrifugal part of ``qfrc_bias`` is
    not reproduced. At the speeds this arm runs that is negligible, and it matches what
    IsaacLab's ``get_gravity_compensation_forces`` supplied.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        device: torch.device | str = "cpu",
        gravity: tuple[float, float, float] | None = None,
        prefix: str = "",
        body_offset: int = 0,
    ) -> None:
        self.device = torch.device(device)
        joint_names = tuple(prefix + n for n in joint_names)
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        self.joint_body_ids = torch.as_tensor(
            [model.jnt_bodyid[j] - body_offset for j in joint_ids],
            dtype=torch.long,
            device=self.device,
        )
        self.axis_local = torch.as_tensor(
            np.array([model.jnt_axis[j] for j in joint_ids]),
            dtype=torch.float32,
            device=self.device,
        )
        self.anchor_local = torch.as_tensor(
            np.array([model.jnt_pos[j] for j in joint_ids]), dtype=torch.float32, device=self.device
        )
        g = model.opt.gravity if gravity is None else np.asarray(gravity)
        self.gravity = torch.as_tensor(np.array(g), dtype=torch.float32, device=self.device)

        # mass_mask[i, j] = m_j if body j hangs off joint i, else 0. Walking parents once here
        # turns the per-step subtree sum into a single matmul.
        nbody = model.nbody
        mask = np.zeros((len(joint_ids), nbody), dtype=np.float32)
        for i, jid in enumerate(joint_ids):
            root = model.jnt_bodyid[jid]
            for body in range(nbody):
                ancestor = body
                while ancestor > 0:
                    if ancestor == root:
                        mask[i, body] = model.body_mass[body]
                        break
                    ancestor = model.body_parentid[ancestor]
        # Drop the leading world column when indexing mjlab EntityData, whose body arrays
        # start at the first real body.
        mask = mask[:, body_offset:]
        self.mass_mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        self.subtree_mass = self.mass_mask.sum(dim=1)

    def compute(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_com_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Return (num_envs, num_joints) torques that cancel gravity on the chain."""
        joint_pos = body_pos_w[:, self.joint_body_ids, :]
        joint_quat = body_quat_w[:, self.joint_body_ids, :]
        axis_w = _quat_rotate(joint_quat, self.axis_local.expand_as(joint_pos))
        anchor_w = joint_pos + _quat_rotate(joint_quat, self.anchor_local.expand_as(joint_pos))

        # g . (axis x r) == r . (g x axis), which collapses the per-body sum into a matmul
        # over centres of mass instead of an (envs, joints, bodies, 3) intermediate.
        moment_axis = torch.cross(self.gravity.expand_as(axis_w), axis_w, dim=-1)  # (N, J, 3)
        weighted_com = torch.einsum("jb,nbc->njc", self.mass_mask, body_com_pos_w)  # (N, J, 3)
        torque = (weighted_com * moment_axis).sum(-1) - self.subtree_mass * (
            anchor_w * moment_axis
        ).sum(-1)
        return -torque
