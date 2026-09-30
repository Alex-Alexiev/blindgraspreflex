"""Runtime patch for an mjlab per-world-mesh-variant bug (mjlab 1.5.2).

When a scene contains a ``VariantEntityCfg`` (per-world mesh variants / "Heterogeneous
Worlds"), the variant realization path never back-assigns the compiled global IDs onto the
*sibling* entities' specs -- their ``MjsBody/MjsSite/...`` ``.id`` fields stay ``-1``.
``Entity._compute_indexing`` reads those ``.id`` values, so an unlucky sibling (here, the
robot) gets ``body_ids = site_ids = joint_ids = [-1, -1, ...]``. Indexing ``data.xpos[:, -1]``
then returns the *last* row for every body, collapsing all forward-kinematics data to a single
position. Concretely this zeroed the arm's gravity compensation, put the impedance control
point on the wrong link, and fed garbage into every positional observation -- so multi-object
training silently never learned and the arm went unstable.

The compiled scene model *does* name the elements correctly (``robot/palm`` -> global id 12),
and the variant spec's element names are already fully prefixed, so we re-resolve the broken
indices by name. Joint *addresses* (qpos/qvel) and ``ctrl_ids`` are name-resolved upstream and
stay correct, so we leave them alone.

Remove this once mjlab fixes sibling-entity index resolution under variants.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.entity.entity import Entity, EntityIndexing

_APPLIED = False


def apply_variant_indexing_patch() -> None:
    """Idempotently install the fix. Safe on non-variant scenes (it no-ops when IDs resolved)."""
    global _APPLIED
    if _APPLIED:
        return

    _orig_compute_indexing = Entity._compute_indexing

    def _compute_indexing(self, model: mujoco.MjModel, device: str) -> EntityIndexing:
        idx = _orig_compute_indexing(self, model, device)
        # Healthy entities (including the variant object itself) resolve to >= 0; only the
        # broken siblings have -1. Bail out fast in the common case.
        if idx.body_ids.numel() == 0 or int(idx.body_ids.min()) >= 0:
            return idx

        def by_name(elems, objtype) -> torch.Tensor:
            return torch.tensor(
                [mujoco.mj_name2id(model, objtype, e.name) for e in elems],
                dtype=torch.int,
                device=device,
            )

        idx.body_ids = by_name(idx.bodies, mujoco.mjtObj.mjOBJ_BODY)
        idx.site_ids = by_name(idx.sites, mujoco.mjtObj.mjOBJ_SITE)
        idx.joint_ids = by_name(idx.joints, mujoco.mjtObj.mjOBJ_JOINT)
        if idx.tendons:
            idx.tendon_ids = by_name(idx.tendons, mujoco.mjtObj.mjOBJ_TENDON)
        if idx.materials:
            idx.mat_ids = by_name(idx.materials, mujoco.mjtObj.mjOBJ_MATERIAL)

        # Geoms can be unnamed, so collect them from each (now-correct) body's geom range
        # rather than by name. Bodies are contiguous, geoms are grouped by body in order.
        geom_ids: list[int] = []
        for bid in idx.body_ids.tolist():
            adr = int(model.body_geomadr[bid])
            geom_ids.extend(range(adr, adr + int(model.body_geomnum[bid])))
        idx.geom_ids = torch.tensor(geom_ids, dtype=torch.int, device=device)

        # mocap_id was computed from the (-1) root body id; recompute from the fixed root.
        if idx.mocap_id is not None:
            idx.mocap_id = int(model.body_mocapid[int(idx.body_ids[0])])

        return idx

    Entity._compute_indexing = _compute_indexing

    # root_body_id is a property that reads bodies[0].id (also -1); fall back to body_ids[0].
    def _root_body_id(self):
        rid = self.bodies[0].id
        return rid if (rid is not None and rid >= 0) else int(self.body_ids[0])

    EntityIndexing.root_body_id = property(_root_body_id)

    _APPLIED = True


_FAR = 1000.0
"""Body-frame offset the dead slots are parked at. Large enough that no scene geometry can reach
them, small enough not to strain single-precision positions."""


def disable_padded_variant_geoms(env, entity_name: str = "object", verbose: bool = True) -> int:
    """Neutralise per-world variant geom slots that carry no mesh (mjlab 1.5.2).

    A ``VariantEntityCfg`` compiles ONE model padded to the largest variant's geom count, then
    per world points ``geom_dataid`` at that variant's meshes. Slots a given variant does not use
    are left with ``geom_dataid = -1`` -- but mjlab does not clear their ``geom_size`` /
    ``geom_rbound`` / ``geom_aabb``, and ``geom_contype``/``geom_conaffinity`` are shared across
    worlds, so they stay collidable. They keep whatever extent the merge last wrote, which is a
    LARGER object's convex hulls sitting a few centimetres off the body origin.

    Measured on the apple (1 real hull, 15 padded slots) with the hand closed: over 60 hold steps
    the padded slots produced 246 penetrating contacts against the real hull's 219. The hand was
    gripping phantom geometry, and the same object was 25-50% graspable in a single-object scene
    against ~100% here.

    ``geom_contype`` is not per-world, so the slots cannot simply be disabled for one world.
    ``geom_pos`` IS per-world, so each dead slot is MOVED far outside the scene instead, and its
    bounding radius zeroed for good measure. Shrinking alone does not work -- zeroing
    ``geom_size``/``geom_rbound``/``geom_aabb`` left the contacts in place (measured: 212 phantom
    contacts before, 212 after), while relocating removed them outright (212 -> 0, with real-hull
    contacts rising 289 -> 400 as the hand finally reached the object).

    Call once after the env is built, before stepping. Returns the number of (world, slot) pairs
    neutralised. Remove when mjlab clears padded slots itself.
    """
    import numpy as _np

    sim = env.sim
    mo = sim.model
    did = mo.geom_dataid
    # sim.model fields are TorchArray wrappers here (per-world), not warp arrays -- .numpy()
    # exists but raises; go through torch.
    if isinstance(did, torch.Tensor):
        did_np = did.detach().cpu().numpy()
    else:
        try:
            did_np = _np.asarray(did.numpy())
        except Exception:  # noqa: BLE001
            did_np = torch.as_tensor(did[...]).detach().cpu().numpy()
    if did_np.ndim != 2:
        if verbose:
            print("[patch] geom_dataid is not per-world; no variant padding to clear")
        return 0

    ent = env.scene[entity_name]
    bid = ent.data.indexing.root_body_id
    body_of = env.sim._mj_model.geom_bodyid
    slots = [g for g in range(did_np.shape[1]) if body_of[g] == bid]
    if not slots:
        return 0

    dead = _np.zeros_like(did_np, dtype=bool)
    dead[:, slots] = did_np[:, slots] < 0
    n = int(dead.sum())
    if n:
        w, g = _np.nonzero(dead)
        wi = torch.as_tensor(w, dtype=torch.long)
        gi = torch.as_tensor(g, dtype=torch.long)
        mo.geom_pos[wi, gi] = _FAR  # body-frame offset, far outside any reachable space
        mo.geom_rbound[wi, gi] = 0.0
        mo.geom_size[wi, gi] = 0.0
    if verbose:
        per_world = dead.sum(axis=1)
        print(
            f"[patch] neutralised {n} padded '{entity_name}' geom slots "
            f"({per_world.min()}-{per_world.max()} per world, of {len(slots)} slots)"
        )
    return n
