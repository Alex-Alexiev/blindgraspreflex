"""Print library versions and a hash of the compiled clutter scene, to compare two machines."""

import hashlib
import importlib.metadata as md

import numpy as np
from mjlab.scene import Scene

from blindgraspreflex.tasks.grasping.clutter_task import make_clutter_task

for pkg in ("mujoco", "mujoco-warp", "warp-lang", "mjlab", "torch"):
    print(f"{pkg} {md.version(pkg)}")
cfg = make_clutter_task(num_envs=1)
m = Scene(cfg.scene, device="cpu").compile()
m.opt.timestep = cfg.sim.mujoco.timestep
fields = (
    "body_pos body_quat body_mass body_inertia geom_type geom_size geom_pos geom_quat geom_friction "
    "geom_contype geom_conaffinity geom_condim geom_solref geom_solimp jnt_range dof_damping "
    "dof_armature dof_frictionloss actuator_gainprm actuator_biasprm actuator_forcerange mesh_vert "
    "sensor_type sensor_objid sensor_refid"
).split()
h = hashlib.md5()
for name in fields:
    h.update(np.round(np.asarray(getattr(m, name), dtype=np.float64), 6).tobytes())
print(f"nbody {m.nbody} ngeom {m.ngeom} nv {m.nv} nu {m.nu} nsensor {m.nsensor} dt {m.opt.timestep}")
print(f"model hash {h.hexdigest()}")
