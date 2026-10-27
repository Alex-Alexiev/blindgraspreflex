"""Primitive clutter grasping task built on the blind grasp MJLab plant."""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from blindgraspreflex.assets.floating_hand import PALM_FACE_SITE_NAME, get_floating_hand_cfg
from blindgraspreflex.assets.primitives import (
    get_primitive_bounding_radius,
    get_primitive_hull_vertices,
    get_primitive_object_cfg,
)
from blindgraspreflex.controllers.batched import quat_rotate
from blindgraspreflex.tasks.grasping import mjlab_task as base
from blindgraspreflex.tasks.grasping.floating_hand_action import FloatingHandActionCfg
from blindgraspreflex.tasks.grasping.mdp.terms import REWARD_CONTACT_LINKS

TARGET = "prim_010"  # round, centrally placed
OBSTACLES = (
    "prim_000",  # box
    "prim_005",  # cylinder
    "prim_014",  # flat ellipsoid
    "prim_018",  # rounded box
    "prim_022",  # rounded cylinder
    "prim_027",  # square-section rounded profile
    "prim_036",  # boxy blob
)
OBSTACLE_NAMES = tuple(f"obstacle_{i}" for i in range(len(OBSTACLES)))
OBSTACLE_SENSORS = tuple(f"hand_{name}" for name in OBSTACLE_NAMES)
OBSERVED_OBSTACLES = 5
MAX_OBSTACLE_PUSH = 0.15  # [m] per-obstacle clearance gain that still earns reward
PACK_GAP_MAX = 0.01  # [m] spawn gap between an obstacle and whatever it is packed against
PACK_DIRECTIONS = 180  # resolution of the shapes' support functions; yaws use the same grid
PACK_ANGLES = 48  # candidate placement directions around the target per obstacle
PACK_CHUNK = 2048  # envs packed at once; memory is ~280 KB per env


def _ids(env, env_ids):
    return (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )


def _obstacle_positions(env) -> torch.Tensor:
    return torch.stack([env.scene[name].data.root_link_pos_w for name in OBSTACLE_NAMES], dim=1)


def _support_table(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal support functions of the collision hulls, sampled on PACK_DIRECTIONS.

    Returns ``(support, directions)``: support is (1 + len(OBSTACLES), PACK_DIRECTIONS), the target
    in its spawn orientation followed by the obstacles at zero yaw; directions is
    (PACK_DIRECTIONS, 2). ``support[s, i]`` is how far shape s reaches along ``directions[i]``.
    """
    if not hasattr(env, "_clutter_support"):
        phi = torch.arange(PACK_DIRECTIONS, dtype=torch.float64) * (2 * math.pi / PACK_DIRECTIONS)
        directions = torch.stack([phi.cos(), phi.sin()], dim=1)
        target_quat = env.scene["object"].data.default_root_state[0, 3:7].double().cpu()
        rows = []
        for name in (TARGET, *OBSTACLES):
            v = torch.as_tensor(get_primitive_hull_vertices(name))
            if name == TARGET:
                v = quat_rotate(target_quat.expand(len(v), 4), v)
            rows.append((v[:, :2] @ directions.T).amax(dim=0))
        env._clutter_support = torch.stack(rows).float().to(env.device)
        env._clutter_directions = directions.float().to(env.device)
    return env._clutter_support, env._clutter_directions


def _pack_obstacles(env, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy tight cluster around the target, in a random order per env.

    Each obstacle takes, among PACK_ANGLES directions from the target, the closest spot whose gap
    to every body already placed (the target included) is at least a gap drawn from
    U(0, PACK_GAP_MAX). The spot is found where that bound is tight, so the obstacle ends up 0-1 cm
    from something. Overlap is tested on the horizontal outlines of the hulls (separating axes on
    the support functions), so it holds whatever their heights. Returns centres (n, 7, 2) relative
    to the target and yaw indices (n, 7) into the PACK_DIRECTIONS grid.
    """
    support, directions = _support_table(env)
    dev, k_dirs, n_obs = env.device, PACK_DIRECTIONS, len(OBSTACLES)
    yaw_idx = torch.randint(k_dirs, (n, n_obs), device=dev)
    # A shape rotated by yaw reaches h(phi - yaw) along phi.
    shift = (torch.arange(k_dirs, device=dev) - yaw_idx[..., None]) % k_dirs
    reach = support[1:].expand(n, -1, -1).gather(2, shift)  # (n, 7, K)
    reach_back = reach.roll(k_dirs // 2, dims=-1)  # reach along -phi
    bodies = torch.cat([support[:1].expand(n, 1, k_dirs), reach], dim=1)  # (n, 8, K)
    centres = torch.zeros(n, 1 + n_obs, 2, device=dev)  # slot 0 is the target, at the origin
    placed = torch.zeros(n, 1 + n_obs, dtype=torch.bool, device=dev)
    placed[:, 0] = True
    order = torch.rand(n, n_obs, device=dev).argsort(dim=1)
    rows = torch.arange(n, device=dev)
    for step in range(n_obs):
        k = order[:, step]
        gap = torch.rand(n, device=dev) * PACK_GAP_MAX
        theta = (torch.arange(PACK_ANGLES, device=dev) + torch.rand(n, 1, device=dev)) * (
            2 * math.pi / PACK_ANGLES
        )
        u = torch.stack([theta.cos(), theta.sin()], dim=-1)  # (n, A, 2)
        along = u @ directions.T  # (n, A, K): u . axis
        # At d*u the obstacle clears body j by >= gap along an axis with u . axis > 0 once
        # d * (u . axis) >= gap + reach_j(axis) + reach_k(-axis) + c_j . axis.
        bound = gap[:, None, None] + bodies + reach_back[rows, k][:, None] + centres @ directions.T
        need = bound[:, None] / along[:, :, None]  # (n, A, 8, K)
        need = need.masked_fill(along[:, :, None] <= 1e-6, math.inf).amin(dim=-1)
        need = need.masked_fill(~placed[:, None], 0.0)
        distance = need.amax(dim=-1).clamp(min=0.0)  # (n, A)
        best = distance.argmin(dim=1)
        centres[rows, 1 + k] = distance[rows, best, None] * u[rows, best]
        placed[rows, 1 + k] = True
    return centres[:, 1:], yaw_idx


def reset_clutter(env, env_ids) -> None:
    """Pack the obstacles in a tight cluster around the target (see _pack_obstacles)."""
    ids = _ids(env, env_ids)
    n = len(ids)
    if not n:
        return
    origin = env.scene.env_origins[ids]
    target_xy = torch.tensor(base.SPAWN_XY, device=env.device).expand(n, 2)
    target_xy = target_xy + (torch.rand(n, 2, device=env.device) - 0.5) * 0.02
    target = env.scene["object"]
    pose = target.data.default_root_state[ids, :7].clone()
    pose[:, :2] = target_xy + origin[:, :2]
    pose[:, 2] += origin[:, 2]
    target.write_root_link_pose_to_sim(pose, env_ids=ids)
    target.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)

    # All objects spawn at the same height above the mat and settle under physics. The packing
    # separates their horizontal outlines, so they cannot touch while they fall.
    packed = [_pack_obstacles(env, len(chunk)) for chunk in torch.split(ids, PACK_CHUNK)]
    offsets = torch.cat([centres for centres, _ in packed])
    yaws = torch.cat([yaw for _, yaw in packed]) * (2 * math.pi / PACK_DIRECTIONS)
    obstacle_centres = []
    for i, name in enumerate(OBSTACLE_NAMES):
        entity = env.scene[name]
        state = entity.data.default_root_state[ids, :7].clone()
        state[:, :2] = pose[:, :2] + offsets[:, i]
        state[:, 2] += origin[:, 2]
        yaw = yaws[:, i]
        state[:, 3] = (yaw / 2).cos()
        state[:, 4:6] = 0.0
        state[:, 6] = (yaw / 2).sin()
        entity.write_root_link_pose_to_sim(state, env_ids=ids)
        entity.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
        obstacle_centres.append(state[:, :3])

    if not hasattr(env, "_clutter_spawn"):
        env._clutter_spawn = torch.zeros(env.num_envs, 3, device=env.device)
        env._clutter_initial_clearance = torch.zeros(
            env.num_envs, len(OBSTACLE_NAMES), device=env.device
        )
    env._clutter_spawn[ids] = pose[:, :3]
    # Reset events run before forward(), so use the positions just written rather than
    # root_link_pos_w, which still reflects the previous episode.
    centres = torch.stack(obstacle_centres, dim=1)
    env._clutter_initial_clearance[ids] = torch.linalg.norm(
        centres[:, :, :2] - pose[:, None, :2], dim=-1
    )


def palm_pose(env) -> torch.Tensor:
    robot = env.scene["robot"]
    sid = base._grasp_site_id(env)
    return torch.cat(
        [
            robot.data.site_pos_w[:, sid] - env.scene.env_origins,
            robot.data.site_quat_w[:, sid],
        ],
        dim=-1,
    )


def palm_twist(env) -> torch.Tensor:
    data = env.scene["robot"].data
    sid = base._grasp_site_id(env)
    angular = data.root_link_ang_vel_w
    site_offset_w = data.site_pos_w[:, sid] - data.root_link_pos_w
    linear = data.root_link_lin_vel_w + torch.linalg.cross(angular, site_offset_w)
    return torch.cat([linear, angular], dim=-1)


def hand_joint_vel(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(base.HAND_JOINT_NAMES), preserve_order=True)
    return robot.data.joint_vel[:, ids]


def nearest_obstacles(env) -> torch.Tensor:
    """Five nearest obstacle centres relative to target, each followed by sphere radius."""
    rel = _obstacle_positions(env) - base.object_pos(env).unsqueeze(1)
    dist = torch.linalg.norm(rel, dim=-1)
    nearest = torch.argsort(dist, dim=1)[:, :OBSERVED_OBSTACLES]
    xyz = torch.gather(rel, 1, nearest.unsqueeze(-1).expand(-1, -1, 3))
    if not hasattr(env, "_clutter_radii"):
        env._clutter_radii = torch.tensor(
            [get_primitive_bounding_radius(name) for name in OBSTACLES],
            device=env.device,
            dtype=xyz.dtype,
        )
    radii = env._clutter_radii.expand(env.num_envs, -1)
    radius = torch.gather(radii, 1, nearest).unsqueeze(-1)
    return torch.cat([xyz, radius], dim=-1).flatten(start_dim=1)


def target_displacement(env) -> torch.Tensor:
    """Table-plane motion from spawn; lifting the grasped target remains possible."""
    return torch.linalg.norm(base.object_pos(env)[:, :2] - env._clutter_spawn[:, :2], dim=-1)


def obstacle_clearance_gain(env) -> torch.Tensor:
    """Increase in mean table-plane target-to-obstacle centre distance.

    Each obstacle's increase counts up to ``MAX_OBSTACLE_PUSH``, so pushing one further earns
    nothing extra. Only the mean is floored at zero: an obstacle pushed toward the target still
    cancels gains on the others.
    """
    distance = torch.linalg.norm(
        _obstacle_positions(env)[:, :, :2] - base.object_pos(env)[:, None, :2], dim=-1
    )
    gain = (distance - env._clutter_initial_clearance).clamp(max=MAX_OBSTACLE_PUSH)
    return gain.mean(dim=1).clamp(min=0.0)


def palm_proximity_facing(env) -> torch.Tensor:
    """Reward a nearby palm whose face points at the target centre.

    Measured from a point on the palm face at the knuckles, along the face normal, the palm's +X
    (+Z runs along the fingers). That point cannot pass the target centre without penetrating the target, so the
    direction stays defined right up to contact.
    """
    robot = env.scene["robot"]
    sid = base._grasp_site_id(env, PALM_FACE_SITE_NAME)
    to_target = base.object_pos(env) - robot.data.site_pos_w[:, sid]
    distance = torch.linalg.norm(to_target, dim=-1)
    direction = to_target / distance.clamp_min(1e-6).unsqueeze(-1)
    palm_normal = quat_rotate(
        robot.data.site_quat_w[:, sid],
        torch.tensor([1.0, 0.0, 0.0], device=env.device).expand_as(to_target),
    )
    facing = ((palm_normal * direction).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) / 2.0
    return torch.exp(-distance / 0.15) * facing.square()


def obstacle_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    """One hand<->obstacle sensor per obstacle, since a sensor takes a single secondary body.

    Primaries are the base task's 16 reward links, so its link order and inward axes apply.
    Only ``normal`` is published: unmatched slots are zeroed, so a nonzero normal is a contact.
    Every MuJoCo contact sensor scans every contact on every physics step, and a separate
    ``found`` field would double that cost.
    """
    links = f"^({'|'.join(REWARD_CONTACT_LINKS)})$"
    return tuple(
        ContactSensorCfg(
            name=sensor,
            primary=ContactMatch(mode="body", pattern=links, entity="robot"),
            secondary=ContactMatch(mode="body", pattern=f"^{name}$", entity=name),
            fields=("normal",),
            reduce="none",
            num_slots=1,
        )
        for sensor, name in zip(OBSTACLE_SENSORS, OBSTACLE_NAMES, strict=True)
    )


def _obstacle_contact_sides(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(inside, outside) per reward link, each (num_envs, 16) bool, over all obstacles.

    A contact is inside when the sensor normal, which points from the link to the obstacle,
    lies along the link's inward axis (+X: the palm face and the finger pads). Any other touch
    on these links, on the back or side of the hand, is outside. The phalanx-1 knuckles are not
    counted because their inward axis is not well defined. This is ``base._contact_inside``'s
    gate; it is repeated here because these sensors publish only ``normal``.
    """
    term = env.action_manager.get_term("grasp")
    quat = env.scene["robot"].data.body_link_quat_w[:, term._reward_body_ids]
    inward = quat_rotate(quat, term._inward_local.unsqueeze(0).expand_as(quat[..., :3]))
    inside = torch.zeros(quat.shape[:2], dtype=torch.bool, device=env.device)
    outside = torch.zeros_like(inside)
    for sensor in OBSTACLE_SENSORS:
        perm = base._reward_link_order(env, sensor)
        normal = env.scene[sensor].data.normal[:, perm]
        found = normal.abs().sum(dim=-1) > 0.0
        faces_in = (normal * inward).sum(dim=-1) > 0.0
        inside |= found & faces_in
        outside |= found & ~faces_in
    return inside, outside


def obstacle_outside_contact(env) -> torch.Tensor:
    """Number of hand links touching an obstacle with their back or side."""
    return _obstacle_contact_sides(env)[1].sum(dim=1).float()


def obstacle_inside_contact(env) -> torch.Tensor:
    """Number of hand links touching an obstacle with the palm or finger pads."""
    return _obstacle_contact_sides(env)[0].sum(dim=1).float()


def clutter_unstable(env) -> torch.Tensor:
    bad = base._unstable(env)
    for name in (*OBSTACLE_NAMES, "robot"):
        data = env.scene[name].data
        bad |= ~torch.isfinite(data.root_link_pos_w).all(dim=-1)
        bad |= torch.linalg.norm(data.root_link_lin_vel_w, dim=-1) > 15.0
    return bad


def reset_hand_joints(env, env_ids) -> None:
    from mjlab.envs.mdp.events import reset_joints_by_offset

    reset_joints_by_offset(
        env,
        env_ids,
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
        asset_cfg=SceneEntityCfg("robot", joint_names=tuple(base.HAND_JOINT_NAMES)),
    )


def make_clutter_task(num_envs: int = 4096):
    cfg = base.make_task(TARGET, num_envs=num_envs, source="primitives")
    cfg.scene.entities["robot"] = get_floating_hand_cfg()
    cfg.scene.entities["object"] = get_primitive_object_cfg(
        TARGET,
        pos=(*base.SPAWN_XY, base.SPAWN_Z),
        rgba=(0.04, 0.28, 0.9, 1.0),
    )
    obstacle_colors = (
        (0.9, 0.16, 0.12, 1.0),
        (0.95, 0.45, 0.05, 1.0),
        (0.75, 0.25, 0.65, 1.0),
        (0.95, 0.7, 0.08, 1.0),
        (0.65, 0.2, 0.65, 1.0),
        (0.85, 0.28, 0.25, 1.0),
        (0.55, 0.3, 0.16, 1.0),
    )
    for i, (name, primitive) in enumerate(zip(OBSTACLE_NAMES, OBSTACLES, strict=True)):
        cfg.scene.entities[name] = get_primitive_object_cfg(
            primitive,
            pos=(base.SPAWN_XY[0], base.SPAWN_XY[1], base.SPAWN_Z),
            body_name=name,
            rgba=obstacle_colors[i],
        )
    def clutter_ground(spec):
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(3.0, 3.0, 0.1),
            pos=(0.0, 0.0, base.FLOOR_HEIGHT),
            condim=6,
            friction=(1.0, 5e-3, 1e-4),
        )
        spec.worldbody.add_geom(
            name="mat",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(*base.MAT_HALF_EXTENT, base.MAT_THICKNESS * 0.5),
            pos=(*base.MAT_CENTRE_XY, base.FLOOR_HEIGHT + base.MAT_THICKNESS * 0.5),
            condim=6,
            friction=(1.2, 5e-3, 1e-4),
            solref=(0.01, 1.0),
            rgba=(0.18, 0.21, 0.25, 1.0),
        )

    cfg.scene.spec_fn = clutter_ground
    # The base task's target sensors feed grasp rewards this task does not use, and each MuJoCo
    # contact sensor scans every contact on every physics step.
    cfg.scene.sensors = (
        *(s for s in cfg.scene.sensors if s.name not in ("hand_object", "hand_object_grasp")),
        *obstacle_contact_sensor_cfgs(),
    )
    cfg.actions["grasp"] = FloatingHandActionCfg()
    actor = {
        "palm_pose": ObservationTermCfg(func=palm_pose, params={}),
        "palm_twist": ObservationTermCfg(func=palm_twist, params={}),
        "hand_joint_pos": ObservationTermCfg(func=base.hand_joint_pos, params={}),
        "hand_joint_vel": ObservationTermCfg(func=hand_joint_vel, params={}),
        "hand_command_error": ObservationTermCfg(func=base.hand_tracking_error, params={}),
        "palm_to_target": ObservationTermCfg(func=base.palm_object_vector, params={}),
        "nearest_obstacles": ObservationTermCfg(func=nearest_obstacles, params={}),
        "contacts": ObservationTermCfg(func=base.fingertip_sphere_contact_binary, params={}),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg(terms=actor, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms=dict(actor), concatenate_terms=True),
    }
    cfg.rewards = {
        # By the triangle inequality, moving only the target can increase mean clearance by
        # no more than its displacement. The larger penalty prevents that shortcut.
        "target_displacement": RewardTermCfg(func=target_displacement, params={}, weight=-12.0),
        "obstacle_clearance": RewardTermCfg(func=obstacle_clearance_gain, params={}, weight=10.0),
        "palm_proximity_facing": RewardTermCfg(func=palm_proximity_facing, params={}, weight=0.5),
        # Push obstacles with the back and sides of the hand, keeping the palm and finger pads
        # for the target. Per link in contact; the penalty is twice the reward, so a contact
        # that loads inside and outside links equally nets negative.
        "obstacle_outside_contact": RewardTermCfg(
            func=obstacle_outside_contact, params={}, weight=0.1
        ),
        "obstacle_inside_contact": RewardTermCfg(
            func=obstacle_inside_contact, params={}, weight=-0.2
        ),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=base._time_out, time_out=True),
        "instability": TerminationTermCfg(func=clutter_unstable, time_out=True),
    }
    cfg.events = {
        "reset_robot_base": EventTermCfg(func=base._reset_robot_base, mode="reset", params={}),
        "reset_robot_joints": EventTermCfg(func=reset_hand_joints, mode="reset", params={}),
        "reset_clutter": EventTermCfg(func=reset_clutter, mode="reset", params={}),
    }
    cfg.episode_length_s = 8.0
    cfg.decimation = 50  # 0.002 s physics step × 50 = 0.1 s policy step (10 Hz).
    cfg.sim.nconmax = 800
    cfg.sim.njmax = 2400
    return cfg


def make_clutter_lift_task(num_envs: int = 4096, with_clearance: bool = False):
    """The clutter scene with the grasp task's lift rewards in place of the decluttering ones.

    Scene, hand, start pose, observations and actions are make_clutter_task's. The rewards, the
    success termination and the reward-state reset come from mjlab_task.make_task: inside-of-hand
    contact coverage, object height, holding the object at the lift goal, and a bonus after a 2 s
    hold, which also ends the episode. Terms with zero weight are left out. ``with_clearance``
    keeps the decluttering task's obstacle clearance term as well, at its weight.
    """
    cfg = make_clutter_task(num_envs)
    clearance = cfg.rewards["obstacle_clearance"]
    grasp = base.make_task(TARGET, num_envs=num_envs, source="primitives")
    cfg.rewards = {name: term for name, term in grasp.rewards.items() if term.weight != 0.0}
    if with_clearance:
        cfg.rewards["obstacle_clearance"] = clearance
    cfg.terminations["success"] = grasp.terminations["success"]
    cfg.events["reset_state"] = grasp.events["reset_state"]
    # The contact-coverage term reads the target sensor; the obstacle sensors feed only the
    # decluttering rewards.
    cfg.scene.sensors = (
        *(s for s in cfg.scene.sensors if s.name not in OBSTACLE_SENSORS),
        base.hand_object_sensor_cfg(),
    )
    return cfg


def register() -> None:
    from mjlab.tasks.registry import register_mjlab_task

    rl_cfg = base.grasp_ppo_cfg()
    rl_cfg.experiment_name = "primitive_clutter_grasp"
    rl_cfg.save_interval = 10
    register_mjlab_task(
        task_id="Reflexive-Grasp-Primitive-Clutter",
        env_cfg=make_clutter_task(),
        play_env_cfg=make_clutter_task(num_envs=16),
        rl_cfg=rl_cfg,
    )
    register_mjlab_task(
        task_id="Reflexive-Grasp-Primitive-Clutter-Lift",
        env_cfg=make_clutter_lift_task(),
        play_env_cfg=make_clutter_lift_task(num_envs=16),
        rl_cfg=rl_cfg,
    )
    register_mjlab_task(
        task_id="Reflexive-Grasp-Primitive-Clutter-Lift-Clear",
        env_cfg=make_clutter_lift_task(with_clearance=True),
        play_env_cfg=make_clutter_lift_task(num_envs=16, with_clearance=True),
        rl_cfg=rl_cfg,
    )
