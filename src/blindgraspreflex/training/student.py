"""Teacher-student distillation for the grasp policy (rsl_rl Distillation / DAgger).

The teacher is a trained residual-arm PPO policy with *privileged* observations -- the
palm->object vector and all 16 hand contact points. The student is trained to reproduce the
teacher's actions from a *reduced* observation only:

    student obs = [5 fingertip binary contacts, hand_joint_pos(20), hand_tracking(20)]   (45-D)
    teacher obs = [16 contacts, palm_to_object(3), hand_joint_pos(20), hand_tracking(20)] (59-D)

i.e. no palm->object vector and only fingertip touch -- the information a real RH-5 hand actually
has. rsl_rl's Distillation rolls out the *student* and regresses it onto the teacher's action
(MSE), so the student is trained on its own state distribution (DAgger), avoiding covariate shift.

The teacher checkpoint is a normal PPO checkpoint; ``Distillation.load`` detects ``actor_state_dict``
and loads it straight into the teacher model (frozen), so no manual weight mapping is needed.
"""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.runners import DistillationRunner

from blindgraspreflex.tasks.grasping.mjlab_task import fingertip_contact_binary, register


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Sanitise obs/rewards so a rare physics blow-up resets the world instead of crashing the
    unattended overnight run (same guard as scripts/train_grasp.py)."""

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        for key in list(obs.keys()):
            obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, rew, dones, extras


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))


def build_env(
    device: str,
    num_envs: int,
    assignment=None,
    hand_mode: str = "pca",
    hand_pca_project=None,
    student_obs: str = "fingertip",
    arm_control: str = "impedance",
    hand_gate: bool = False,
    episode_length_s: float | None = None,
    reach_hand_pose: bool = True,
    teacher_contacts: bool = False,
    lift_height: float = 0.0,
    mass_range=None,
    pca_latent_scale: float = 1.0,
    no_success: bool = False,
    hold_weight: float | None = None,
    belief_student: bool = False,
    history: int = 5,
    student_history: int = 0,
    disturbance: float = 0.0,
    disturbance_angvel: float = 0.8,
    vanish_prob: float = 0.0,
    arm_residual: bool = True,
    extra_events: dict | None = None,
    physics_dt: float = 0.004,
    decimation: int = 21,
    task_id: str = "Reflexive-Grasp-YCB-All",
    freeze_object_on_engage: bool = False,
    render_mode: str | None = None,
) -> RslRlVecEnvWrapper:
    """YCB-All env matching the teacher's training config, but with student+teacher obs groups.

    ``task_id`` swaps the *object set* while keeping every other setting identical, so a student
    can be evaluated on GSO or GraspXL objects it never saw (see scripts/eval_object_set.py).

    ``hand_mode``/``hand_pca_project`` must match the teacher's action space so its weights load and
    the cloned actions mean the same thing (e.g. joints + 0.5 projection for the ft_gru_project50
    teacher). ``reach_hand_pose``/``teacher_contacts``/``lift_height`` must likewise match the
    teacher so its action dim (27 with reach off vs 47) and teacher-obs dim (59 with 16 contacts vs
    48) line up and its weights load. Object scale DR is set by the caller via REFLEXIVE_OBJECT_SCALES
    before register(); ``mass_range`` adds the per-episode mass DR event."""
    env_cfg = load_env_cfg(task_id)
    env_cfg.scene.num_envs = num_envs
    env_cfg.actions["grasp"].hand_mode = hand_mode
    # Must MATCH the teacher's action space, not be assumed. A teacher trained without the
    # residual has a 21-D action ([grasp_score, joints(20)]); building the student env with
    # arm_residual=True makes it 27-D and the teacher checkpoint will not load. This was hardcoded
    # True, which would have silently broken every no-residual distillation.
    env_cfg.actions["grasp"].arm_residual = arm_residual
    env_cfg.actions["grasp"].phase.reach_randomize = False
    env_cfg.actions["grasp"].arm_control = arm_control
    env_cfg.actions["grasp"].hand_gate = hand_gate
    env_cfg.actions["grasp"].reach_hand_pose = reach_hand_pose
    env_cfg.actions["grasp"].lift_height_offset = lift_height
    env_cfg.actions[
        "grasp"
    ].pca_latent_scale = pca_latent_scale  # match teacher's PCA range (e.g. 2.0)
    # Disturbances during DISTILLATION, not just during teacher training. Distillation is behaviour
    # cloning: the student only learns responses to states it actually visits. A teacher that
    # correctly lowers grasp_score when the object is knocked out teaches the student nothing about
    # that unless the student's own rollouts contain drops. (This is also the flag whose ABSENCE
    # silently killed the disturb run's student stage -- train_grasp had --disturbance, distill did
    # not, so passing it here was an argparse error and stage 2 exited instantly.)
    if disturbance > 0.0:
        from mjlab.envs.mdp.events import push_by_setting_velocity
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        v, w = disturbance, disturbance_angvel
        env_cfg.events["object_disturbance"] = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.3, 0.8),
            params={
                "velocity_range": {
                    "x": (-v, v),
                    "y": (-v, v),
                    "z": (-v, v),
                    "roll": (-w, w),
                    "pitch": (-w, w),
                    "yaw": (-w, w),
                },
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
    if vanish_prob > 0.0:
        from mjlab.managers.event_manager import EventTermCfg as _ETC

        from blindgraspreflex.tasks.grasping.mjlab_task import object_vanish

        env_cfg.events["object_vanish"] = _ETC(
            func=object_vanish,
            mode="interval",
            interval_range_s=(0.084, 0.084),
            params={"prob": vanish_prob},
        )
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
    if freeze_object_on_engage:
        env_cfg.actions["grasp"].freeze_object_on_engage = True
    if hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = hand_pca_project
    # Control rate = physics_dt * decimation. Vary the two TOGETHER to hold the policy tick fixed
    # (0.004 x 21 = 0.084 s); changing physics_dt alone silently changes how often the policy acts,
    # which would confound any physics-rate invariance test.
    env_cfg.sim.mujoco.timestep = physics_dt
    env_cfg.decimation = decimation
    # Match the teacher's reward/termination env so the cloned state distribution lines up. The
    # distillation loss is pure behaviour-cloning (no reward), so the weights themselves don't touch
    # the student -- but dropping the success TERMINATION matters: it stops episodes ending early at
    # the 2 s hold, so the student sees the teacher's full-10 s hold, exactly as the teacher trained.
    if hold_weight is not None:
        env_cfg.rewards["hold"].weight = hold_weight
    if no_success:
        env_cfg.rewards.pop("success", None)
        env_cfg.terminations.pop("success", None)
    if mass_range is not None:
        from mjlab.envs import mdp
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        env_cfg.events["object_mass_dr"] = EventTermCfg(
            func=mdp.dr.body_mass,
            mode="reset",
            params={
                "ranges": tuple(mass_range),
                "operation": "abs",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
    if assignment is not None:
        env_cfg.scene.entities["object"].assignment = assignment

    # "teacher" = exactly the terms the teacher trained on (its old "actor" group, same order, so
    # its baked obs-normaliser stats still line up). "student" = reduced obs.
    teacher_terms = dict(env_cfg.observations["actor"].terms)  # contacts, palm_to_object, hand_*
    if teacher_contacts:
        from blindgraspreflex.tasks.grasping.mjlab_task import contact_binary

        teacher_terms["contacts"] = ObservationTermCfg(func=contact_binary, params={})
    observations = {"teacher": ObservationGroupCfg(terms=teacher_terms, concatenate_terms=True)}

    if belief_student:
        # Belief/RMA student: input is a proprioception HISTORY (last `history` frames of
        # [hand_joint_pos, hand_tracking]; hand_tracking = target-q encodes the commanded action).
        # No measured contacts, no palm. The model reconstructs the 16 contacts from this history,
        # supervised by the separate "contacts" group (which no model reads as input).
        from blindgraspreflex.tasks.grasping.mjlab_task import contact_binary

        jp = copy.copy(teacher_terms["hand_joint_pos"])
        jp.history_length = history
        ht = copy.copy(teacher_terms["hand_tracking"])
        ht.history_length = history
        observations["student"] = ObservationGroupCfg(
            terms={"hand_joint_pos": jp, "hand_tracking": ht}, concatenate_terms=True
        )
        observations["contacts"] = ObservationGroupCfg(
            terms={"contacts": ObservationTermCfg(func=contact_binary, params={})},
            concatenate_terms=True,
        )
    else:
        # "fingertip" = fingertip contacts + proprioception (45-D); "proprio" = proprioception ONLY
        # (40-D, no tactile) -- the deploy target when the hand's fingertip sensors aren't used.
        student_terms = {
            "hand_joint_pos": teacher_terms["hand_joint_pos"],
            "hand_tracking": teacher_terms["hand_tracking"],
        }
        if student_obs == "pos":
            # Ablation: joint POSITIONS only, no error term of any kind. Pair with
            # --student-history so the stall itself is visible (a blocked finger's q goes flat),
            # which is the one contact cue position history can carry on its own. Isolates how much
            # the error block actually contributes over plain proprioception.
            student_terms.pop("hand_tracking", None)
        if student_obs == "m9":
            # Must be set BEFORE the env is constructed: GraspAction reads it to decide whether to
            # allocate the ring buffer and run the estimator at all.
            os.environ["REFLEXIVE_M9_OBS"] = "1"
            # Same 40-D width as "proprio", but the 20-D error block becomes the invariant contact
            # metric instead of a raw position error. The TEACHER's obs is untouched -- it stays the
            # 59-D group its weights were trained on, so this is a student learning the teacher's
            # actions from a different, plant-invariant view.
            from blindgraspreflex.tasks.grasping.mjlab_task import hand_contact_metric

            student_terms["hand_tracking"] = ObservationTermCfg(func=hand_contact_metric, params={})
        if student_obs == "fingertip":
            student_terms = {
                "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
                **student_terms,
            }
        if student_history and student_history > 1:
            # Frame-stack the student's own terms. A feed-forward student has no memory of its
            # own, so without this it cannot see the temporal structure (velocity-field lag decay,
            # stall onset) that the GRU integrates -- and the comparison would be about memory
            # rather than about architecture.
            student_terms = {k: copy.copy(v) for k, v in student_terms.items()}
            for v in student_terms.values():
                v.history_length = student_history
        observations["student"] = ObservationGroupCfg(terms=student_terms, concatenate_terms=True)

    env_cfg.observations = observations
    # Extra event terms, registered BEFORE construction. Required for any DR that touches a model
    # field: mjlab only allocates per-world memory for fields declared by a registered event
    # (@requires_model_fields -> sim.expand_model_fields), so calling e.g. dr.pd_gains on a built
    # env whose cfg never declared it writes into a shared array and silently does nothing.
    if extra_events:
        env_cfg.events.update(extra_events)
    return SafeVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device, render_mode=render_mode))


def build_cfg(args) -> dict:
    """Distillation train-cfg: student+teacher MLPModels + Distillation algorithm, mirroring the
    PPO actor architecture (so the teacher weights load)."""
    ppo = asdict(load_rl_cfg("Reflexive-Grasp-YCB-All"))
    model = {  # same architecture as the teacher's PPO actor -> teacher weights load, student mirrors it
        "class_name": "MLPModel",
        "hidden_dims": (256, 128, 64),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    }
    rnn = getattr(args, "rnn", None)
    if (
        rnn is not None
    ):  # match a recurrent teacher (e.g. ft_gru_project50): RNNModel core + MLP head
        model["class_name"] = "RNNModel"
        model["rnn_type"] = rnn
        model["rnn_hidden_dim"] = 256
        model["rnn_num_layers"] = 1
    cfg = dict(ppo)
    cfg.pop("actor", None)
    cfg.pop("critic", None)
    cfg["class_name"] = "DistillationRunner"
    # Deep-copy so student/teacher don't share the nested distribution_cfg dict (MLPModel pops
    # class_name out of it at build time, which would leave the second model's copy missing it).
    cfg["student"] = copy.deepcopy(model)
    cfg["teacher"] = copy.deepcopy(model)
    # The teacher's architecture is fixed by its checkpoint; the STUDENT's is free. --student-arch
    # mlp gives a feed-forward student (pair it with --student-history so it still sees the past)
    # while the teacher stays the RNN its weights were trained as -- setting --rnn none instead
    # would make BOTH an MLP and the GRU teacher checkpoint would fail to load.
    if getattr(args, "student_arch", None) == "mlp":
        cfg["student"] = {k: v for k, v in cfg["student"].items() if not k.startswith("rnn_")}
        cfg["student"]["class_name"] = "MLPModel"
    cfg["obs_groups"] = {"student": ("student",), "teacher": ("teacher",)}
    cfg["algorithm"] = {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,
        "learning_rate": 1.0e-3,
        "max_grad_norm": 1.0,
        "loss_type": "mse",
        "optimizer": "adam",
    }
    cfg["num_steps_per_env"] = 24
    cfg["max_iterations"] = args.iterations
    cfg["logger"] = args.logger
    cfg["wandb_project"] = args.wandb_project
    cfg["run_name"] = args.run_name
    cfg["experiment_name"] = "reflexive_grasp_distill"
    cfg["save_interval"] = 50
    if getattr(args, "belief_student", False):
        # Two-headed recurrent student (contact head off the shared latent) + the matching
        # Distillation subclass that adds the supervised contact-BCE. Teacher stays a plain RNNModel
        # so the trained 16-contact teacher weights still load. Requires --rnn (recurrent belief).
        cfg["student"]["class_name"] = (
            "blindgraspreflex.models.contact_belief_model:ContactBeliefModel"
        )
        cfg["algorithm"]["class_name"] = (
            "blindgraspreflex.models.contact_belief_model:ContactDistillation"
        )
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-run", default="logs/ycb_all_residual_fixed_pca")
    p.add_argument("--teacher-checkpoint", default=None, help="Else latest in --teacher-run.")
    p.add_argument("--num-envs", type=int, default=768)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--log-dir", default="logs/distill_fingertip")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
    p.add_argument("--wandb-project", default="reflexive-rl")
    p.add_argument("--run-name", default="distill_fingertip")
    p.add_argument(
        "--hand-mode",
        default="pca",
        choices=("joints", "pca"),
        help="Teacher's hand action space (must match the teacher checkpoint).",
    )
    p.add_argument(
        "--hand-pca-project",
        type=float,
        default=None,
        help="Teacher's joints-mode PCA projection alpha (e.g. 0.5 for ft_gru_project50).",
    )
    p.add_argument(
        "--rnn",
        choices=("lstm", "gru"),
        default=None,
        help="Use recurrent student+teacher to match a recurrent teacher (e.g. gru).",
    )
    p.add_argument(
        "--no-arm-residual",
        dest="arm_residual",
        action="store_false",
        help="Teacher was trained WITHOUT the 6-D palm residual (21-D action). Normally "
        "unnecessary: the action space is auto-detected from the teacher checkpoint.",
    )
    p.add_argument(
        "--disturbance",
        type=float,
        default=0.0,
        metavar="V",
        help="Peak per-axis linear velocity kick [m/s] applied to the object during "
        "DISTILLATION. The student only learns drop responses from states it "
        "actually visits, so a drop-aware teacher needs drops here too.",
    )
    p.add_argument(
        "--disturbance-angvel",
        type=float,
        default=0.8,
        metavar="W",
        help="(with --disturbance) peak per-axis angular kick [rad/s].",
    )
    p.add_argument(
        "--vanish-prob",
        type=float,
        default=0.0,
        metavar="P",
        help="Per-step probability of removing a lifted object mid-hold during "
        "distillation (episode continues) -- makes 'hand closed on nothing' common.",
    )
    p.add_argument(
        "--student-obs",
        default="fingertip",
        choices=("fingertip", "proprio", "m9", "pos"),
        help="Student obs: 'fingertip' (5 contacts + proprioception, 45-D), 'proprio' "
        "(hand_joint_pos + hand_tracking, 40-D -- no tactile, for deploy), or 'm9' "
        "(hand_joint_pos + the invariant contact metric, 40-D). 'm9' swaps the raw "
        "position error for a dimensionless torque ratio that carries no stiffness or "
        "velocity scale -- see blindgraspreflex/contact_metrics.py), or 'pos' (hand_joint_pos "
        "ONLY, no error term -- the ablation baseline).",
    )
    p.add_argument(
        "--arm-velocity",
        action="store_true",
        help="Distill under the Jacobian-pseudoinverse velocity arm controller (the deploy "
        "controller), so the student co-adapts to the deploy arm. Match the teacher.",
    )
    p.add_argument(
        "--hand-gate",
        action="store_true",
        help="Distill with the arm->hand grasp enable/disable gate on (match a gated teacher).",
    )
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override the episode length [s] to match the teacher's training.",
    )
    p.add_argument(
        "--no-reach-hand-pose",
        dest="reach_hand_pose",
        action="store_false",
        help="Match a teacher trained with the learned REACH pose off (fixed reach pose, "
        "action dim 27 not 47). Required for the fixed-reach joints teachers.",
    )
    p.set_defaults(reach_hand_pose=True)
    p.add_argument(
        "--teacher-contacts",
        action="store_true",
        help="Match a --teacher-contacts teacher: its obs group gets the full 16-link "
        "contact vector (teacher obs 59-D). The STUDENT still gets only 5 fingertips.",
    )
    p.add_argument(
        "--lift-height",
        type=float,
        default=0.0,
        metavar="M",
        help="Match the teacher's lift_height_offset (raised hold target), e.g. 0.3.",
    )
    p.add_argument(
        "--object-mass-range",
        default=None,
        metavar="LO,HI",
        help="Match the teacher's mass DR (kg), e.g. 0.03,0.30, so the student sees the "
        "same object-weight distribution.",
    )
    p.add_argument(
        "--object-scale-range",
        default=None,
        metavar="LO,HI",
        help="Match the teacher's size DR, e.g. 0.6,1.0 (5 per-world scale variants).",
    )
    p.add_argument(
        "--pca-latent-scale",
        type=float,
        default=1.0,
        metavar="S",
        help="Match a PCA teacher's latent range scale (e.g. 2.0). Ignored in joints mode.",
    )
    p.add_argument(
        "--no-success",
        action="store_true",
        help="Match a teacher trained with --no-success: drop the success reward AND the "
        "success termination so distillation episodes run the full length (no early 2 s "
        "reset) and the student clones the teacher's sustained hold.",
    )
    p.add_argument(
        "--hold-weight",
        type=float,
        default=None,
        metavar="W",
        help="Match the teacher's hold reward weight (e.g. 150). Cosmetic for behaviour "
        "cloning, but keeps the student env identical to the teacher's.",
    )
    p.add_argument(
        "--belief-student",
        action="store_true",
        help="Contact-belief (RMA-style) student: input is a proprioception HISTORY (see "
        "--history), a head reconstructs the teacher's 16 contacts (supervised BCE), and "
        "the policy acts on that contact-aware latent. Drops palm + measured contacts. "
        "Requires --rnn. Teacher must be a --teacher-contacts (16-contact) run.",
    )
    p.add_argument(
        "--student-arch",
        choices=("rnn", "mlp"),
        default="rnn",
        help="Student network. 'rnn' mirrors the teacher (default); 'mlp' is feed-forward "
        "-- pair with --student-history so it still sees the past.",
    )
    p.add_argument(
        "--student-history",
        type=int,
        default=0,
        metavar="N",
        help="Frame-stack the STUDENT's observation over the last N control steps "
        "(non-belief students). Obs width becomes N x base.",
    )
    p.add_argument(
        "--freeze-object-on-engage",
        action="store_true",
        help="Freeze the ESTIMATED object position at the instant the arm->hand gate engages, instead of tracking the object live for the whole episode. Models hardware occlusion: the hand closing on the object is what hides it from the camera, so the arm servos on the last good estimate from that moment. Must MATCH between teacher and student -- it changes the arm's setpoint, so a student distilled under a different setting learns from a teacher acting in a different world.",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="CKPT",
        help="Continue a distillation run: load student + teacher + optimiser AND keep "
        "the iteration counter, so checkpoint numbering and the W&B x-axis carry on "
        "across time-limited blocks. A student checkpoint carries the frozen teacher "
        "with it, so this replaces (not supplements) the --teacher-checkpoint load. "
        "Required to run a >8 h distillation on a preemptable queue.",
    )
    p.add_argument(
        "--history",
        type=int,
        default=5,
        metavar="N",
        help="(with --belief-student) frames of proprioception history fed to the belief "
        "encoder (last N steps of [joint_pos, target-joint_pos]).",
    )
    p.add_argument(
        "--contact-weight",
        type=float,
        default=1.0,
        metavar="W",
        help="(with --belief-student) weight of the contact-BCE auxiliary loss vs the "
        "behaviour-cloning loss.",
    )
    args = p.parse_args()
    if args.belief_student:
        import os as _os

        _os.environ["REFLEXIVE_CONTACT_WEIGHT"] = str(args.contact_weight)
        assert args.rnn is not None, (
            "--belief-student requires --rnn (the belief encoder is recurrent)"
        )
        assert args.teacher_contacts, (
            "--belief-student needs a 16-contact teacher: pass --teacher-contacts"
        )

    if args.object_scale_range is not None:
        import os

        import numpy as _np

        _lo, _hi = (float(x) for x in args.object_scale_range.split(","))
        os.environ["REFLEXIVE_OBJECT_SCALES"] = ",".join(
            f"{s:g}" for s in _np.linspace(_lo, _hi, 5)
        )
    _mass_range = None
    if args.object_mass_range is not None:
        _mlo, _mhi = (float(x) for x in args.object_mass_range.split(","))
        _mass_range = (_mlo, _mhi)

    register()
    # Auto-detect the teacher's action space from its checkpoint and match it. The action is
    # n_pre + num_hand, with n_pre = 7 with the palm residual and 1 without, so joints -> 27 or 21
    # and pca -> 12 or 6. Detecting beats trusting a flag: a mismatch does not warn, the teacher
    # checkpoint simply fails to load (or worse, loads into the wrong-sized head).
    from blindgraspreflex.checkpoint_introspect import latest_checkpoint, resolve_residual

    _ck = args.teacher_checkpoint or latest_checkpoint(args.teacher_run)
    args.arm_residual = resolve_residual(_ck, args.arm_residual, tag="distill")
    env = build_env(
        args.device,
        args.num_envs,
        hand_mode=args.hand_mode,
        hand_pca_project=args.hand_pca_project,
        student_obs=args.student_obs,
        disturbance=args.disturbance,
        disturbance_angvel=args.disturbance_angvel,
        vanish_prob=args.vanish_prob,
        arm_residual=args.arm_residual,
        arm_control="velocity" if args.arm_velocity else "impedance",
        hand_gate=args.hand_gate,
        episode_length_s=args.episode_length,
        reach_hand_pose=args.reach_hand_pose,
        teacher_contacts=args.teacher_contacts,
        lift_height=args.lift_height,
        mass_range=_mass_range,
        pca_latent_scale=args.pca_latent_scale,
        no_success=args.no_success,
        hold_weight=args.hold_weight,
        freeze_object_on_engage=args.freeze_object_on_engage,
        belief_student=args.belief_student,
        history=args.history,
        student_history=args.student_history,
        render_mode=None,
    )
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    print(
        f"[distill] student obs {tuple(obs['student'].shape)}  teacher obs {tuple(obs['teacher'].shape)}"
    )

    cfg = build_cfg(args)
    runner = DistillationRunner(env, cfg, log_dir=args.log_dir, device=args.device)

    teacher_ckpt = args.teacher_checkpoint or str(_checkpoints(Path(args.teacher_run))[-1])
    runner.load(teacher_ckpt)  # Distillation.load detects actor_state_dict -> loads the teacher
    print(
        f"[distill] teacher loaded from {teacher_ckpt}; teacher_loaded={runner.alg.teacher_loaded}"
    )

    if args.resume_from is not None:
        # A distillation checkpoint holds student + teacher + optimiser + iter, so loading it AFTER
        # the teacher restores the whole training state (and re-loads the same frozen teacher).
        # The counter is deliberately NOT reset: zeroing it restarts checkpoint numbering, so
        # `ls -v | tail -1` across blocks returns an EARLIER block's file and the next block
        # silently resumes from it, discarding a full block -- the bug scripts/train_grasp.py
        # documents on its own --resume-from.
        runner.load(args.resume_from)
        print(
            f"[distill] resumed from {args.resume_from} at iteration "
            f"{runner.current_learning_iteration} (counter preserved)"
        )

    # --iterations is how many MORE to run, matching scripts/train_grasp.py. A caller chaining
    # time-limited blocks toward a fixed total must pass the remainder itself (ablations/train.py
    # does; a study whose arms differ in total iterations is not sample-matched).
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
