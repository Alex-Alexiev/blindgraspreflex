"""Train the primitive clutter task with SimToolReal's SAPG trainer.

SimToolReal (and ~/brl/simtoolreal-flexiv) trains with the reference repository's own rl_games fork:
``a2c_continuous`` + SAPG (6 policy blocks sharing one network, leader-follower off-policy updates,
a per-block entropy bonus), an LSTM actor and an MLP central-value critic. ``train_config`` is that
resolved config. Only the env count, the run naming and the reward scale differ (see REWARD_SCALE).

The fork is a pinned sparse checkout (``scripts/fetch_rl_games.sh``), not the PyPI rl_games. Its
imports are the ``sapg`` extra (``uv sync --extra cu128 --extra sapg``).

    python -m blindgraspreflex.training.sapg --num-envs 24576 --run-name <name>
    python -m blindgraspreflex.training.sapg --task clutter-lift --reward-scale 0.01 \
        --reward-clamp 1000 --run-name <name>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

RL_GAMES_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "simtoolreal" / "rl_games"
NUM_SAPG_BLOCKS = 6  # launch_training.py num_blocks (the rl_games player also hardcodes 6)
HORIZON = 16
# The reference shapes its rewards by 0.01, which brings SimToolReal's ~1-100 per step to ~0.01-1.
# mjlab already scales the clutter rewards by the 0.1 s step, to ~0.01-0.2, so they go in unscaled.
# Values and advantages are normalised either way, and SAPG's entropy bonus acts on the loss.
REWARD_SCALE = 1.0
# A valid step pays at most ~2.5 in magnitude (target displacement before the 2 m instability
# reset), so the clamp only touches non-physical steps, which would poison the value normaliser.
REWARD_CLAMP = 5.0
TASKS = {
    "clutter": "Reflexive-Grasp-Primitive-Clutter",
    "clutter-lift": "Reflexive-Grasp-Primitive-Clutter-Lift",
    "clutter-lift-clear": "Reflexive-Grasp-Primitive-Clutter-Lift-Clear",
}
# Contact budget per world. MuJoCo Warp's convex-collision scratch costs ~23.5 KB per slot and is
# sized separately (NCCD, see mjlab_patches.cap_ccd_scratch); the task's 800 / 2400 would put it
# at ~19 MiB per world, ~470 GB at 24,576 worlds. Measured: ~20-28 contacts per world (pooled
# across worlds), at most ~50 in any one world, and at most 640 constraint rows (a hard per-world
# cap). nconmax stays at 128 although 64 holds that: the hand pressed palm down into the packed
# pile faulted MuJoCo Warp at 64 (illegal memory access, no overflow flag) and ran clean at 96,
# 128 and 400. The contact arrays themselves are small; NCCD keeps the scratch at 64 per world.
NCONMAX = 128
NCCD = 64
NJMAX = 1024


def use_reference_rl_games() -> None:
    """Put the reference's rl_games fork first on sys.path."""
    if not (RL_GAMES_ROOT / "rl_games" / "torch_runner.py").is_file():
        raise FileNotFoundError(
            f"rl_games fork not found under {RL_GAMES_ROOT}; run scripts/fetch_rl_games.sh"
        )
    if str(RL_GAMES_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_GAMES_ROOT))


def train_config(
    num_envs: int,
    iterations: int,
    run_name: str,
    train_dir: Path,
    device: str,
    seed: int = 0,
    save_frequency: int = 500,
    reward_scale: float = REWARD_SCALE,
) -> dict:
    """The reference's resolved rl_games config (pretrained_policy/config.yaml + launch_training.py)."""
    if num_envs % NUM_SAPG_BLOCKS:
        raise ValueError(
            f"num_envs must be divisible by {NUM_SAPG_BLOCKS} SAPG blocks, got {num_envs}"
        )
    batch = num_envs * HORIZON
    minibatch = batch // 4  # 98,304 = 24,576 x 16 / 4 in the reference
    mlp = {
        "units": [1024, 1024, 512, 512],
        "activation": "elu",
        "d2rl": False,
        "initializer": {"name": "default"},
        "regularizer": {"name": "None"},
    }
    return {
        "params": {
            "seed": seed,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": "coef_cond",
                    }
                },
                "mlp": mlp,
                "rnn": {
                    "name": "lstm",
                    "units": 1024,
                    "layers": 1,
                    "before_mlp": True,
                    "layer_norm": True,
                },
            },
            "load_checkpoint": False,
            "load_path": "",
            "config": {
                # rl_games parses the SAPG policy index from the "<int>_" prefix.
                "name": f"00_{run_name}",
                "full_experiment_name": f"00_{run_name}",
                "train_dir": str(train_dir),
                "device": device,
                "device_name": device,
                "env_name": "rlgpu",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": True,
                "normalize_input": True,
                "normalize_value": True,
                "normalize_advantage": True,
                "value_bootstrap": True,
                "reward_shaper": {"scale_value": reward_scale},
                "num_actors": num_envs,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 1e-4,
                "lr_schedule": "adaptive",
                "schedule_type": "standard",
                "kl_threshold": 0.016,
                "score_to_win": 1_000_000,
                "max_epochs": iterations,
                "max_frames": 100_000_000_000_000,
                "save_best_after": 100,
                # Checkpoint cadence only (reference 3000); does not affect training.
                "save_frequency": save_frequency,
                "print_stats": True,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.1,
                "minibatch_size": minibatch,
                "mini_epochs": 2,
                "critic_coef": 4.0,
                "clip_value": True,
                "horizon_length": HORIZON,
                "seq_length": 16,
                "bounds_loss_coef": 0.0001,
                "defer_summaries_sec": 5,
                "summaries_interval_sec_min": 5,
                "summaries_interval_sec_max": 300,
                "good_reset_boundary": 0,
                # SAPG (launch_training.py overrides).
                "use_others_experience": "lf",
                "off_policy_ratio": 1.0,
                "expl_type": "mixed_expl_learn_param",
                "expl_reward_coef_embd_size": 32,
                "expl_reward_coef_scale": 0.002,
                "expl_reward_type": "entropy",
                "expl_coef_block_size": num_envs // NUM_SAPG_BLOCKS,
                "player": {"deterministic": False, "games_num": 100000, "print_stats": False},
                "central_value_config": {
                    "minibatch_size": minibatch,
                    "mini_epochs": 2,
                    "learning_rate": 1e-4,
                    "kl_threshold": 0.016,
                    "clip_value": True,
                    "normalize_input": True,
                    "truncate_grads": True,
                    "network": {"name": "actor_critic", "central_value": True, "mlp": mlp},
                },
            },
        }
    }


class MjlabVecEnv:
    """rl_games vec-env over an mjlab env: ``obs`` = actor group, ``states`` = critic group.

    Rewards are returned raw (rl_games applies the reward shaper). ``infos`` carries
    ``time_outs`` for value bootstrapping and mjlab's per-term episode logs.
    """

    def __init__(self, env, reward_clamp: float = REWARD_CLAMP):
        import gym

        self.env = env  # rl_games reads self.env.device
        self.base = env.unwrapped
        self.num_envs = self.base.num_envs
        self.device = self.base.device
        self.reward_clamp = reward_clamp
        obs, _ = env.reset()
        # rsl-rl's init_at_random_ep_len, which the local teacher runs use: without it every world
        # times out on the same step for the whole run.
        self.base.episode_length_buf[:] = torch.randint_like(
            self.base.episode_length_buf, high=int(self.base.max_episode_length)
        )
        self.num_obs = obs["actor"].shape[-1]
        self.num_states = obs["critic"].shape[-1]
        self.num_actions = self.base.action_manager.total_action_dim
        self.action_space = gym.spaces.Box(-1.0, 1.0, (self.num_actions,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.num_obs,), dtype=np.float32)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (self.num_states,), dtype=np.float32)
        self._last_obs = self._wrap(obs)

    # -- rl_games IVecEnv ---------------------------------------------------------------------

    def get_number_of_agents(self) -> int:
        return 1

    def get_env_info(self) -> dict:
        return {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "state_space": self.state_space,
        }

    def set_train_info(self, env_frames, *args, **kwargs) -> None:
        pass

    def get_env_state(self):
        return None

    def set_env_state(self, env_state) -> None:
        pass

    def reset(self) -> dict:
        return self._last_obs

    def step(self, actions: torch.Tensor):
        import warp as wp

        obs, reward, terminated, time_outs, extras = self.env.step(actions)
        dones = terminated | time_outs
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        infos = {"time_outs": time_outs, **extras.get("log", {})}
        # Fraction of worlds whose contact or constraint buffers have overflowed (see NCONMAX).
        overflow = wp.to_torch(self.base.sim._wp_data.overflow)
        infos["physics_overflow"] = (overflow != 0).float().mean()
        return (
            self._wrap(obs),
            reward.clamp(-self.reward_clamp, self.reward_clamp),
            dones.long(),
            infos,
        )

    @staticmethod
    def _wrap(obs: dict) -> dict:
        return {
            "obs": torch.nan_to_num(obs["actor"], nan=0.0, posinf=0.0, neginf=0.0),
            "states": torch.nan_to_num(obs["critic"], nan=0.0, posinf=0.0, neginf=0.0),
        }


# The fork refreshes last/model.pth only every 200 epochs, and nn/ snapshots only when
# epoch / save_frequency is a perfect square. On a preemptible queue that loses up to 199 epochs
# per preemption, so the observer also saves, in the fork's own format, more often.
LAST_EVERY = 25  # epochs between last/model.pth refreshes
SNAPSHOT_EVERY = 1000  # ~330 MB each; the ORCD home quota is 200 GB


def make_observer(logger: str, project: str, run_name: str, video_dir: Path | None, config: dict):
    """rl_games observer: W&B (synced from TensorBoard), env logs, videos, frequent checkpoints,
    and a graceful stop on SIGTERM."""
    from rl_games.common.algo_observer import AlgoObserver

    class Observer(AlgoObserver):
        def __init__(self):
            super().__init__()
            self.scalars: dict[str, list[float]] = {}
            self.uploaded: set[str] = set()

        def before_init(self, base_name, cfg, experiment_name):
            # Stop gracefully on SIGTERM (the sbatch's wall-clock `timeout`, or a Slurm preemption):
            # finish the epoch, save last/model.pth and close the W&B run.
            import signal

            self.stop_requested = False

            def _on_sigterm(signum, frame):
                self.stop_requested = True
                print("[sapg] SIGTERM: stopping after this epoch", flush=True)

            signal.signal(signal.SIGTERM, _on_sigterm)
            if logger == "wandb":
                import wandb

                # A fixed id per run name, so a preempted-and-requeued job continues the same run.
                wandb.init(
                    project=project,
                    entity=os.environ.get("WANDB_ENTITY"),
                    name=run_name,
                    id=run_name.replace("/", "-"),
                    resume="allow",
                    sync_tensorboard=True,
                    config=config,
                )

        def after_init(self, algo):
            self.algo = algo
            self.writer = algo.writer

        def process_infos(self, infos, done_indices, **kwargs):
            # mjlab fills its Episode_* logs only on steps where some world resets, so average over
            # the steps that report them.
            for key, value in infos.items():
                if key == "time_outs":
                    continue
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.scalars.setdefault(key, []).append(float(value))
                elif isinstance(value, int | float):
                    self.scalars.setdefault(key, []).append(float(value))

        def save_checkpoints(self, epoch_num: int, force_last: bool = False) -> None:
            exp = Path(self.algo.experiment_dir)
            if force_last or epoch_num % LAST_EVERY == 0:
                last = exp / "last"
                last.mkdir(parents=True, exist_ok=True)
                if (last / "model.pth").exists():  # keep the previous one, as the fork does
                    os.replace(last / "model.pth", last / "model.pth.old")
                self.algo.save(str(last / "model"))
            if epoch_num % SNAPSHOT_EVERY == 0:
                (exp / "snapshots").mkdir(parents=True, exist_ok=True)
                self.algo.save(str(exp / "snapshots" / f"ep_{epoch_num:05d}"))

        def after_print_stats(self, frame, epoch_num, total_time):
            self.save_checkpoints(epoch_num)
            for key, values in self.scalars.items():
                self.writer.add_scalar(f"env/{key}", float(np.mean(values)), frame)
            self.scalars.clear()
            if logger == "wandb" and video_dir is not None and video_dir.is_dir():
                import wandb

                for path in sorted(video_dir.rglob("*.mp4")):
                    if path.name not in self.uploaded:
                        wandb.log(
                            {"video": wandb.Video(str(path), format="mp4"), "epoch": epoch_num}
                        )
                        self.uploaded.add(path.name)
            if self.stop_requested:
                if epoch_num % LAST_EVERY:  # (a multiple of LAST_EVERY was just saved above)
                    self.save_checkpoints(epoch_num, force_last=True)
                self.writer.flush()
                if logger == "wandb":
                    import wandb

                    wandb.finish(exit_code=0)
                print(f"[sapg] stopped cleanly at epoch {epoch_num}; resume from last/model.pth")
                sys.stdout.flush()
                # Skip interpreter teardown (renderer/EGL); the sbatch sees timeout's 124.
                os._exit(0)

    return Observer()


def _video_camera(env_cfg) -> None:
    """The local teacher's clutter camera (training.teacher --record-video)."""
    viewer = env_cfg.viewer
    viewer.origin_type = viewer.OriginType.WORLD
    viewer.lookat = (0.55, 0.12, 0.16)
    viewer.distance = 1.4
    viewer.azimuth = 130.0
    viewer.elevation = -30.0
    viewer.width, viewer.height = 640, 480
    viewer.max_extra_envs = 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--num-envs", type=int, default=24576, help="Multiple of 6 (SAPG blocks).")
    p.add_argument("--iterations", type=int, default=5000, help="rl_games epochs.")
    p.add_argument("--log-dir", default="logs/sapg", help="Runs go under <log-dir>/<run-name>.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
    p.add_argument("--wandb-project", default="blindgraspreflex")
    p.add_argument("--resume-from", default=None, help="Continue a run from its checkpoint.")
    p.add_argument(
        "--init-from", default=None, help="Warm-start weights and normalisers from a checkpoint."
    )
    p.add_argument("--video-every", type=int, default=200, help="Epochs between videos (0: off).")
    p.add_argument("--video-seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nconmax", type=int, default=NCONMAX)
    p.add_argument("--njmax", type=int, default=NJMAX)
    p.add_argument("--nccdmax", type=int, default=NCCD, help="Convex-collision scratch per world.")
    p.add_argument("--reward-scale", type=float, default=REWARD_SCALE)
    p.add_argument(
        "--task",
        choices=tuple(TASKS),
        default="clutter",
        help="clutter: decluttering rewards; clutter-lift: the grasp task's lift rewards; "
        "clutter-lift-clear: the lift rewards plus the decluttering obstacle clearance term.",
    )
    p.add_argument(
        "--reward-clamp",
        type=float,
        default=REWARD_CLAMP,
        help="Per-step bound on the raw reward. clutter-lift pays up to ~510 on a success step.",
    )
    args = p.parse_args()

    use_reference_rl_games()
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.wrappers import VideoRecorder
    from rl_games.torch_runner import Runner

    from blindgraspreflex.tasks.grasping.clutter_task import (
        make_clutter_lift_task,
        make_clutter_task,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_args = {
        **vars(args),
        "task": TASKS[args.task],
        "REFLEXIVE_TRACKING_QCMD": os.environ.get("REFLEXIVE_TRACKING_QCMD"),
    }
    (run_dir / "run_args.json").write_text(json.dumps(run_args, indent=2) + "\n")

    make_cfg = {
        "clutter": make_clutter_task,
        "clutter-lift": make_clutter_lift_task,
        "clutter-lift-clear": partial(make_clutter_lift_task, with_clearance=True),
    }[args.task]
    env_cfg = make_cfg(num_envs=args.num_envs)
    env_cfg.actions["grasp"].bounded = True
    from blindgraspreflex.mjlab_patches import cap_ccd_scratch

    cap_ccd_scratch(args.nccdmax)
    env_cfg.sim.nconmax = args.nconmax
    env_cfg.sim.njmax = args.njmax
    record = args.video_every > 0 and args.video_seconds > 0
    if record:
        _video_camera(env_cfg)
    try:
        env = ManagerBasedRlEnv(
            env_cfg, device=args.device, render_mode="rgb_array" if record else None
        )
    except Exception as exc:  # e.g. no EGL on a cluster node: train without videos
        if not record:
            raise
        print(f"[sapg] offscreen rendering unavailable ({type(exc).__name__}: {exc}); no videos")
        record = False
        env = ManagerBasedRlEnv(env_cfg, device=args.device, render_mode=None)
    video_dir = run_dir / "videos"
    if record:
        from blindgraspreflex.assets.mounts import hide_enclosure_geoms

        hide_enclosure_geoms(env.sim._mj_model)
        env.sim.expanded_fields.discard("geom_rgba")
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % (args.video_every * HORIZON) == 0,
            video_length=int(round(args.video_seconds / env.step_dt)),
            disable_logger=True,
        )
    vec_env = MjlabVecEnv(env, reward_clamp=args.reward_clamp)

    config = train_config(
        args.num_envs,
        args.iterations,
        args.run_name,
        run_dir,
        args.device,
        seed=args.seed,
        reward_scale=args.reward_scale,
    )
    config["params"]["config"]["run_args"] = run_args
    observer = make_observer(
        args.logger, args.wandb_project, args.run_name, video_dir if record else None, config
    )
    runner = Runner(observer)
    runner.load(config)
    runner.params["config"]["vec_env"] = vec_env
    runner.params["config"]["env_info"] = vec_env.get_env_info()
    if args.resume_from:
        runner.run({"train": True, "play": False, "checkpoint": args.resume_from, "sigma": None})
    elif args.init_from:
        runner.run(
            {
                "train": True,
                "play": False,
                "checkpoint": args.init_from,
                "checkpoint_load_mode": "weights",
                "sigma": None,
            }
        )
    else:
        runner.run({"train": True, "play": False, "checkpoint": None, "sigma": None})


if __name__ == "__main__":
    main()
