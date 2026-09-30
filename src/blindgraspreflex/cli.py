"""Command-line interface for training and embodiment inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from blindgraspreflex.config import CANONICAL_METHOD
from blindgraspreflex.embodiments import available, load
from blindgraspreflex.embodiments.pca import load_hand_pca


def _list_embodiments() -> int:
    for name in available():
        print(name)
    return 0


def _validate_embodiment(name: str, *, skip_mjcf: bool) -> int:
    embodiment = load(name)
    errors = embodiment.validate(require_mjcf=not skip_mjcf)
    if errors:
        print(f"{name}: invalid")
        for error in errors:
            print(f"- {error}")
        return 1
    if embodiment.hand.pca_directory is not None:
        pca = load_hand_pca(embodiment.hand.pca_directory, embodiment.hand.joint_names)
        pca_summary = f", PCA={pca.latent_size}D"
    else:
        pca_summary = ", no PCA"
    obs_size = CANONICAL_METHOD.student_observation_size(embodiment.hand.dof)
    print(
        f"{name}: valid ({len(embodiment.arm.joint_names)} arm DOF, "
        f"{embodiment.hand.dof} hand DOF, student observation={obs_size}D{pca_summary})"
    )
    return 0


def _load_config(path: str) -> dict:
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _configure_process(config: dict) -> None:
    paths = config["paths"]
    wandb = config["wandb"]
    os.environ["BLINDGRASPREFLEX_MANIPULATOR_SOFTWARE"] = paths["manipulator_software"]
    os.environ["BLINDGRASPREFLEX_YCB_ROOT"] = paths["ycb"]
    os.environ["WANDB_ENTITY"] = wandb["entity"]
    # rsl-rl 5.4 uses this older spelling when it forwards the entity to wandb.init().
    os.environ["WANDB_USERNAME"] = wandb["entity"]
    os.environ["WANDB_MODE"] = wandb.get("mode", "online")
    # EGL keeps MuJoCo's off-screen training recorder independent of an X server.
    os.environ.setdefault("MUJOCO_GL", "egl")
    # The migrated task still reads this legacy spelling internally. It is intentionally set in
    # one compatibility boundary and will disappear when the task extraction is complete.
    os.environ["REFLEXIVE_TRACKING_QCMD"] = "1"
    # The package is imported to expose this CLI before the config is read. Refresh the public
    # path constant before importing any asset module, which binds it at module import time.
    import blindgraspreflex

    blindgraspreflex.MANIPULATOR_SOFTWARE_PATH = Path(paths["manipulator_software"])


def _run_module(main, arguments: list[str]) -> int:
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        main()
    finally:
        sys.argv = previous
    return 0


def _train(args) -> int:
    config = _load_config(args.config)
    _configure_process(config)
    training = config["training"]
    paths = config["paths"]
    wandb = config["wandb"]
    device = args.device or training["device"]
    iterations_key = f"{args.stage}_iterations"
    iterations = args.iterations or int(training[iterations_key])
    run_name = args.run_name or (
        f"blindgraspreflex-{args.stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = Path(paths["logs"]) / args.stage / run_name
    common = [
        "--num-envs",
        str(training["num_envs"]),
        "--iterations",
        str(iterations),
        "--device",
        device,
        "--logger",
        "wandb",
        "--wandb-project",
        wandb["project"],
    ]
    if args.stage == "teacher":
        from blindgraspreflex.training.teacher import main

        arguments = [
            *common,
            "--task",
            "Reflexive-Grasp-YCB-All",
            "--log-dir",
            str(log_dir),
            "--run-name",
            run_name,
            "--hand-mode",
            "joints",
            "--no-reach-hand-pose",
            "--rnn",
            "gru",
            "--arm-velocity",
            "--hand-gate",
            "--lift-height",
            "0.3",
            "--hand-pca-project",
            "0.5",
            "--object-mass-range",
            "0.03,0.20",
            "--object-scale-range",
            "0.7,1.0",
            "--episode-length",
            "10",
            "--physics-dt",
            "0.004",
            "--decimation",
            "21",
            "--hold-weight",
            "150",
            "--no-success",
            "--teacher-contacts",
            "--equal-contact-weights",
            "--palm-contact-bonus",
            "5",
            "--reach-dr",
            "--reach-dr-pos",
            "0.05,0.05,0.05",
            "--reach-dr-rpy",
            "10,10,10",
            "--disturbance",
            "--disturbance-vel",
            "0.35",
            "--disturbance-angvel",
            "0.9",
            "--hand-vf-dr",
            "0.6,2.0,0.15,0.45",
            "--hand-gain-dr",
            "0.1,1.5,0.3,1.5",
        ]
        return _run_module(main, arguments)

    if not args.teacher_checkpoint:
        raise SystemExit("student training requires --teacher-checkpoint")
    from blindgraspreflex.training.student import main

    arguments = [
        *common,
        "--teacher-checkpoint",
        args.teacher_checkpoint,
        "--log-dir",
        str(log_dir),
        "--run-name",
        run_name,
        "--hand-mode",
        "joints",
        "--no-reach-hand-pose",
        "--teacher-contacts",
        "--lift-height",
        "0.3",
        "--rnn",
        "gru",
        "--student-obs",
        "proprio",
        "--student-arch",
        "mlp",
        "--student-history",
        "5",
        "--arm-velocity",
        "--hand-gate",
        "--hand-pca-project",
        "0.5",
        "--object-mass-range",
        "0.03,0.20",
        "--object-scale-range",
        "0.7,1.0",
        "--episode-length",
        "10",
        "--no-success",
        "--hold-weight",
        "150",
        "--disturbance",
        "0.35",
        "--disturbance-angvel",
        "0.9",
    ]
    return _run_module(main, arguments)


def _play(args) -> int:
    config = _load_config(args.config)
    _configure_process(config)
    viewer = config["viewer"]
    paths = config["paths"]
    from blindgraspreflex.play import main

    if getattr(args, "checkpoint_root", None) is not None:
        source = ["--run-dir", args.checkpoint_root]
    elif getattr(args, "serve", False):
        source = ["--run-dir", str(Path(paths["logs"]) / args.kind)]
    else:
        source = ["--checkpoint", args.checkpoint]

    arguments = [
        "--task",
        "Reflexive-Grasp-YCB-All",
        *source,
        "--device",
        viewer["device"],
        "--port",
        str(viewer["port"]),
        "--hand-mode",
        "joints",
        "--hand-pca-project",
        "0.5",
        "--no-reach-hand-pose",
        "--arm-velocity",
        "--hand-gate",
        "--teacher-contacts",
        "--lift-height",
        "0.3",
        "--episode-length",
        "10",
        "--physics-dt",
        "0.004",
        "--decimation",
        "21",
        "--no-success",
    ]
    if args.kind == "student":
        arguments.extend(
            [
                "--student",
                "--student-obs",
                "proprio",
                "--student-arch",
                "mlp",
                "--student-history",
                "5",
                "--rnn",
                "gru",
            ]
        )
    else:
        arguments.extend(["--rnn", "gru"])
    return _run_module(main, arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blindgraspreflex")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train the canonical teacher or student.")
    train.add_argument("stage", choices=("teacher", "student"))
    train.add_argument("--config", default="configs/local.toml")
    train.add_argument("--teacher-checkpoint")
    train.add_argument("--iterations", type=int)
    train.add_argument("--device")
    train.add_argument("--run-name")

    play = commands.add_parser("play", help="Open a checkpoint in the Viser test environment.")
    play.add_argument("--config", default="configs/local.toml")
    play.add_argument("--checkpoint", required=True)
    play.add_argument("--kind", choices=("teacher", "student"), default="student")

    serve = commands.add_parser(
        "serve", help="Watch training checkpoints and select policies and objects in Viser."
    )
    serve.add_argument("--config", default="configs/local.toml")
    serve.add_argument("--kind", choices=("teacher", "student"), default="student")
    serve.add_argument(
        "--checkpoint-root",
        help="Directory to watch recursively (default: <logs>/<kind> from the config).",
    )
    serve.set_defaults(serve=True, checkpoint=None)

    method = commands.add_parser("method", help="Inspect the canonical q_cmd recipe.")
    method_commands = method.add_subparsers(dest="method_command", required=True)
    method_commands.add_parser("show")

    embodiment = commands.add_parser("embodiment", help="Inspect embodiment plugins.")
    embodiment_commands = embodiment.add_subparsers(dest="embodiment_command", required=True)
    embodiment_commands.add_parser("list")
    validate = embodiment_commands.add_parser("validate")
    validate.add_argument("name")
    validate.add_argument(
        "--skip-mjcf",
        action="store_true",
        help="Validate metadata and PCA without requiring the separately supplied robot MJCF.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        return _train(args)
    if args.command in ("play", "serve"):
        return _play(args)
    if args.command == "method":
        print(json.dumps(asdict(CANONICAL_METHOD), indent=2))
        return 0
    if args.embodiment_command == "list":
        return _list_embodiments()
    return _validate_embodiment(args.name, skip_mjcf=args.skip_mjcf)


if __name__ == "__main__":
    raise SystemExit(main())
