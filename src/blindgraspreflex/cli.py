"""Command-line interface for training and embodiment inspection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blindgraspreflex")
    commands = parser.add_subparsers(dest="command", required=True)

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
    if args.command == "method":
        print(json.dumps(asdict(CANONICAL_METHOD), indent=2))
        return 0
    if args.embodiment_command == "list":
        return _list_embodiments()
    return _validate_embodiment(args.name, skip_mjcf=args.skip_mjcf)


if __name__ == "__main__":
    raise SystemExit(main())

