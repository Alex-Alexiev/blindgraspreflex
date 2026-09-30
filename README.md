# blindgraspreflex

Minimal training and evaluation code for **Blind Grasp Reflex**: train a privileged grasping
teacher, then distill it into a blind policy whose canonical observation is recent hand
proprioception plus actuator tracking error (`q_cmd - q`). This repository does not include robot
deployment, hardware communication, paper-generation scripts, or historical experiments.

## Method

Training has one public entry point and two stages:

1. A privileged GRU teacher observes object-relative and distributed contact information.
2. A feed-forward student is distilled with DAgger from five frames of
   `[hand_joint_position, q_cmd - q]`.

A model-based phase controller moves the arm through reach, lift, and retry phases while the
policy controls the hand. Hand commands are projected halfway toward a hand-specific grasp PCA
manifold. The fixed five-dimensional PCA model used for the Robotis 5F hand is bundled; fitting
new PCA models is out of scope.

## Install and configure

The working example currently expects a checkout of the robot MJCF assets and a prepared YCB
dataset. Clone those separately, then:

```bash
uv sync --extra cu128 --extra dev  # NVIDIA CUDA 12.8
# or: uv sync --extra cpu --extra dev
cp configs/example.toml configs/local.toml
```

Edit `configs/local.toml`. Set the two asset paths and put your W&B username or team in
`wandb.entity`. The local file is ignored by Git so machine-specific paths and account names are
not published. Use `mode = "offline"` if you do not want to upload a run.

## Train

Train a teacher from scratch:

```bash
uv run blindgraspreflex train teacher --config configs/local.toml
```

Each run records rollout videos at `training.video_every`, stores them beneath the configured log
directory, and uploads them to the same W&B run. For a quick installation check:

```bash
uv run blindgraspreflex train teacher \
  --config configs/local.toml --iterations 1 --run-name smoke-test
```

Then distill the canonical q_cmd student from a teacher checkpoint:

```bash
uv run blindgraspreflex train student \
  --config configs/local.toml \
  --teacher-checkpoint logs/teacher/<teacher-run>/model_3500.pt
```

The TOML file is the single place for device, parallel environment count, iteration counts,
video cadence, output paths, and W&B account/project. The CLI only exposes useful per-run
overrides.

## Test a checkpoint in Viser

Load a teacher checkpoint and open the interactive test environment:

```bash
uv run blindgraspreflex play \
  --config configs/local.toml \
  --checkpoint logs/teacher/<teacher-run>/model_3500.pt \
  --kind teacher
```

For a distilled policy, use `--kind student`. Open the printed `http://localhost:8080` URL. The
Viser UI lets you select an environment/object, reset episodes, inspect policy state, and play the
policy interactively. Port, device, and test-environment count live in `[viewer]` in the config.

## Add a robot or hand

Robot-specific metadata uses an embodiment plugin. An embodiment declares its combined arm-hand
MJCF, arm and hand joint names, controlled palm site and contact links, poses, limits, and optional
PCA model. A separate Python package can register one without editing this repository:

```toml
[project.entry-points."blindgraspreflex.embodiments"]
my_robot = "my_robot.blindgraspreflex:make_embodiment"
```

Inspect `src/blindgraspreflex/embodiments/robotis_5f.py`, then validate a plugin before training:

```bash
uv run blindgraspreflex embodiment list
uv run blindgraspreflex embodiment validate my_robot
```

The bundled `robotis_5f` plugin includes metadata and PCA vectors. Its combined MJCF remains an
external asset; point `BLINDGRASPREFLEX_ROBOTIS_5F_MJCF` at that file for standalone plugin
validation.

## Development

```bash
uv run pytest
uv run ruff check src tests
```

Licensed under the MIT License.
