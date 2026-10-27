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

### Primitive clutter grasping

The `Reflexive-Grasp-Primitive-Clutter` task places a central primitive target among seven
different primitive obstacles. Each reset packs them tightly around the target in a random
order and with random yaws: each obstacle takes the closest spot that leaves a 0-1 cm gap to
everything already placed, so 3-5 start within 1 cm of the target and the rest within 1 cm of
those. It uses a free moving RH-5 hand with no robot arm. The policy controls world-frame palm linear and angular velocity
(6 outputs) and all 20 hand joint targets directly; there is no hand PCA projection. Its actor
sees palm pose and twist, hand joint position and velocity, hand command error, palm-to-target
vector, fingertip contacts, and the target-relative centres
and bounding-sphere radii of the five nearest obstacles. Rewards encourage obstacle
clearance (each obstacle's push away from the target counts up to 15 cm), penalize target motion in the table plane, and encourage the palm to approach and
face the target. That reward is measured from a point on the palm face at the knuckles, along
the face normal, the palm's local +X; +Z runs from the wrist toward the fingers. Proximity and facing are combined
in one reward. Two contact terms count the palm and phalanx links touching an
obstacle: contact on the back or side of the hand is rewarded, and contact on the palm or finger
pads, which the hand needs for the target, is penalized at twice that rate. The palm velocity is integrated directly at the physics rate while
finger and object contacts remain simulated. The policy runs at 10 Hz, and the table-plane
penalty allows lifting.

```bash
uv run python -m blindgraspreflex.training.teacher \
  --task Reflexive-Grasp-Primitive-Clutter \
  --num-envs 512 --device cuda:0 --iterations 2000 \
  --logger wandb --wandb-project blindgraspreflex \
  --record-video --video-length 80 --video-interval 2400 \
  --log-dir logs/primitive_clutter_grasp
```

The repository's robot MJCF is required as for the other grasp tasks. Set
`BLINDGRASPREFLEX_MANIPULATOR_SOFTWARE` if it lives outside this checkout.

The same task also trains with SimToolReal's SAPG trainer, the reference repository's rl_games
fork, using 6 policy blocks, an LSTM actor and a central-value critic:

```bash
uv sync --extra cu128 --extra sapg && scripts/fetch_rl_games.sh
uv run python -m blindgraspreflex.training.sapg --run-name <name> --num-envs 24576
```

rl_games clips actions to [-1, 1], so this trainer maps them linearly onto the palm twist limits
and finger joint ranges. It caps the contact buffers at 128 contacts and 1024 constraint rows per world,
with convex-collision scratch sized separately at 64 slots per world. A run with 24,576
environments needs about 54 GB of GPU memory. On ORCD, `scripts/orcd_sync.sh` pushes
the code, `scripts/orcd_setup.sh` installs it once, and `RUN=<name> scripts/orcd_submit.sh` queues
one copy per GPU type. The first copy to start cancels the rest. The job resumes from its last
checkpoint and resubmits itself at the wall-clock limit.

Train a teacher from scratch:

```bash
uv run blindgraspreflex train teacher --config configs/local.toml
```

For a quick installation check:

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
output paths, viewer settings, and W&B account/project. The CLI only exposes useful per-run
overrides.

## Watch training in Viser

Start one persistent viewer alongside training. By default it recursively watches every student
run beneath the configured `logs` directory:

```bash
uv run blindgraspreflex serve --config configs/local.toml
```

Use `--kind teacher` to watch teacher runs, or `--checkpoint-root PATH` to restrict the server to
one run or another checkpoint tree. The Checkpoints tab refreshes automatically as training writes
new `model_*.pt` files. Select any checkpoint from its dropdown (or click **Use Latest**), and use
the Object dropdown to choose the YCB object. Loading a checkpoint replaces the running policy
without restarting the environment or server.

## Test one checkpoint in Viser

Load a teacher checkpoint and open the interactive test environment:

```bash
uv run blindgraspreflex play \
  --config configs/local.toml \
  --checkpoint logs/teacher/<teacher-run>/model_3500.pt \
  --kind teacher
```

For a distilled policy, use `--kind student`. Open the printed `http://localhost:8080` URL. The
Viser UI lets you select an object, reset episodes, inspect policy state, and play the policy
interactively. Port and device live in `[viewer]` in the config.

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
