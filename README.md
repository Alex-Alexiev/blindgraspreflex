# blindgraspreflex

Minimal training code for **Blind Grasp Reflex**: a privileged grasping teacher distilled into a
blind, proprioceptive policy that acts from hand joint positions and actuator tracking error.

This repository is intentionally limited to policy training. It does not contain robot deployment,
hardware communication, paper-generation scripts, historical experiments, or unrelated baselines.

## Canonical method

Training has two stages:

1. A privileged GRU teacher observes object-relative and distributed contact information.
2. A feed-forward student is distilled with DAgger from five frames of
   `[hand_joint_position, q_cmd - q]`.

The student controls the hand while a model-based phase controller moves the arm through reach,
lift, and retry phases. Hand commands are projected halfway toward a hand-specific grasp PCA
manifold. The repository bundles the fixed five-dimensional PCA vectors used by the Robotis 5F
hand; fitting new PCA models is deliberately out of scope for the initial release.

## Embodiments

Robot-specific information is provided by an embodiment plugin. Core training code does not assume
seven arm joints, twenty hand joints, or any particular MuJoCo body names.

An embodiment declares:

- its combined arm-hand MJCF;
- arm and hand joint names;
- the controlled palm site and contact links;
- home, reach, and open poses;
- actuator and velocity-field limits; and
- an optional hand PCA model.

Installed packages can register an embodiment with a Python entry point:

```toml
[project.entry-points."blindgraspreflex.embodiments"]
my_robot = "my_robot.blindgraspreflex:make_embodiment"
```

Validate one without starting training:

```bash
uv run blindgraspreflex embodiment validate robotis_5f
uv run blindgraspreflex embodiment list
```

The bundled `robotis_5f` metadata and PCA vectors are ready. Its combined robot MJCF is not copied
from the research repository: pass its location using `BLINDGRASPREFLEX_ROBOTIS_5F_MJCF` until the
asset redistribution boundary is finalized.

## Status

The new package and embodiment boundary are in place. The canonical environment and two training
stages are being extracted from the research repository next.

