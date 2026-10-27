"""Run the primitive clutter scene in Viser with an untrained random policy."""

from __future__ import annotations

import argparse

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer

from blindgraspreflex.tasks.grasping.clutter_task import make_clutter_task


class RandomActions:
    """Hold sampled actions briefly so the free hand moves smoothly."""

    def __init__(self, num_envs: int, device: str) -> None:
        self.actions = torch.zeros(num_envs, 26, device=device)
        self.steps = 0

    def __call__(self, _observations) -> torch.Tensor:
        if self.steps % 10 == 0:
            self.actions[:, :6] = torch.randn_like(self.actions[:, :6]) * 0.9
            self.actions[:, 6:] = torch.randn_like(self.actions[:, 6:]) * 1.2
        self.steps += 1
        return self.actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(make_clutter_task(num_envs=1), device=args.device))

    server = viser.ViserServer(port=args.port, label="primitive clutter — random actions")
    viewer = ViserPlayViewer(env, RandomActions(env.num_envs, args.device), viser_server=server)
    viewer.cfg.lookat = (0.55, 0.12, 0.16)
    viewer.cfg.distance = 1.4
    viewer.cfg.azimuth = 130.0
    viewer.cfg.elevation = 24.0
    print(f"Open http://localhost:{args.port} for the live random-action clutter scene", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
