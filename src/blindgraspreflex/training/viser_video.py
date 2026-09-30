"""Automated Viser rollout rendering for training videos."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


class ViserVideoRenderer:
    """Render an mjlab environment through the same WebGL scene as the Viser player."""

    def __init__(self, env, *, width: int = 640, height: int = 480, env_idx: int = 0) -> None:
        import mujoco
        import viser
        from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS, sync_model_fields
        from mjlab.viewer.viser.scene import MjlabViserScene

        from blindgraspreflex.assets.mounts import hide_enclosure_geoms

        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome is None:
            raise RuntimeError("Viser video rendering requires Google Chrome or Chromium")

        self._base = env.unwrapped
        self.width = width
        self.height = height
        self.env_idx = env_idx
        self._server = viser.ViserServer(
            host="127.0.0.1", port=0, label="blindgraspreflex recorder", verbose=False
        )
        sim = self._base.sim
        self._mj_model = sim._mj_model
        self._mj_data = mujoco.MjData(self._mj_model)
        # Bake the selected world's randomized model fields into the host model first. Passing
        # the live batched model into MjlabViserScene would immediately restore the vendor cage's
        # alpha after we hide it.
        sync_model_fields(
            self._mj_model,
            sim.model,
            sim.expanded_fields & VIEWER_MODEL_FIELDS,
            env_idx,
        )
        hide_enclosure_geoms(self._mj_model)
        self._scene = MjlabViserScene(
            server=self._server,
            mj_model=self._mj_model,
            # Only one world is visible in a training video. Building WebGL handles for every
            # parallel training world makes a 256/2048-env scene unnecessarily enormous.
            num_envs=1,
        )
        self._scene.env_idx = env_idx
        self._scene.debug_visualization_enabled = False
        # The vendor MJCF contains a large wireframe work-cell enclosure. It obscures the grasp
        # and is expensive for software WebGL; the table and infinite ground grid remain visible.
        for (body_id, _group_id, _sub_idx), handle in self._scene._fixed_geom_handles.items():
            body_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # Scene entities are namespaced (normally ``robot/enclosure``).
            if body_name and body_name.rsplit("/", 1)[-1] == "enclosure":
                handle.visible = False
        self._server.scene.set_up_direction("+z")

        self._profile = tempfile.TemporaryDirectory(prefix="blindgraspreflex-viser-")
        url = f"http://127.0.0.1:{self._server.get_port()}"
        browser_args = [
            chrome,
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--window-size={width + 320},{height + 240}",
            f"--user-data-dir={self._profile.name}",
        ]
        using_display = bool(os.environ.get("DISPLAY"))
        if using_display:
            # A real display lets Chrome use the NVIDIA GPU. Keep the recorder window off-screen.
            browser_args.extend(["--window-position=-10000,-10000", f"--app={url}"])
        else:
            browser_args.extend(
                [
                    "--headless=new",
                    "--enable-webgl",
                    "--enable-unsafe-swiftshader",
                    "--use-angle=swiftshader",
                    url,
                ]
            )
        self._browser = subprocess.Popen(
            browser_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + (12.0 if using_display else 20.0)
        while not self._server.get_clients() and time.monotonic() < deadline:
            if self._browser.poll() is not None:
                break
            time.sleep(0.05)

        if not self._server.get_clients() and using_display:
            # Some remote desktop sessions advertise DISPLAY but reject a new Chrome window.
            # Retry with Chrome's self-contained WebGL software backend so video logging remains
            # reliable on SSH/headless training nodes.
            self._browser.terminate()
            self._browser.wait(timeout=5.0)
            self._browser = subprocess.Popen(
                browser_args[:5]
                + [
                    "--headless=new",
                    "--enable-webgl",
                    "--enable-unsafe-swiftshader",
                    "--use-angle=swiftshader",
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 20.0
            while not self._server.get_clients() and time.monotonic() < deadline:
                if self._browser.poll() is not None:
                    break
                time.sleep(0.05)

        if not self._server.get_clients():
            self.close()
            raise TimeoutError("Chrome did not connect to Viser within the recording timeout")

        self._client = next(iter(self._server.get_clients().values()))
        self._client.camera.position = (1.05, -0.82, 0.62)
        self._client.camera.look_at = (0.55, 0.08, 0.12)
        self._client.camera.up_direction = (0.0, 0.0, 1.0)
        self._client.camera.fov = np.deg2rad(48.0)
        self._update_scene()
        self._server.flush()
        # Give WebGL time to decode the robot and object meshes before the first capture.
        time.sleep(3.0)

    def render(self) -> np.ndarray:
        self._update_scene()
        self._server.flush()
        return self._client.get_render(
            self.height,
            self.width,
            transport_format="jpeg",
            timeout=30.0,
        )

    def _update_scene(self) -> None:
        """Copy one GPU world to CPU and update only that world's Viser handles."""
        import mujoco

        sim_data = self._base.sim.data
        self._mj_data.qpos[:] = sim_data.qpos[self.env_idx].detach().cpu().numpy()
        self._mj_data.qvel[:] = sim_data.qvel[self.env_idx].detach().cpu().numpy()
        mujoco.mj_forward(self._mj_model, self._mj_data)
        self._scene.update_from_mjdata(self._mj_data)

    def close(self) -> None:
        if getattr(self, "_browser", None) is not None:
            self._browser.terminate()
            try:
                self._browser.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._browser.kill()
                self._browser.wait(timeout=5.0)
        if getattr(self, "_server", None) is not None:
            self._server.stop()
        if getattr(self, "_profile", None) is not None:
            self._profile.cleanup()

    def __enter__(self) -> ViserVideoRenderer:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def capture_rollout(
    runner,
    env,
    *,
    device: str,
    seconds: float,
    observation_group: str,
    fps: int = 20,
) -> np.ndarray:
    """Roll out the current policy and return Viser frames as ``(T, H, W, 3)``."""

    base = env.unwrapped
    steps = max(1, int(round(seconds / base.step_dt)))
    frame_every = max(1, int(round(1.0 / (fps * base.step_dt))))
    policy = runner.get_inference_policy(device=device)
    observations = env.get_observations()
    observations = observations[0] if isinstance(observations, tuple) else observations
    frames: list[np.ndarray] = []

    with ViserVideoRenderer(env) as renderer, torch.inference_mode():
        for step in range(steps):
            policy_input = (
                observations[observation_group] if isinstance(observations, dict) else observations
            )
            transition = env.step(policy(policy_input))
            observations = transition[0]
            if step % frame_every == 0:
                frames.append(renderer.render())

    runner.alg.train_mode()
    if not frames:
        raise RuntimeError("Viser recorder produced no frames")
    return np.stack(frames)


def write_mp4(path: Path, frames: np.ndarray, *, fps: int = 12) -> None:
    """Write RGB frames to a browser-compatible H.264 video."""

    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=8,
    )
