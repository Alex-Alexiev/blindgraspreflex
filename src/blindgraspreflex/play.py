"""Visualise trained grasping checkpoints in a viser browser viewer.

Two modes:

* ``--checkpoint model_200.pt``  -- load one checkpoint.
* ``--run-dir logs/grasp_run4``  -- browse every checkpoint in the run from a dropdown in
  the viewer's Checkpoints tab, switching between them live.

The viser viewer renders one env at a time (chosen by ``env_idx``), so for the diverse-object
task (``Reflexive-Grasp-YCB``, the default) it spawns one env per YCB object -- the same set
training saw -- and an "environment" slider in the viewer flips between the objects live. Pass
``--task Reflexive-Grasp-Apple`` (or ``--num-envs 1``) for a single object. Runs on cuda:0 by
default so it does not contend with a training run on cuda:1.
"""

import argparse
import os
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
from rsl_rl.runners import DistillationRunner
from scipy.spatial.transform import Rotation

from blindgraspreflex.tasks.grasping.mjlab_task import (
    fingertip_contact_binary,
    grasp_matrix_metrics,
    net_wrench_components,
    net_wrench_reward,
    register,
)

_PLOT_WINDOW = 250  # policy steps kept in the scrolling grasp-score plot (~10 s at 25 Hz)


class GraspViewer(ViserPlayViewer):
    """ViserPlayViewer plus a live grasp-score plot, the latched phase, and the arm target.

    The grasp score drives a *Schmitt trigger*, not a 0.5 threshold: >=0.55 enters LIFT,
    <=0.45 leaves it, and the 0.45-0.55 deadband holds the previous phase. Plotting the score
    together with the resulting phase makes that hysteresis visible -- the score can dip below
    0.5 while the phase stays LIFT, which is correct, not a bug. Two triads on the palm body
    (``/palm_orient_target`` long, ``/palm_orient_actual`` short) show the commanded vs achieved
    attitude, and the ``/impedance_setpoint`` (orange) and ``/final_target`` (green) spheres show
    where the controller is driving the palm, so the phase decision is legible in the 3-D view.
    """

    def setup(self) -> None:
        super().setup()
        self._score_t: deque[int] = deque(maxlen=_PLOT_WINDOW)
        self._score_v: deque[float] = deque(maxlen=_PLOT_WINDOW)
        self._phase_v: deque[float] = deque(maxlen=_PLOT_WINDOW)
        # Object selector. Under per-world mesh variants each env holds a different YCB object
        # (the same set training saw), and the viser viewer renders one env at a time, so a
        # slider over env_idx is how you flip between objects live. env i -> object name comes
        # from the same largest-remainder allocation the variant entity uses.
        names = getattr(self, "_env_object_names", None)
        self._last_env_idx = self._scene.env_idx
        if self.env.num_envs > 1:
            with self._server.gui.add_folder("Object"):
                self._object_text = self._server.gui.add_text(
                    "object", names[self._scene.env_idx] if names else "?", disabled=True
                )
                self._env_slider = self._server.gui.add_slider(
                    "environment",
                    min=0,
                    max=self.env.num_envs - 1,
                    step=1,
                    initial_value=self._scene.env_idx,
                )

                @self._env_slider.on_update
                def _(_ev) -> None:
                    self._scene.env_idx = int(self._env_slider.value)
                    self._scene.request_update()
        else:
            self._object_text = None
        with self._server.gui.add_folder("Policy"):
            self._phase_text = self._server.gui.add_text("phase", "REACH", disabled=True)
            self._score_plot = self._server.gui.add_uplot(
                data=(np.array([0.0]), np.array([0.0]), np.array([0.0])),
                series=(
                    viser.uplot.Series(label="step"),
                    viser.uplot.Series(label="grasp score", stroke="#e04040", width=2),
                    viser.uplot.Series(label="phase (0 reach / 1 lift)", stroke="#4060e0", width=2),
                ),
                scales={"y": viser.uplot.Scale(range=(0.0, 1.0))},
                # Real thresholds, not 0.5: enter LIFT at 0.55, leave at 0.45, latched between.
                title="grasp score & phase  (LIFT >=0.55, REACH <=0.45, latched between)",
                height=170,
                aspect=2.0,
            )
        # Free-play: drag the LIFT target so the arm carries the grasped object to an x/y/z point.
        # Sets the phase controller's lift_override_pos, which replaces the fixed home retreat pose
        # in LIFT (the leash makes the arm pursue it smoothly). Persists across resets.
        self._lift_sliders: dict = {}
        if getattr(self, "_free_play", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            home = term.home_goal_pos.detach().cpu().numpy()
            ranges = {"x": (0.30, 0.75), "y": (-0.40, 0.40), "z": (0.0, 0.55)}
            with self._server.gui.add_folder("LIFT target (m) — move the grasped object"):
                for i, ax in enumerate(("x", "y", "z")):
                    lo, hi = ranges[ax]
                    self._lift_sliders[ax] = self._server.gui.add_slider(
                        ax,
                        min=lo,
                        max=hi,
                        step=0.005,
                        initial_value=float(min(max(home[i], lo), hi)),
                    )
                    self._lift_sliders[ax].on_update(lambda _ev: self._apply_lift_target())
            self._apply_lift_target()
        # --- ghost (believed) object -------------------------------------------------------
        # The policy never reads the object's true pose directly: every perception channel goes
        # through `_obj_pos_bias`, so believed = true + bias. These sliders ARE that bias, and the
        # ghost sphere is drawn where the policy thinks the object is. Slide it away from the real
        # apple and the arm follows the ghost -- which is the check that the perturbation reaches
        # the policy rather than only the scripted parts of the controller.
        self._ghost_sliders: dict = {}
        self._ghost_marker = None
        self._ghost_dist_text = None
        if getattr(self, "_ghost_object", False):
            with self._server.gui.add_folder("Ghost object — believed-pose offset (m)"):
                for ax in ("x", "y", "z"):
                    self._ghost_sliders[ax] = self._server.gui.add_slider(
                        ax, min=-0.20, max=0.20, step=0.005, initial_value=0.0
                    )
                    self._ghost_sliders[ax].on_update(lambda _ev: self._apply_ghost_bias())
                self._ghost_reset_btn = self._server.gui.add_button("reset offset to 0")
                self._ghost_reset_btn.on_click(lambda _ev: self._reset_ghost_bias())
                # Live proof the perturbation lands: the closest hand-link distance the POLICY
                # reads. With link-distance perception this is literally an input; with
                # palm_to_object it tracks the same believed point.
                self._ghost_dist_text = self._server.gui.add_text(
                    "believed |palm-obj|", initial_value="--", disabled=True
                )
            self._ghost_marker = self._server.scene.add_icosphere(
                "/ghost_object",
                radius=0.037,
                color=(120, 200, 255),
                opacity=0.45,
                position=(0.0, 0.0, 0.0),
            )
            self._apply_ghost_bias()

        # Orientation is shown as two co-located triads centred on the palm:
        #   /palm_orient_target  LONG axes  -- the target (flat hold) attitude.
        #   /palm_orient_actual  SHORT axes -- the palm's actual attitude.
        # When the triads align, the arm is holding the commanded orientation; a gap is the
        # tracking error. Orientation comes from the `palm` body -- the frame the impedance law
        # actually regulates (identical to grasp_site's). Note: at the OPEN pose every finger
        # phalanx shares this exact orientation, so the triad can *look* like it is on a finger;
        # close the hand and it stays with the palm while the fingers curl 178 deg away. The
        # triads are drawn at `palm_surface` (centre of the palm) rather than the `palm` body
        # origin (which sits back at the wrist) purely so they sit on the visible palm.
        robot = self.env.unwrapped.scene["robot"]
        self._palm_body_id = robot.find_bodies(["palm"])[0][0]
        # The new (v2) hand has no separate ``palm_surface`` body; fall back to ``palm``.
        try:
            self._palm_pos_id = robot.find_bodies(["palm_surface"])[0][0]
        except Exception:
            self._palm_pos_id = self._palm_body_id
        self._palm_target_frame = self._server.scene.add_frame(
            "/palm_orient_target",
            axes_length=0.13,
            axes_radius=0.006,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._palm_actual_frame = self._server.scene.add_frame(
            "/palm_orient_actual",
            axes_length=0.08,
            axes_radius=0.004,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        # Position markers (orientation-free): the tracked setpoint (orange, the control point
        # the impedance law drives) and the ultimate goal (green: object in REACH, home in LIFT).
        self._setpoint_marker = self._server.scene.add_icosphere(
            "/impedance_setpoint",
            radius=0.02,
            color=(230, 150, 40),
            position=(0.0, 0.0, 0.0),
        )
        self._final_target_marker = self._server.scene.add_icosphere(
            "/final_target",
            radius=0.025,
            color=(60, 200, 90),
            position=(0.0, 0.0, 0.0),
        )
        # Orientation-tracking error [deg] between the target and actual palm attitude, so the
        # "is it holding orientation?" question has a number, not just overlaid axes.
        self._orient_text = self._server.gui.add_text("palm orient err (deg)", "0", disabled=True)

        # Grasp-wrench panel: the net contact wrench the hand + gravity apply to the object (what
        # the net-wrench reward drives to zero) plus the grasp-matrix quality metrics, for the
        # displayed env. The orange arrow at the object COM is the net *force* imbalance -- it
        # shrinks to nothing as the grasp holds the object in balanced equilibrium. All read 0 until
        # the object is lifted and held (the rewards are gated on a real in-air grasp).
        with self._server.gui.add_folder("Grasp wrench"):
            self._nw_reward_text = self._server.gui.add_text(
                "net-wrench reward", "0", disabled=True
            )
            self._nw_force_text = self._server.gui.add_text(
                "net force / weight", "0", disabled=True
            )
            self._nw_torque_text = self._server.gui.add_text(
                "net torque / (w*L)", "0", disabled=True
            )
            self._qmsv_text = self._server.gui.add_text("Q_MSV (grasp matrix)", "0", disabled=True)
            self._qgii_text = self._server.gui.add_text("Q_GII (isotropy)", "0", disabled=True)
        self._wrench_arrow = self._server.scene.add_arrows(
            "/net_wrench_force",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(240, 120, 40),
            shaft_radius=0.006,
            head_radius=0.015,
            head_length=0.03,
            visible=False,
        )

        # Arm velocity-control tuning (arm_control="velocity"): live sliders for the Jacobian-
        # pseudoinverse velocity controller so the arm's reach/approach can be tuned against the
        # running hand policy. Each change calls term.set_velocity_params(...) immediately.
        _vterm = self.env.unwrapped.action_manager.get_term("grasp")
        if getattr(_vterm.cfg, "arm_control", "impedance") == "velocity":
            # cfg.arm_kp/arm_kd are None when the gains come from the selected ARM registry
            # rather than an explicit override, so read the RESOLVED per-arm tensors the term
            # built instead of assuming the cfg holds a tuple.
            def _gains(cfg_val, resolved):
                return (
                    tuple(cfg_val)
                    if cfg_val is not None
                    else tuple(float(x) for x in resolved.squeeze(0).tolist())
                )

            base_kp = _gains(_vterm.cfg.arm_kp, _vterm._arm_kp)
            base_kd = _gains(_vterm.cfg.arm_kd, _vterm._arm_kd)
            v = _vterm._vel
            with self._server.gui.add_folder("Arm velocity control"):
                s_vlin = self._server.gui.add_slider(
                    "vf_lin_speed", 0.05, 1.2, 0.01, v["vf_lin_speed"]
                )
                s_elin = self._server.gui.add_slider(
                    "vf_lin_eps", 0.005, 0.25, 0.005, v["vf_lin_eps"]
                )
                s_vang = self._server.gui.add_slider(
                    "vf_ang_speed", 0.2, 5.0, 0.1, v["vf_ang_speed"]
                )
                s_eang = self._server.gui.add_slider("vf_ang_eps", 0.02, 0.8, 0.01, v["vf_ang_eps"])
                s_damp = self._server.gui.add_slider(
                    "pinv_damping", 0.001, 0.3, 0.001, v["vel_damping"]
                )
                s_qdl = self._server.gui.add_slider("qd_limit", 0.5, 5.0, 0.1, v["qd_limit"])
                s_kp = self._server.gui.add_slider("kp_scale", 0.2, 3.0, 0.05, 1.0)
                s_kd = self._server.gui.add_slider("kd_scale", 0.2, 5.0, 0.05, 1.0)

            def _apply_vel(_ev=None):
                _vterm.set_velocity_params(
                    vf_lin_speed=s_vlin.value,
                    vf_lin_eps=s_elin.value,
                    vf_ang_speed=s_vang.value,
                    vf_ang_eps=s_eang.value,
                    vel_damping=s_damp.value,
                    qd_limit=s_qdl.value,
                    arm_kp=[k * s_kp.value for k in base_kp],
                    arm_kd=[k * s_kd.value for k in base_kd],
                )

            for _s in (s_vlin, s_elin, s_vang, s_eang, s_damp, s_qdl, s_kp, s_kd):
                _s.on_update(_apply_vel)
            _apply_vel()

        # Arm->hand gate radii, live. BatchedGraspPhase.update re-reads cfg.engage_radius and
        # cfg.release_radius every control step, so writing them here takes effect on the next
        # step -- no rebuild. Worth having as a slider because the gate is what decides when the
        # policy's fingers are allowed to drive at all, and its two radii are the difference
        # between a hand that closes early on a distant object and one that never engages.
        _gterm = self.env.unwrapped.action_manager.get_term("grasp")
        if getattr(_gterm.cfg, "hand_gate", False) or getattr(_gterm.cfg, "reach_hand_pose", False):
            _pcfg = _gterm.phase.cfg
            with self._server.gui.add_folder("Hand gate"):
                s_eng = self._server.gui.add_slider(
                    "engage_radius [m]", 0.02, 0.40, 0.005, float(_pcfg.engage_radius)
                )
                s_rel = self._server.gui.add_slider(
                    "release_radius [m]", 0.02, 0.60, 0.005, float(_pcfg.release_radius)
                )
                self._gate_text = self._server.gui.add_text("engaged", initial_value="-")
                self._release_text = self._server.gui.add_text(
                    "release rule",
                    initial_value=f"score <= {_pcfg.score_low:.2f}"
                    if getattr(_pcfg, "release_on_score", False)
                    else "distance > release_radius",
                )

            def _apply_gate(_ev=None):
                # Keep the Schmitt ordering: a release radius at or below the engage radius makes
                # the latch chatter at the boundary, which is the exact failure the hysteresis is
                # there to prevent. Push release up rather than silently accepting it.
                eng = float(s_eng.value)
                rel = max(float(s_rel.value), eng + 0.005)
                if rel != float(s_rel.value):
                    s_rel.value = rel
                _pcfg.engage_radius = eng
                _pcfg.release_radius = rel

            s_eng.on_update(_apply_gate)
            s_rel.on_update(_apply_gate)
            _apply_gate()

        # Manual debug mode: a slider per RAW action dim, all 0 by default, driving every env.
        # Raw = what the policy outputs: sigmoid maps grasp/hand to [0,1]; the residual is
        # clamped to [-1,1] then scaled to +-0.05 m. So all-zero is the neutral action (grasp
        # score 0.5 -> REACH, no residual, hand joints mid-range). Replaces the policy with a
        # reader of these sliders so you can pose the arm and hand by hand.
        if getattr(self, "_manual", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            specs = [("Grasp", "grasp_score", -6.0, 6.0)]
            if term._use_residual:
                # 6-D palm residual: 3 position (dims 1:4) + 3 orientation (dims 4:7).
                specs += [("Arm residual", f"pos_{ax}", -1.0, 1.0) for ax in ("x", "y", "z")]
                specs += [("Arm residual", f"rot_{ax}", -1.0, 1.0) for ax in ("x", "y", "z")]
            if term._pca is not None:
                # Hand block matches the policy's outputs: the GRASP latent, plus (when
                # reach_hand_pose) the REACH latent held during the approach.
                specs += [("Hand", f"grasp_z{i}", -6.0, 6.0) for i in range(term._num_hand)]
                if getattr(term, "_reach_hand", False):
                    specs += [("Hand", f"reach_z{i}", -6.0, 6.0) for i in range(term._num_hand)]
            else:
                from blindgraspreflex.assets import HAND_JOINT_NAMES

                specs += [("Hand", n, -6.0, 6.0) for n in HAND_JOINT_NAMES]
            self._action_sliders = [None] * len(specs)
            for group in ("Grasp", "Arm residual", "Hand"):
                with self._server.gui.add_folder(f"Actions: {group}"):
                    for i, (g, lbl, lo, hi) in enumerate(specs):
                        if g == group:
                            self._action_sliders[i] = self._server.gui.add_slider(
                                lbl, min=lo, max=hi, step=0.01, initial_value=0.0
                            )
            with self._server.gui.add_folder("Actions"):
                zero_btn = self._server.gui.add_button("zero all actions")

                @zero_btn.on_click
                def _(_ev) -> None:
                    for s in self._action_sliders:
                        s.value = 0.0

            self._setup_fingertip_sphere_gui()
            self.policy = self._manual_policy

        # REACH-pose tuning: six sliders set a SINGLE offset (position + orientation) of the palm
        # relative to the object; the real arm is driven there (hand open) so you see the actual
        # hand at that pose. Move a slider to an extreme you'd accept, click "save bound point";
        # the per-axis min/max envelope over saved points is the domain-randomisation range.
        self._sliders_off: dict = {}
        if getattr(self, "_tune_reach", False):
            term = self.env.unwrapped.action_manager.get_term("grasp")
            self._base_quat = term._home_site_quat.detach().cpu().numpy()  # flat hold attitude
            self._reach_target_frame = self._server.scene.add_frame(
                "/reach_target",
                axes_length=0.10,
                axes_radius=0.006,
                position=(0.0, 0.0, 0.0),
                wxyz=(1.0, 0.0, 0.0, 0.0),
            )
            with self._server.gui.add_folder("REACH offset: position (m)"):
                for ax in ("x", "y", "z"):
                    self._sliders_off[ax] = self._server.gui.add_slider(
                        ax, min=-0.12, max=0.12, step=0.005, initial_value=0.0
                    )
            with self._server.gui.add_folder("REACH offset: orientation (deg)"):
                for ax in ("roll", "pitch", "yaw"):
                    self._sliders_off[ax] = self._server.gui.add_slider(
                        ax, min=-90.0, max=90.0, step=1.0, initial_value=0.0
                    )
            # Single fixed pose readout: click to print the current 6-slider offset in a form
            # ready to paste as the default grasp pose (pre_grasp_offset + reach orientation).
            with self._server.gui.add_folder("REACH fixed pose"):
                self._pose_text = self._server.gui.add_text(
                    "offset", "(click to read)", disabled=True
                )
                pose_btn = self._server.gui.add_button("print current offset")
                pose_btn.on_click(lambda _ev: self._print_offset())
            self._saved_bounds: list[list[float]] = []
            with self._server.gui.add_folder("REACH bounds (for DR)"):
                save_btn = self._server.gui.add_button("save bound point")
                clear_btn = self._server.gui.add_button("clear bounds")
                self._bounds_text = self._server.gui.add_text(
                    "bounds", "(none saved)", disabled=True
                )
                save_btn.on_click(lambda _ev: self._save_bound_point())
                clear_btn.on_click(lambda _ev: self._clear_bounds())
            self.policy = self._tune_policy

    def _apply_lift_target(self) -> None:
        """Push the LIFT-target sliders into the controller's lift_override_pos (broadcasts to all envs)."""
        term = self.env.unwrapped.action_manager.get_term("grasp")
        xyz = [self._lift_sliders[a].value for a in ("x", "y", "z")]
        term.phase.lift_override_pos = torch.tensor(
            xyz, device=self.env.unwrapped.device, dtype=torch.float32
        )

    def _print_offset(self) -> None:
        """Print the current single slider offset, ready to paste as the fixed default pose."""
        s = self._sliders_off
        v = {a: s[a].value for a in ("x", "y", "z", "roll", "pitch", "yaw")}
        txt = (
            f"pos=({v['x']:+.3f}, {v['y']:+.3f}, {v['z']:+.3f}) m  "
            f"rpy=({v['roll']:+.1f}, {v['pitch']:+.1f}, {v['yaw']:+.1f}) deg"
        )
        self._pose_text.value = txt
        print("[tune-reach] fixed pose:", txt)

    def _offset_pose(self):
        """Current slider offset as (pos[3], quat_wxyz[4]); orientation = flat-hold x rpy."""
        s = self._sliders_off
        off = np.array([s["x"].value, s["y"].value, s["z"].value])
        rpy = np.radians([s["roll"].value, s["pitch"].value, s["yaw"].value])
        b = self._base_quat  # wxyz -> scipy xyzw
        q = (
            Rotation.from_quat([b[1], b[2], b[3], b[0]]) * Rotation.from_euler("xyz", rpy)
        ).as_quat()
        return off, np.array([q[3], q[0], q[1], q[2]])  # wxyz

    def _save_bound_point(self) -> None:
        s = self._sliders_off
        self._saved_bounds.append([s[a].value for a in ("x", "y", "z", "roll", "pitch", "yaw")])
        self._update_bounds_text()

    def _clear_bounds(self) -> None:
        self._saved_bounds = []
        self._update_bounds_text()

    def _update_bounds_text(self) -> None:
        if not self._saved_bounds:
            self._bounds_text.value = "(none saved)"
            return
        arr = np.array(self._saved_bounds)
        axes = ("x", "y", "z", "roll", "pitch", "yaw")
        units = ("m", "m", "m", "deg", "deg", "deg")
        lines = [
            f"{a}:[{arr[:, i].min():+.3g},{arr[:, i].max():+.3g}]{u}"
            for i, (a, u) in enumerate(zip(axes, units, strict=True))
        ]
        self._bounds_text.value = f"{len(self._saved_bounds)} pts  " + "  ".join(lines)
        print("[tune-reach] bounds:", "  ".join(lines))

    def _tune_policy(self, _obs=None) -> torch.Tensor:
        """Drive the palm to object + slider offset (hand open) so the real hand shows the pose."""
        term = self.env.unwrapped.action_manager.get_term("grasp")
        dev = self.env.unwrapped.device
        off, q_wxyz = self._offset_pose()
        obj = self.env.unwrapped.scene["object"].data.root_link_pos_w  # (N, 3)
        term._reach_override_pos = obj + torch.tensor(off, device=dev, dtype=torch.float32)
        term._reach_override_quat = (
            torch.tensor(q_wxyz, device=dev, dtype=torch.float32)
            .unsqueeze(0)
            .expand(obj.shape[0], 4)
            .contiguous()
        )
        a = torch.zeros(
            obj.shape[0], self.env.unwrapped.action_manager.total_action_dim, device=dev
        )
        a[:, term._n_pre :] = -5.0  # hand open (dims after grasp [+ residual])
        return a

    def _manual_policy(self, _obs=None) -> torch.Tensor:
        """Actions from the manual sliders, broadcast to every env (debug mode)."""
        dev = self.env.unwrapped.device
        vals = torch.tensor(
            [s.value for s in self._action_sliders], device=dev, dtype=torch.float32
        )
        return vals.unsqueeze(0).expand(self.env.num_envs, -1).contiguous()

    def _setup_fingertip_sphere_gui(self) -> None:
        """Fingertip contact-sphere readout + live tuning, using the EXACT training obs function.

        The readout is ``fingertip_sphere_contact_binary(env)`` -- the same sphere-gated,
        all-contacts (object/self/table) signal the actor observes during training. The
        offset/radius sliders edit the fingertip spheres live (via the env's cached tensors), so
        you can tune them and watch the training binary change, then paste the values back.
        """
        from blindgraspreflex.tasks.grasping.mjlab_task import _fingertip_sense_spheres

        env = self.env.unwrapped
        off, rad = _fingertip_sense_spheres(env)  # (1,5,3),(1,5) on device
        self._ft_off = off[0].detach().cpu().numpy().copy()
        self._ft_rad = rad[0].detach().cpu().numpy().copy()
        self._ft_labels = ("thumb", "index", "middle", "ring", "little")
        lbls = self._ft_labels
        with self._server.gui.add_folder("Fingertip sensing (training obs)"):
            self._ft_touch_text = self._server.gui.add_text(
                "touch (in-sphere)", "T:0 I:0 M:0 R:0 L:0", disabled=True
            )
            self._ft_sel = self._server.gui.add_dropdown("fingertip", lbls, initial_value=lbls[0])
            self._ft_ox = self._server.gui.add_slider(
                "offset x", -0.05, 0.05, 0.001, float(self._ft_off[0, 0])
            )
            self._ft_oy = self._server.gui.add_slider(
                "offset y", -0.05, 0.05, 0.001, float(self._ft_off[0, 1])
            )
            self._ft_oz = self._server.gui.add_slider(
                "offset z", -0.05, 0.05, 0.001, float(self._ft_off[0, 2])
            )
            self._ft_r = self._server.gui.add_slider(
                "radius", 0.002, 0.05, 0.001, float(self._ft_rad[0])
            )
            self._ft_snip = self._server.gui.add_markdown("")
        self._ft_sync = False

        def push() -> None:
            env._fingertip_sense_spheres_cache = (
                torch.tensor(self._ft_off, dtype=torch.float32, device=env.device).unsqueeze(0),
                torch.tensor(self._ft_rad, dtype=torch.float32, device=env.device).unsqueeze(0),
            )
            rows = ",\n".join(
                f"    ({self._ft_off[k, 0]:+.4f}, {self._ft_off[k, 1]:+.4f}, "
                f"{self._ft_off[k, 2]:+.4f}),  # {label}"
                for k, label in enumerate(lbls)
            )
            rad_s = ", ".join(f"{self._ft_rad[k]:.4f}" for k in range(len(lbls)))
            self._ft_snip.content = (
                "Paste into `rizon4_hand_v2.py`:\n```python\n"
                f"FINGERTIP_SENSE_OFFSET = (\n{rows}\n)\n"
                f"FINGERTIP_SENSE_RADIUS = ({rad_s})\n```"
            )

        @self._ft_sel.on_update
        def _(_=None) -> None:
            k = lbls.index(self._ft_sel.value)
            self._ft_sync = True
            self._ft_ox.value = float(self._ft_off[k, 0])
            self._ft_oy.value = float(self._ft_off[k, 1])
            self._ft_oz.value = float(self._ft_off[k, 2])
            self._ft_r.value = float(self._ft_rad[k])
            self._ft_sync = False

        def _on_off(_=None) -> None:
            if self._ft_sync:
                return
            k = lbls.index(self._ft_sel.value)
            self._ft_off[k] = (self._ft_ox.value, self._ft_oy.value, self._ft_oz.value)
            push()

        for s in (self._ft_ox, self._ft_oy, self._ft_oz):
            s.on_update(_on_off)

        @self._ft_r.on_update
        def _(_=None) -> None:
            if self._ft_sync:
                return
            k = lbls.index(self._ft_sel.value)
            self._ft_rad[k] = float(self._ft_r.value)
            push()

        push()

    def _apply_ghost_bias(self) -> None:
        """Push the slider values into the action term's perception bias, for every env."""
        if not self._ghost_sliders:
            return
        term = self.env.unwrapped.action_manager.get_term("grasp")
        b = torch.tensor(
            [[float(self._ghost_sliders[a].value) for a in ("x", "y", "z")]],
            dtype=torch.float32,
            device=term.device,
        )
        # _bias_override survives episode resets (the reset draw would otherwise zero the bias
        # the moment the episode ends), so the ghost stays where it was put.
        term._bias_override = b
        term._obj_pos_bias[:] = b

    def _reset_ghost_bias(self) -> None:
        for a in ("x", "y", "z"):
            self._ghost_sliders[a].value = 0.0
        self._apply_ghost_bias()

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        term = self.env.unwrapped.action_manager.get_term("grasp")
        idx = self._scene.env_idx  # the env (hence object) currently on screen
        if idx != self._last_env_idx:
            # Switched object: the plotted history belongs to the old env, so clear it and
            # relabel rather than splicing two objects' traces together.
            self._last_env_idx = idx
            self._score_t.clear()
            self._score_v.clear()
            self._phase_v.clear()
            names = getattr(self, "_env_object_names", None)
            if self._object_text is not None and names:
                self._object_text.value = names[idx]
        phase = float(term.phase.phase[idx].item())  # 0 reach, 1 lift
        self._score_t.append(self._step_count)
        self._score_v.append(float(term.grasp_score[idx]))
        self._phase_v.append(phase)
        t = np.array(self._score_t, dtype=float)
        self._score_plot.data = (t, np.array(self._score_v), np.array(self._phase_v))
        self._phase_text.value = "LIFT" if phase > 0.5 else "REACH"
        if getattr(self, "_gate_text", None) is not None:
            _eng = bool(term.phase.engaged[idx].item())
            # The distance the GATE sees, which is not always the true one: with
            # freeze_object_on_engage the estimate is latched at engage, so this keeps growing as
            # the palm lifts away from a stale belief -- which is exactly the behaviour that makes
            # the distance rule misfire, and worth being able to watch.
            _palm = self.env.unwrapped.scene["robot"].data.site_pos_w[idx, term._site_id]
            _obj = (
                self.env.unwrapped.scene["object"].data.root_link_pos_w[idx]
                + term._obj_pos_bias[idx]
            )
            if getattr(term.cfg, "freeze_object_on_engage", False) and bool(
                term._obj_pos_frozen_valid[idx].item()
            ):
                _obj = term._obj_pos_frozen[idx]
            _d = float(torch.linalg.norm(_palm - _obj).item())
            self._gate_text.value = f"{'ENGAGED' if _eng else 'open':8} d={_d:.3f} m"

        # Fingertip touch (the exact training observation): sphere-gated, all-contacts binary.
        if getattr(self, "_ft_touch_text", None) is not None:
            from blindgraspreflex.tasks.grasping.mjlab_task import fingertip_sphere_contact_binary

            tt = fingertip_sphere_contact_binary(self.env.unwrapped)[idx]
            self._ft_touch_text.value = " ".join(
                f"{label[0].upper()}:{int(tt[k])}" for k, label in enumerate(self._ft_labels)
            )

        if self._ghost_marker is not None:
            env_u = self.env.unwrapped
            true_pos = env_u.scene["object"].data.root_link_pos_w[idx].detach().cpu().numpy()
            bias = term._obj_pos_bias[idx].detach().cpu().numpy()
            self._ghost_marker.position = tuple(float(v) for v in (true_pos + bias))
            if self._ghost_dist_text is not None:
                palm = env_u.scene["robot"].data.body_link_pos_w[idx, self._palm_pos_id]
                palm = palm.detach().cpu().numpy()
                d_true = float(np.linalg.norm(palm - true_pos))
                d_ghost = float(np.linalg.norm(palm - (true_pos + bias)))
                self._ghost_dist_text.value = f"ghost {d_ghost:.3f}  (true {d_true:.3f})"

        # Position markers: the tracked setpoint (orange) and the ultimate goal (green).
        setp = term.impedance.target_pos[idx].detach().cpu().numpy()
        tgt_quat = term.impedance.target_quat[idx].detach().cpu().numpy()
        self._setpoint_marker.position = tuple(float(v) for v in setp)
        final = term.phase.base_target[idx].detach().cpu().numpy()
        self._final_target_marker.position = tuple(float(v) for v in final)

        # Orientation triads centred on the palm: target vs actual attitude, plus the error
        # angle. Orientation is the `palm` body's (== grasp_site's == the impedance control
        # frame); position is palm_surface's, purely to place the triads on the visible palm.
        robot = self.env.unwrapped.scene["robot"]
        palm_pos = robot.data.body_link_pos_w[idx, self._palm_pos_id].detach().cpu().numpy()
        palm_quat = robot.data.body_link_quat_w[idx, self._palm_body_id].detach().cpu().numpy()
        palm_xyz = tuple(float(v) for v in palm_pos)
        self._palm_target_frame.position = palm_xyz
        self._palm_target_frame.wxyz = tuple(float(v) for v in tgt_quat)
        self._palm_actual_frame.position = palm_xyz
        self._palm_actual_frame.wxyz = tuple(float(v) for v in palm_quat)
        dot = abs(float(np.dot(tgt_quat, palm_quat)))
        self._orient_text.value = f"{np.degrees(2 * np.arccos(min(1.0, dot))):.1f}"

        # Grasp-wrench readouts + net-force arrow for the displayed env. Best-effort: the sensor
        # data is only populated after a physics step, and the reward helpers touch the model
        # arrays, so any transient hiccup should not take down the viewer loop.
        try:
            base = self.env.unwrapped
            nf, _, fres, tres = net_wrench_components(base)
            nwr = net_wrench_reward(base)
            qmsv, _, qgii = grasp_matrix_metrics(base)
            self._nw_reward_text.value = f"{float(nwr[idx]):.3f}"
            self._nw_force_text.value = f"{float(fres[idx]):.3f}"
            self._nw_torque_text.value = f"{float(tres[idx]):.3f}"
            self._qmsv_text.value = f"{float(qmsv[idx]):.3f}"
            self._qgii_text.value = f"{float(qgii[idx]):.3f}"
            # Arrow = net contact-force imbalance at the object COM, length ~ residual (capped),
            # pointing the way the object is being net-pushed. Vanishes at a balanced hold.
            com = base.scene["object"].data.root_com_pos_w[idx].detach().cpu().numpy()
            fvec = nf[idx].detach().cpu().numpy()
            mag = float(np.linalg.norm(fvec))
            resid = float(fres[idx])
            if mag > 1e-6 and resid > 1e-3:
                end = com + (fvec / mag) * min(resid, 2.0) * 0.15
                self._wrench_arrow.points = np.array([[com, end]], dtype=np.float32)
                self._wrench_arrow.visible = True
            else:
                self._wrench_arrow.visible = False
        except Exception:
            pass

        # REACH tuning: place the /reach_target frame at the commanded offset pose for the viewed
        # object; the real palm is driven there by the tune policy so you see the actual hand.
        if getattr(self, "_tune_reach", False):
            obj = (
                self.env.unwrapped.scene["object"].data.root_link_pos_w[idx].detach().cpu().numpy()
            )
            off, q_wxyz = self._offset_pose()
            pos = obj + off
            self._reach_target_frame.position = (float(pos[0]), float(pos[1]), float(pos[2]))
            self._reach_target_frame.wxyz = tuple(float(v) for v in q_wxyz)


def _checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoints in a run directory, sorted by training step."""

    def step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return sorted(run_dir.glob("model_*.pt"), key=step)


def _resolve_object_names(obj_cfg, num_envs: int) -> list[str] | None:
    """Per-env object name, matching the variant entity's world->variant allocation.

    Returns None for a single-object (non-variant) task. Reproduces the same largest-remainder
    allocation ``VariantEntityCfg`` applies at sim init, so the viewer's labels line up with the
    meshes actually spawned in each env.
    """
    from mjlab.entity.variants import allocate_worlds

    variants = getattr(obj_cfg, "variants", None)
    if not variants:
        return None
    vnames = list(variants.keys())
    assign = getattr(obj_cfg, "assignment", None)
    if assign is None:
        alloc = allocate_worlds((1.0,) * len(vnames), num_envs)
    elif callable(assign):
        alloc = list(assign(num_envs))
    else:  # dict of per-variant weights
        alloc = allocate_worlds(tuple(assign.get(n, 1.0) for n in vnames), num_envs)
    return [vnames[i] for i in alloc]


def main() -> None:
    p = argparse.ArgumentParser()
    # Default to the diverse-object task so the viewer spawns the same YCB set training saw;
    # pass --task Reflexive-Grasp-Apple for the single-object runs.
    p.add_argument("--task", default="Reflexive-Grasp-YCB")
    p.add_argument("--checkpoint", default=None, help="Single checkpoint file.")
    p.add_argument("--run-dir", default=None, help="Directory of checkpoints to browse.")
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Envs to spawn. Default: one per object for a variant task (so every object "
        "appears), else 1. Use the 'environment' slider in the viewer to switch objects.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--squash", default="sigmoid", help="'sigmoid' for run 2+, 'clamp' for run 1.")
    p.add_argument(
        "--share",
        action="store_true",
        help="Request a public viser share URL so the viewer is reachable from anywhere (no port "
        "forwarding). Prints 'SHARE URL: <url>' once connected.",
    )
    p.add_argument(
        "--manual",
        action="store_true",
        help="Debug mode: drive the raw actions from GUI sliders (all 0 by default) instead of "
        "a policy. No checkpoint needed; the episode never auto-resets.",
    )
    p.add_argument(
        "--tune-reach",
        action="store_true",
        help="REACH-pose tuning: draw a cloud of candidate REACH palm poses around each object, "
        "with sliders for position/orientation bounds, to pick good domain-randomisation ranges. "
        "No checkpoint needed.",
    )
    # These MUST match the flags the checkpoint was trained with, or the obs/action dims (and
    # model class) won't line up and loading fails.
    p.add_argument(
        "--blind", action="store_true", help="Match a --blind run (no palm->object in actor)."
    )
    p.add_argument(
        "--teacher-contacts",
        action="store_true",
        help="Match a --teacher-contacts run: the actor gets the full 16-link contact "
        "vector instead of the 5 fingertip sensors (actor obs +11). Required to load "
        "such a checkpoint -- the obs dim won't match otherwise.",
    )
    p.add_argument(
        "--history",
        type=int,
        default=0,
        help="Match a --history N run (frame-stacked proprioception).",
    )
    p.add_argument(
        "--rnn", choices=("lstm", "gru"), default=None, help="Match a --rnn run (recurrent policy)."
    )
    p.add_argument(
        "--hand-mode",
        choices=("joints", "pca"),
        default=None,
        help="Match the run's hand action space.",
    )
    p.add_argument(
        "--decimation", type=int, default=None, help="Match the run's control decimation."
    )
    p.add_argument(
        "--physics-dt", type=float, default=None, help="Match the run's physics timestep."
    )
    p.add_argument(
        "--arm-residual",
        action="store_true",
        help="Match a --arm-residual run (6-D palm residual, no REACH randomisation).",
    )
    p.add_argument(
        "--arm-velocity",
        action="store_true",
        help="Use the sim-to-real deploy's Jacobian-pseudoinverse velocity arm controller instead of the training impedance controller (same policy, same phase target). Adds live tuning sliders for the velocity params in the viewer.",
    )
    p.add_argument(
        "--hand-pca-project",
        type=float,
        default=None,
        metavar="A",
        help="Match a --hand-pca-project run: blend the commanded hand pose toward its PCA-manifold projection by A (joints mode). REQUIRED to faithfully replay a run fine-tuned with projection.",
    )
    p.add_argument(
        "--pca-latent-scale",
        type=float,
        default=None,
        metavar="S",
        help="Match a --pca-latent-scale run: widen the PCA latent range on each axis by S. REQUIRED to faithfully replay a run trained with a scaled latent range.",
    )
    p.add_argument(
        "--lift-height",
        type=float,
        default=None,
        metavar="M",
        help="Match a --lift-height run: raise the LIFT/hold target (home_goal_pos) by M metres. REQUIRED to replay a run trained with a raised lift target.",
    )
    p.add_argument(
        "--object-mass-range",
        default=None,
        metavar="LO,HI",
        help="Add object mass DR (absolute uniform [LO,HI] kg per episode), to view a --object-mass-range run under varied masses. Optional (does not affect obs/action dims).",
    )
    p.add_argument(
        "--lift-wrench",
        type=float,
        default=None,
        metavar="N",
        help="Add the lift wrench disturbance (random 6-D wrench <= N at a random point once object z>0.2 m), to watch the policy hold through it. Optional (does not affect obs/action dims).",
    )
    p.add_argument(
        "--student",
        action="store_true",
        help="Load a distilled STUDENT checkpoint (fingertip-only obs) from a distill run; implies the teacher's config (PCA + arm-residual, 250Hz/12Hz).",
    )
    p.add_argument(
        "--limited-obs",
        action="store_true",
        help="Match a --limited-obs run: reduce the ACTOR obs to fingertip contacts + hand_joint_pos + hand_tracking (critic keeps full). REQUIRED to load a checkpoint trained with --limited-obs.",
    )
    p.add_argument(
        "--arm-joint-target",
        action="store_true",
        help="Match an --arm-joint-target run: the POLICY drives the arm joints "
        "(action [arm(7), hand(20)]), no phase-driven arm, no residual, no hand "
        "gate, and arm_joint_pos + arm_tracking added to the obs. Auto-enabled "
        "when the checkpoint's action is 27-D with a >=73-D obs.",
    )
    p.add_argument(
        "--arm-v-max",
        type=float,
        default=None,
        metavar="RAD_S",
        help="Match the run's --arm-v-max (joint-target arm velocity limit).",
    )
    p.add_argument(
        "--pregrasp-standoff",
        type=float,
        default=None,
        metavar="M",
        help="Seed each episode with the palm this far from the object, as training "
        "did. MUST match the run.",
    )
    p.add_argument(
        "--pregrasp-hand-open",
        type=float,
        default=None,
        metavar="F",
        help="Initial finger curl (default 0.1; the joint-target lineage trains 0.25). "
        "MUST match the run.",
    )
    p.add_argument(
        "--ghost-object",
        action="store_true",
        help="Add x/y/z sliders that offset the object position the POLICY believes "
        "(GraspAction._obj_pos_bias), and draw a translucent ghost sphere at that "
        "believed pose. The ghost is what every perception channel reads -- "
        "palm_to_object or the per-link distances, the pre-grasp seeding, and the "
        "phase REACH target -- so dragging it apart from the real object shows "
        "directly what the policy is chasing.",
    )
    p.add_argument(
        "--free-play",
        action="store_true",
        help="Never auto-reset (run until you press 'Reset Environment'), and add LIFT-target x/y/z sliders to drag the grasped object around.",
    )
    p.add_argument(
        "--student-obs",
        choices=("fingertip", "proprio", "pos", "m9"),
        default="fingertip",
        help="(with --student) student obs group: 'fingertip' (5 contacts + proprioception, "
        "45-D) or 'proprio' (hand_joint_pos + hand_tracking only, 40-D). Must match the "
        "distilled student (e.g. 'proprio' for distill_gru_proprio).",
    )
    p.add_argument(
        "--hand-gate",
        action="store_true",
        help="Arm-gates-hand: the hand policy drives the fingers only once the palm is "
        "within engage-radius of the object; otherwise the fingers hold the default "
        "pre-grasp pose. Fixes 'the hand just closes while standing still'.",
    )
    p.add_argument(
        "--no-reach-hand-pose",
        dest="reach_hand_pose",
        action="store_false",
        help="Disable the learned REACH-phase hand pose (the second 5-D PCA output, on "
        "by default in PCA mode). Pass this to play a checkpoint trained before "
        "reach_hand_pose -- its action dim is 5 smaller, so it won't load otherwise.",
    )
    p.set_defaults(reach_hand_pose=True)
    p.add_argument(
        "--student-arch",
        choices=("rnn", "mlp"),
        default="rnn",
        help="MUST match how the student was distilled, or its weights will not load.",
    )
    p.add_argument(
        "--student-history",
        type=int,
        default=0,
        metavar="N",
        help="Frame-stack the student obs over N control steps (feed-forward students).",
    )
    p.add_argument(
        "--const-reach",
        action="store_true",
        help="Match a --const-reach run: the REACH pose is an obs-independent learnable "
        "constant, so the actor is RNNModelConstReach (carries a reach_bias param). "
        "Required to load such a checkpoint -- plain RNNModel lacks reach_bias and won't load.",
    )
    p.add_argument(
        "--freeze-object-on-engage",
        action="store_true",
        help="Match a policy trained with the object estimate frozen at hand engage: "
        "the ARM servos on a snapshot taken when the fingers began closing instead of "
        "tracking the object live. Changes what the arm does, so a freeze-trained "
        "checkpoint viewed without it is not the policy in the file.",
    )
    p.add_argument(
        "--engage-radius",
        type=float,
        default=None,
        help="(with --hand-gate) palm->object distance [m] below which the hand engages "
        "the policy (Schmitt trigger; releases above --release-radius).",
    )
    p.add_argument(
        "--release-radius",
        type=float,
        default=None,
        help="(with --hand-gate) palm->object distance [m] above which the hand disengages "
        "back to the default pose.",
    )
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        metavar="S",
        help="Match a --episode-length run [s]. The play cfg defaults to 5 s, half the "
        "10 s most runs train with, which resets mid-approach and never lets a slow "
        "settler bank its 2 s hold.",
    )
    p.add_argument(
        "--no-success",
        action="store_true",
        help="Match a run trained without the success reward and early termination.",
    )
    args = p.parse_args()

    if not args.manual and not args.tune_reach and not args.checkpoint and not args.run_dir:
        p.error("pass --checkpoint or --run-dir (or --manual / --tune-reach)")

    register()
    # The env's action layout must match the checkpoint's, or the load fails on a size mismatch
    # after the env is already built. The checkpoint is ground truth; the flag is only a fallback.
    # Both the student branch (which used to hardcode True) and the --arm-residual flag go through
    # this, so a 21-D no-residual policy plays without any extra flags.
    from blindgraspreflex.checkpoint_introspect import latest_checkpoint, resolve_residual

    _ckpt = args.checkpoint or (latest_checkpoint(args.run_dir) if args.run_dir else None)
    # A student's pre-refactor default was True; a teacher's is whatever --arm-residual says.
    arm_residual = resolve_residual(_ckpt, True if args.student else args.arm_residual, tag="play")

    env_cfg = load_env_cfg(args.task, play=True)
    obj_cfg = env_cfg.scene.entities["object"]
    if args.num_envs is None:
        variants = getattr(obj_cfg, "variants", None)
        args.num_envs = len(variants) if variants else 1
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.actions["grasp"].squash = args.squash
    # Reproduce the training-time env config so the checkpoint's obs/action dims (and model)
    # match. Mirrors the same flags in train_grasp.py.
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.physics_dt is not None:
        env_cfg.sim.mujoco.timestep = args.physics_dt
    if args.hand_mode is not None:
        env_cfg.actions["grasp"].hand_mode = args.hand_mode
    env_cfg.actions["grasp"].reach_hand_pose = args.reach_hand_pose
    if arm_residual:
        env_cfg.actions["grasp"].arm_residual = True
        env_cfg.actions["grasp"].phase.reach_randomize = False
    if args.arm_velocity:
        env_cfg.actions["grasp"].arm_control = "velocity"
    if args.arm_joint_target:
        # Mirrors train_grasp.py's --arm-joint-target block exactly. The three switches come off
        # together: the policy owns the arm, so there is no scripted approach to gate the hand on,
        # no learned reach pose and no residual to add to a target nobody else sets.
        g = env_cfg.actions["grasp"]
        g.arm_control = "joint_target"
        g.hand_gate = False
        g.reach_hand_pose = False
        g.arm_residual = False
        if args.arm_v_max is not None:
            g.arm_v_max = args.arm_v_max
        from mjlab.managers.observation_manager import ObservationTermCfg as _OTC

        from blindgraspreflex.tasks.grasping.mjlab_task import arm_joint_pos, arm_tracking_error

        _arm = {
            "arm_joint_pos": _OTC(func=arm_joint_pos, params={}),
            "arm_tracking": _OTC(func=arm_tracking_error, params={}),
        }
        for _grp in ("actor", "critic"):
            if _grp in env_cfg.observations:
                env_cfg.observations[_grp].terms = {
                    **_arm,
                    **dict(env_cfg.observations[_grp].terms),
                }
        print("[play] arm_control = joint_target; arm obs added")
    if args.pregrasp_standoff is not None:
        from mjlab.managers.event_manager import EventTermCfg as _EvCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import reset_pregrasp_approach

        _pp = {"standoff": float(args.pregrasp_standoff)}
        if args.pregrasp_hand_open is not None:
            _pp["hand_open_frac"] = float(args.pregrasp_hand_open)
        env_cfg.events["reset_pregrasp"] = _EvCfg(
            func=reset_pregrasp_approach, mode="reset", params=_pp
        )
        print(f"[play] pre-grasp seeding {_pp}")
    if args.hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = args.hand_pca_project
    if args.freeze_object_on_engage:
        env_cfg.actions["grasp"].freeze_object_on_engage = True
        print("[play] object position FROZEN at hand engage")
    if args.pca_latent_scale is not None:
        env_cfg.actions["grasp"].pca_latent_scale = args.pca_latent_scale
    if args.lift_height is not None:
        env_cfg.actions["grasp"].lift_height_offset = args.lift_height
    if args.object_mass_range is not None:
        from mjlab.envs import mdp
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        _mlo, _mhi = (float(x) for x in args.object_mass_range.split(","))
        env_cfg.events["object_mass_dr"] = EventTermCfg(
            func=mdp.dr.body_mass,
            mode="reset",
            params={
                "ranges": (_mlo, _mhi),
                "operation": "abs",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
    if args.lift_wrench is not None:
        from mjlab.managers.event_manager import EventTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import object_high_wrench

        env_cfg.events["object_high_wrench"] = EventTermCfg(
            func=object_high_wrench,
            mode="step",
            params={
                "z_thresh": 0.2,
                "max_force": args.lift_wrench,
                "max_torque": 0.1,
                "offset_range": 0.05,
            },
        )
    if args.limited_obs and not args.student:
        # Reduce the actor obs to the limited set the checkpoint was trained on (critic keeps full).
        actor = env_cfg.observations["actor"].terms
        env_cfg.observations["actor"].terms = {
            "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
            "hand_joint_pos": actor["hand_joint_pos"],
            "hand_tracking": actor["hand_tracking"],
        }
    if args.blind:
        env_cfg.observations["actor"].terms.pop("palm_to_object", None)
    if args.teacher_contacts:
        from blindgraspreflex.tasks.grasping.mjlab_task import contact_binary

        env_cfg.observations["actor"].terms["contacts"] = ObservationTermCfg(
            func=contact_binary, params={}
        )
    if args.history > 0:
        for group in ("actor", "critic"):
            for t in ("hand_joint_pos", "hand_tracking"):
                if t in env_cfg.observations[group].terms:
                    env_cfg.observations[group].terms[t].history_length = args.history
    if args.student:
        # Distilled student: same env as the teacher (PCA + residual, 250Hz/12Hz), but the obs
        # are split into teacher (full) + student (fingertip contacts + proprioception) groups so
        # the DistillationRunner can load and run the student policy.
        env_cfg.actions["grasp"].hand_mode = args.hand_mode or "pca"
        env_cfg.actions["grasp"].arm_residual = arm_residual
        env_cfg.actions["grasp"].phase.reach_randomize = False
        env_cfg.sim.mujoco.timestep = 0.004
        env_cfg.decimation = 21
        teacher_terms = dict(env_cfg.observations["actor"].terms)
        # Match distill_grasp.build_env exactly (same term order) so the student weights line up.
        student_terms = {
            "hand_joint_pos": teacher_terms["hand_joint_pos"],
            "hand_tracking": teacher_terms["hand_tracking"],
        }
        if args.student_obs == "pos":
            student_terms.pop("hand_tracking", None)  # positions only, no error term
        elif args.student_obs == "m9":
            import os as _os

            _os.environ["REFLEXIVE_M9_OBS"] = "1"  # the estimator is opt-in; enable it
            from blindgraspreflex.tasks.grasping.mjlab_task import hand_contact_metric

            student_terms["hand_tracking"] = ObservationTermCfg(func=hand_contact_metric, params={})
        elif args.student_obs == "fingertip":
            student_terms = {
                "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
                **student_terms,
            }
        if args.student_history and args.student_history > 1:
            import copy as _copy

            student_terms = {k: _copy.copy(v) for k, v in student_terms.items()}
            for _v in student_terms.values():
                _v.history_length = args.student_history
        env_cfg.observations = {
            "teacher": ObservationGroupCfg(terms=teacher_terms, concatenate_terms=True),
            "student": ObservationGroupCfg(terms=student_terms, concatenate_terms=True),
        }
    if args.episode_length is not None:
        env_cfg.episode_length_s = args.episode_length
    if args.no_success:
        env_cfg.rewards.pop("success", None)
        env_cfg.terminations.pop("success", None)
    if args.manual or args.tune_reach or args.free_play:
        # Don't auto-reset while hand-driving the sliders / tuning bounds / free-playing: drop the
        # success termination and make the timeout effectively never fire, so the episode runs
        # until the viewer's "Reset Environment" button. The instability guard stays as a safety.
        env_cfg.terminations.pop("success", None)
        env_cfg.episode_length_s = 1.0e6
    env_object_names = _resolve_object_names(obj_cfg, args.num_envs)
    rl_cfg = load_rl_cfg(args.task)
    if args.rnn is not None:
        for m in (rl_cfg.actor, rl_cfg.critic):
            m.class_name = "RNNModel"
            m.rnn_type = args.rnn
            m.rnn_hidden_dim = 256
            m.rnn_num_layers = 1
    if args.const_reach:
        # The actor's reach mean is an obs-independent constant (reach_bias). Build the matching
        # class + tell it how many trailing action dims are the constant (reach block = hand DOF).
        os.environ["REFLEXIVE_CONST_REACH_DIM"] = str(20 if args.hand_mode == "joints" else 5)
        rl_cfg.actor.class_name = "blindgraspreflex.models.const_reach_model:RNNModelConstReach"

    if args.hand_gate:
        env_cfg.actions["grasp"].hand_gate = True
        if args.engage_radius is not None:
            env_cfg.actions["grasp"].phase.engage_radius = args.engage_radius
        if args.release_radius is not None:
            env_cfg.actions["grasp"].phase.release_radius = args.release_radius

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))

    runner = None

    def load(path: str):
        nonlocal runner
        if runner is None:
            if args.student:
                from types import SimpleNamespace

                from blindgraspreflex.training.student import build_cfg

                cfg = build_cfg(
                    SimpleNamespace(
                        iterations=1,
                        logger="tensorboard",
                        wandb_project="x",
                        run_name="play",
                        rnn=args.rnn,
                        student_arch=args.student_arch,
                    )
                )
                runner = DistillationRunner(env, cfg, None, args.device)
            else:
                runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), None, args.device)
        runner.load(path)  # distill checkpoint carries student+teacher; PPO checkpoint is actor
        return runner.get_inference_policy(device=args.device)

    server = viser.ViserServer(port=args.port, label="reflexive grasp - play")
    if args.share:
        share_url = server.request_share_url(verbose=False)
        print(f"[play] SHARE URL: {share_url}", flush=True)
    ckpt_mgr = None

    if args.manual or args.tune_reach:
        # Placeholder policy; GraspViewer.setup replaces self.policy (slider reader / hold action).
        action_dim = env.unwrapped.action_manager.total_action_dim
        policy = lambda obs: torch.zeros(env.num_envs, action_dim, device=args.device)  # noqa: E731
        mode = "TUNE-REACH" if args.tune_reach else "MANUAL"
        print(f"[play] {mode} mode: no checkpoint; use the viewer controls.")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        ckpts = _checkpoints(run_dir)
        if not ckpts:
            p.error(f"no model_*.pt in {run_dir}")
        latest = ckpts[-1]

        def fetch_available() -> list[tuple[str, str]]:
            found = _checkpoints(run_dir)
            now = int(os.path.getmtime(max(found, key=os.path.getmtime)))
            # Newest first, each labelled with how long ago it was written. Re-scans every
            # call so checkpoints written after launch (a live training run) appear.
            return [
                (c.name, format_time_ago(now - int(os.path.getmtime(c)))) for c in reversed(found)
            ]

        ckpt_mgr = CheckpointManager(
            current_name=latest.name,
            fetch_available=fetch_available,
            load_checkpoint=lambda name: load(str(run_dir / name)),
            run_name=run_dir.name,
        )
        policy = load(str(latest))
        print(f"[play] browsing {len(ckpts)} checkpoints in {run_dir}; latest {latest.name}")
    else:
        policy = load(args.checkpoint)
        print(f"[play] checkpoint {args.checkpoint}")

    if env_object_names is not None:
        shown = ", ".join(dict.fromkeys(env_object_names))
        print(f"[play] {args.num_envs} envs across {len(set(env_object_names))} objects: {shown}")
        print("[play] use the 'environment' slider in the viewer to switch object")

    print(f"[play] open http://localhost:{args.port}")
    viewer = GraspViewer(env, policy, viser_server=server, checkpoint_manager=ckpt_mgr)
    viewer._env_object_names = env_object_names
    viewer._manual = args.manual
    viewer._tune_reach = args.tune_reach
    viewer._free_play = args.free_play
    viewer._ghost_object = args.ghost_object
    viewer.run()


if __name__ == "__main__":
    main()
