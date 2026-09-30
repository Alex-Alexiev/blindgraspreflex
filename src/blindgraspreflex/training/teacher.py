"""Train the reflexive grasping task. Registers the task, then runs rsl_rl PPO."""

import argparse
import math
import os
import pathlib
from dataclasses import asdict

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from blindgraspreflex.tasks.grasping.mjlab_task import register


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Sanitise obs/rewards so an occasional physics blow-up resets the world instead of
    crashing the whole run.

    The task's ``instability`` termination resets diverged worlds (and the env recomputes the
    observation post-reset), but the reward is computed on the *pre*-reset state -- so a world
    that jumps straight to NaN in one step would still trip rsl_rl's ``check_nan`` on the reward.
    ``nan_to_num`` turns that single bad step into a finite (zero) one; the termination then
    resets the world on the same or next step.
    """

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        for key in list(obs.keys()):
            obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, rew, dones, extras


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Reflexive-Grasp-Apple")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--log-dir", default="logs/reflexive_grasp")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="wandb")
    p.add_argument("--wandb-project", default="reflexive-rl")
    p.add_argument("--run-name", default="")
    p.add_argument("--video", action="store_true", help="Record rollout videos during training.")
    p.add_argument("--video-interval", type=int, default=100, help="Iterations between videos.")
    p.add_argument("--video-length", type=int, default=150, help="Frames per video (~1 episode).")
    p.add_argument(
        "--contact-weight", type=float, default=None, help="Override contact reward weight."
    )
    p.add_argument(
        "--height-weight", type=float, default=None, help="Override height reward weight."
    )
    p.add_argument(
        "--hold-weight",
        type=float,
        default=None,
        help="Override the hold-at-goal reward weight (task default 50; e.g. 150 to triple it).",
    )
    p.add_argument(
        "--equal-contact-weights",
        action="store_true",
        help="Contact-coverage reward weights every hand link equally (1.0) instead of the default "
        "IsaacLab per-link weights, then adds --palm-contact-bonus to the palm link.",
    )
    p.add_argument(
        "--palm-contact-bonus",
        type=float,
        default=5.0,
        metavar="B",
        help="(with --equal-contact-weights) extra weight ADDED to the palm contact (default 5 -> "
        "palm 6, all others 1), to push for seating the object against the palm.",
    )
    p.add_argument(
        "--palm-frame-residual",
        action="store_true",
        help="Apply the policy's 3-D position residual in the PALM frame (rotated by the reach "
        "attitude) instead of the world frame. Changes the action semantics -> needs its own "
        "teacher+student; the deployed world-frame policies are unaffected (default off).",
    )
    p.add_argument(
        "--score-honesty",
        type=float,
        default=0.0,
        metavar="W",
        help="Reward weight for grasp_score being TRUE (high while the object is in the hand, low "
        "when it is not). 0 = off. The score is otherwise only shaped indirectly through the "
        "REACH/LIFT latch and ends up RISING when the object is lost (measured). Uses the "
        "palm->object distance, which is privileged (training-only) -- the policy still has to "
        "infer the drop from proprioception + touch.",
    )
    p.add_argument(
        "--score-honesty-radius",
        type=float,
        default=0.15,
        metavar="R",
        help="(with --score-honesty) palm->object distance [m] below which the object counts as "
        "in the hand.",
    )
    p.add_argument(
        "--hand-vf-dr",
        default=None,
        metavar="VLO,VHI,SLO,SHI",
        help="Per-episode DR of the hand velocity field: v_max in [VLO,VHI] rad/s and slowdown "
        "distance in [SLO,SHI] rad. e.g. '0.6,2.0,0.15,0.45', which spans BOTH the sim default "
        "(0.8/0.35) and the deployed low_level values (1.8/0.2).",
    )
    p.add_argument(
        "--hand-gain-dr",
        default=None,
        metavar="KPLO,KPHI,KDLO,KDHI",
        help="Per-episode DR of the hand actuator PD gains, as SCALE factors on the trained "
        "defaults ([6,5,3,2]/[0.25,0.20,0.12,0.08] by phalanx level). e.g. '0.1,1.5,0.3,1.5' -- "
        "the low end reaches the deployed hand_kp=0.5, which is 0.08-0.25x these.",
    )
    p.add_argument(
        "--vanish-prob",
        type=float,
        default=0.0,
        metavar="P",
        help="Per-step probability of REMOVING a lifted object from the hand mid-hold (teleported "
        "away, episode continues). Makes the 'hand closed on nothing' state common in training -- "
        "without it that state is rare, the score-honesty reward is satisfied by the approach-vs-"
        "hold asymmetry alone, and the drop response does not improve (measured). Pair with "
        "--score-honesty. 0 = off; ~0.02 gives roughly one removal per held episode.",
    )
    p.add_argument(
        "--score-honesty-false-high",
        type=float,
        default=5.0,
        metavar="F",
        help="(with --score-honesty) how much more a FALSE HIGH score costs than under-claiming a "
        "real grasp. Symmetric shaping is dominated by the common holding case and just teaches "
        "'keep the score high', which is the bug being fixed.",
    )
    p.add_argument(
        "--reach-dr",
        action="store_true",
        help="Domain-randomise the REACH palm pose the arm presents: each episode adds a sampled "
        "position offset and attitude perturbation on top of the tuned pose, with the hand's "
        "residual still composed on top. Widens the engage/object radii to match (default off).",
    )
    p.add_argument(
        "--reach-dr-pos",
        default="0.02,0.02,0.015",
        metavar="X,Y,Z",
        help="(with --reach-dr) symmetric +- position perturbation per axis [m] (default "
        "0.02,0.02,0.015).",
    )
    p.add_argument(
        "--reach-dr-rpy",
        default="15,10,20",
        metavar="R,P,Y",
        help="(with --reach-dr) symmetric +- attitude perturbation per axis [deg] (default "
        "15,10,20).",
    )
    p.add_argument(
        "--no-success",
        action="store_true",
        help="Remove BOTH the success reward (the 5000 held-at-goal bonus) AND the success "
        "termination, so the episode runs to the timeout and the (dense) hold reward pays for every "
        "step the object stays at goal -- a pure reward-for-holding setup with no success jackpot.",
    )
    p.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Override control decimation (physics steps per policy step). control Hz = "
        "1/(physics_dt*decimation).",
    )
    p.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help="Override physics timestep [s]. physics Hz = 1/physics_dt.",
    )
    p.add_argument(
        "--blind",
        action="store_true",
        help="Drop the palm->object vector from the ACTOR obs only (critic keeps it): the hand "
        "must grasp by proprioception + contact alone (asymmetric blind actor / privileged critic).",
    )
    p.add_argument(
        "--limited-obs",
        action="store_true",
        help="Reduce the ACTOR obs to the distillation student's set: 5 fingertip binary contacts "
        "+ hand_joint_pos + hand_tracking (drops the 16 full contacts + palm->object). Critic keeps "
        "full obs (privileged). For a from-scratch limited-obs baseline to compare vs the distilled "
        "student.",
    )
    p.add_argument(
        "--history",
        type=int,
        default=0,
        help="Frame-stack the last N joint-position + tracking-error frames into the obs "
        "(0 = off). Gives the policy short-horizon proprioceptive history for contact inference.",
    )
    p.add_argument(
        "--rnn",
        choices=("lstm", "gru"),
        default=None,
        help="Use a recurrent (RNNModel) actor+critic of this type instead of the MLP, so the "
        "policy can reason over time.",
    )
    p.add_argument(
        "--hidden-dims",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated actor/critic hidden layer sizes, e.g. '512,256,128' or '128,64'. "
        "Default: the task's (256,128,64). With --rnn this sets the MLP head after the recurrent "
        "core.",
    )
    p.add_argument(
        "--hand-mode",
        choices=("joints", "pca"),
        default=None,
        help="Hand action space: 'joints' (20 direct joint targets) or 'pca' (5 PCA latents on "
        "the fitted human-grasp manifold). Default: the task's setting.",
    )
    p.add_argument(
        "--arm-residual",
        action="store_true",
        help="Let the hand policy output a 6-D palm residual (3 position + 3 orientation) added "
        "to the arm phase target, and turn OFF REACH-pose randomisation (the two are mutually "
        "exclusive). The hand thus shapes both where and how the palm meets the object.",
    )
    p.add_argument(
        "--arm-pose-target",
        action="store_true",
        help="TASK-SPACE BASELINE: the policy commands a palm POSE -- action [pos(3), rpy(3), "
        "hand(20)] -- and the tuned Jacobian-pseudoinverse velocity controller (the deploy arm "
        "law) drives the arm to it. The hand stays in joint space. Position spans a workspace box "
        "and attitude is a bounded offset from flat-hold, because a network's raw output has no "
        "units. Mutually exclusive with --arm-joint-target.",
    )
    p.add_argument(
        "--arm-joint-target",
        action="store_true",
        help="BASELINE: the policy commands all 7 arm joint positions directly, alongside the 20 "
        "hand joints -- action [arm(7), hand(20)], no grasp score and no palm residual. The phase "
        "controller stops driving the arm entirely. Everything else is unchanged (same obs, same "
        "four rewards, same hand velocity field + PD, same home_goal_pos), so a comparison against "
        "the phase-controlled policy isolates the controller. Overrides --arm-velocity.",
    )
    p.add_argument(
        "--pregrasp-approach",
        default="0,0,1",
        metavar="X,Y,Z",
        help="With --pregrasp-standoff: direction from the object to the starting palm position. "
        "The default (0,0,1) starts the palm directly above the object.",
    )
    p.add_argument(
        "--pregrasp-orientation",
        default="home",
        choices=("home", "aim"),
        help="Starting wrist attitude. 'home' (default) holds the flat-hold home attitude -- palm "
        "normal down, fingers horizontal, palm parallel to the table -- the same attitude the "
        "arm starts every episode in and that play_grasp.py --tune-reach opens at. 'aim' is "
        "the old RDG behaviour: spin the wrist so an axis points down the approach line, "
        "which makes the starting attitude depend on the approach direction.",
    )
    p.add_argument(
        "--pregrasp-aim-axis",
        default="x",
        choices=("x", "z"),
        help="--pregrasp-orientation aim only: which palm axis points at the object. 'x' is the "
        "palm normal (palm faces the object), 'z' the finger axis (fingertips point at it).",
    )
    p.add_argument(
        "--pregrasp-init-vel",
        default=None,
        metavar="VX,VY,VZ",
        help="Palm velocity [m/s] at reset, mapped to joint velocities by the arm's own damped "
        "pseudoinverse. Default: start from rest.",
    )
    p.add_argument(
        "--pregrasp-standoff",
        type=float,
        default=None,
        metavar="M",
        help="Start each episode with the palm this far from the object, aimed at it, fingers "
        "partially curled (RobustDexGrasp's initialisation; 0.25 is their value). Removes the "
        "reach-discovery problem from the joint-target baseline, which unlike the phase-controlled "
        "task has to learn the approach itself.",
    )
    p.add_argument(
        "--pregrasp-hand-open",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="With --pregrasp-standoff: finger curl at reset, 0 = open pre-grasp pose, 1 = closed.",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="CKPT",
        help="Continue a run: load weights AND keep the iteration counter, so checkpoint "
        "numbering, the LR schedule and the W&B x-axis carry on. Use this to chain time-limited "
        "blocks; --init-from is for fine-tuning a DIFFERENT run and deliberately restarts at 0.",
    )
    p.add_argument(
        "--video-every",
        type=int,
        default=0,
        metavar="N",
        help="Log a rollout clip to W&B every N iterations (0 = off). Frames are captured from "
        "the LIVE training sim -- no second env, so no extra GPU memory -- by copying qpos into a "
        "CPU MjData and rendering offscreen, the same way render_grasp_video.py replays a "
        "checkpoint.",
    )
    p.add_argument(
        "--video-seconds",
        type=float,
        default=6.0,
        metavar="S",
        help="Length of each logged clip.",
    )
    p.add_argument(
        "--reach-reward",
        type=float,
        default=0.0,
        metavar="W",
        help="Weight of a dense palm->object proximity reward, 1 - tanh(d/sigma), to shape the "
        "initial approach (0 = off). Needed by the joint-target baseline, which has to learn the "
        "approach the phase controller otherwise scripts.",
    )
    p.add_argument(
        "--reach-reward-sigma",
        type=float,
        default=0.20,
        metavar="M",
        help="Distance scale of --reach-reward. At the default 0.20 m it pays 0.15 at the 0.25 m "
        "standoff and 0.76 at 5 cm; at 0.10 m it pays only 0.013 at the standoff, i.e. no usable "
        "gradient where the approach actually starts.",
    )
    p.add_argument(
        "--arm-v-max",
        type=float,
        default=0.5,
        metavar="RAD_S",
        help="With --arm-joint-target: max arm joint speed [rad/s] of the arm velocity field, the "
        "same quasi-static rate limit the hand uses. Without it the joint target lands as a step "
        "and the policy learns to swing through the object and launch it.",
    )
    p.add_argument(
        "--arm-slowdown",
        type=float,
        default=0.35,
        metavar="RAD",
        help="Joint error below which the arm velocity field ramps down (mirrors the hand's).",
    )
    p.add_argument(
        "--arm-joint-span",
        type=float,
        default=None,
        metavar="RAD",
        help="With --arm-joint-target: half-span of the commanded joint range, centred on the home "
        "pose. Default (unset) spans each joint's full +-180 deg range, where a zero action is a "
        "pose 1.07 m from reset and most of the action box is unreachable junk.",
    )
    p.add_argument(
        "--arm-velocity",
        action="store_true",
        help="Train with the Jacobian-pseudoinverse velocity arm controller (the sim-to-real DEPLOY "
        "controller) instead of the impedance controller. The hand policy co-adapts its residual + "
        "closure to how the velocity arm presents the palm. Pair with --hand-gate.",
    )
    p.add_argument(
        "--grasp-reward",
        choices=("none", "wrench", "matrix"),
        default="none",
        help="Add a grasp-quality reward on top of the contact/height/hold/success stack: "
        "'wrench' rewards the net hand+gravity wrench on the object -> 0 (a balanced static hold); "
        "'matrix' rewards the grasp matrix min-singular-value Q_MSV (contacts distributed so the "
        "grasp resists a wrench in every direction). Both only pay once the object is held aloft.",
    )
    p.add_argument(
        "--grasp-reward-weight",
        type=float,
        default=None,
        help="Weight for the --grasp-reward term (default: 30 for wrench, 30 for matrix). Both "
        "terms are dense and in [0,1], so this is roughly the per-step reward at a perfect hold "
        "(compare hold=50/step, so ~30 keeps grasp quality a strong-but-secondary shaping term).",
    )
    p.add_argument(
        "--disturbance",
        action="store_true",
        help="Periodically kick the object with a random velocity delta (mjlab "
        "push_by_setting_velocity, mode=interval) so the hand learns a grasp robust to sudden "
        "external disturbances. Mass-independent (a bounded velocity shove, not a force), so -- "
        "unlike a fixed force -- it cannot accelerate light YCB objects past the instability "
        "threshold and launch them.",
    )
    p.add_argument(
        "--disturbance-vel",
        type=float,
        default=0.3,
        help="Peak per-axis linear velocity kick [m/s] (range +-this). Default 0.3 -- a ~3 cm "
        "shove per policy step the hand must catch, well under the 15 m/s instability threshold.",
    )
    p.add_argument(
        "--disturbance-angvel",
        type=float,
        default=0.8,
        help="Peak per-axis angular velocity kick [rad/s] (range +-this). Default 0.8.",
    )
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override the episode length [s] (task default 5). Use a longer episode to train a "
        "longer sustained hold.",
    )
    p.add_argument(
        "--success-hold-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds the object must be held within 0.15 m of the goal (continuously) to count as "
        "success. Sets BOTH the success bonus and the success termination (task default 2 s).",
    )
    p.add_argument(
        "--step-penalty",
        type=float,
        default=None,
        metavar="R",
        help="Per-step time cost (actual reward units/step, dt-adjusted). Applied as a constant "
        "negative reward so idling -- e.g. sitting in a failed LIFT -- bleeds reward, making a "
        "re-approach for another grasp attempt worthwhile. Small values (~0.3-1.0) recommended.",
    )
    p.add_argument(
        "--teacher-contacts",
        action="store_true",
        help="Give the ACTOR the full 16-link contact vector (contact_binary) instead of the 5 "
        "hardware-realistic fingertip sensors. Teacher-privileged: use when this run is a teacher "
        "to be distilled to a proprio/fingertip student later.",
    )
    p.add_argument(
        "--hold-to-end",
        action="store_true",
        help="Reward holding all the way to the episode end instead of terminating after a 2 s "
        "hold: drops the early 'success' termination (so the episode always runs full length) and "
        "makes 'success' fire only if the object is still at the goal when the episode times out. "
        "The dense per-step hold reward keeps paying the whole time, so a policy that slips partway "
        "forfeits both the rest of the hold reward and the end-of-episode success bonus.",
    )
    p.add_argument(
        "--pca-manifold-weight",
        type=float,
        default=None,
        metavar="W",
        help="Add a reward (weight W) for the hand pose lying on the PCA grasp manifold "
        "(exp(-||q-reconstruct(q)||), gated on a held grasp). Soft pull toward natural PCA-like "
        "hand shapes for a joint-space policy; grasping is preserved. Higher W = stronger pull.",
    )
    p.add_argument(
        "--goal-proximity-weight",
        type=float,
        default=None,
        metavar="W",
        help="Weight of a dense Gaussian reward on the object's distance to the fixed goal pose. "
        "ADDITIVE: the height reward is kept, so the lift gradient is unchanged. Off by "
        "default.",
    )
    p.add_argument(
        "--goal-proximity-sigma",
        type=float,
        default=0.2,
        metavar="M",
        help="Width of that kernel [m]. Keep it well under the ~0.50 m an object on the mat sits "
        "from the elevated goal, or the reward becomes slidable.",
    )
    p.add_argument(
        "--hand-pca-project",
        type=float,
        default=None,
        metavar="A",
        help="(joints mode) Blend the commanded hand pose toward its PCA-manifold projection by A "
        "in [0,1] before sending it: 0.5 = halfway, 1 = hard-project (exactly PCA-like shapes). A "
        "harder constraint than --pca-manifold-weight; the policy fine-tunes around the projection.",
    )
    p.add_argument(
        "--hand-gate",
        action="store_true",
        help="Arm-gates-hand: the hand policy drives the fingers only once the palm is within the "
        "phase engage-radius of the object; until then the fingers hold the default pre-grasp pose. "
        "Teaches/uses a policy that grasps on arm-arrival instead of closing while standing still.",
    )
    p.add_argument(
        "--no-reach-hand-pose",
        dest="reach_hand_pose",
        action="store_false",
        help="Disable the learned REACH-phase hand pose (the second 5-D PCA output, on by default "
        "in PCA mode). The fingers then hold the fixed pre-grasp pose during approach instead of a "
        "pose the policy picks on the grasp manifold. Use to reproduce pre-reach-pose checkpoints.",
    )
    p.set_defaults(reach_hand_pose=True)
    p.add_argument(
        "--const-reach",
        action="store_true",
        help="Make the learned REACH pose an observation-INDEPENDENT constant: its mean becomes a "
        "single learnable bias vector (tuned by PPO) instead of a function of the observation. The "
        "reach dims stay sampled actions (still policy-gradient optimized) but are the same pose in "
        "every state. Requires --rnn and reach_hand_pose on. Uses RNNModelConstReach for the actor.",
    )
    p.add_argument(
        "--pca-latent-scale",
        type=float,
        default=None,
        metavar="S",
        help="(PCA mode) Widen the commandable PCA latent range on each axis by this factor "
        "(about its centre); e.g. 2.0 doubles it. Decoded poses stay clamped to joint limits. "
        "Default 1.0 (the fitted percentile range).",
    )
    p.add_argument(
        "--lift-height",
        type=float,
        default=None,
        metavar="M",
        help="Raise the LIFT retreat / hold target (home_goal_pos) by this many metres in +z so "
        "the arm lifts the object higher. The height/hold/success rewards follow it. Default 0.0.",
    )
    p.add_argument(
        "--object-mass-range",
        default=None,
        metavar="LO,HI",
        help="Domain-randomize the object mass to an absolute uniform range [LO,HI] kg per episode "
        "(e.g. 0.03,0.30). Sets mass only; the mesh's nominal inertia tensor is left unchanged.",
    )
    p.add_argument(
        "--object-scale-range",
        default=None,
        metavar="LO,HI",
        help="Per-WORLD object size DR: spawn the objects at 5 discrete scales spanning [LO,HI] "
        "(e.g. 0.6,1.0), one fixed scale per world across the env population (NOT per-episode -- "
        "mesh scale can't change at runtime). Multiplies the variant count -> longer env build.",
    )
    p.add_argument(
        "--lift-wrench",
        type=float,
        default=None,
        metavar="N",
        help="Once the object rises above z=0.2 m, apply a random 6-D wrench (force magnitude <= N) "
        "at a random point on it every step, to train the grasp to hold through disturbances.",
    )
    p.add_argument(
        "--freeze-object-on-engage",
        action="store_true",
        help="Freeze the ESTIMATED object position at the instant the arm->hand gate engages, instead of tracking the object live for the whole episode. Models hardware occlusion: the hand closing on the object is what hides it from the camera, so the arm servos on the last good estimate from that moment. Must MATCH between teacher and student -- it changes the arm's setpoint, so a student distilled under a different setting learns from a teacher acting in a different world.",
    )
    p.add_argument(
        "--init-from",
        default=None,
        metavar="CKPT",
        help="Warm-start from a trained checkpoint (fine-tune). Loads actor+critic+obs-normaliser, "
        "resets the iteration counter to 0. Intended for adding a grasp-quality reward on top of "
        "the base teacher: the new reward is live from step 1 because the policy already grasps, "
        "instead of contributing nothing for the ~hundreds of iters a from-scratch run needs to "
        "first discover grasping. The env config (task/hand-mode/residual/dt) MUST match the "
        "checkpoint's so obs/action dims and the model architecture line up.",
    )
    args = p.parse_args()

    # Contact-coverage reward weights are read at env-state init (mjlab_task._reward_contact_weights),
    # so set the env vars before register()/env build.
    if args.equal_contact_weights:
        os.environ["REFLEXIVE_EQUAL_CONTACTS"] = "1"
        os.environ["REFLEXIVE_PALM_CONTACT_BONUS"] = str(args.palm_contact_bonus)
        print(f"[train] contact reward: equal weights (1.0) + palm bonus {args.palm_contact_bonus}")
    if args.palm_frame_residual:
        os.environ["REFLEXIVE_PALM_FRAME_RESIDUAL"] = "1"
        print("[train] position residual applied in the PALM frame")
    if args.reach_dr:
        # Env vars ONLY here -- this block runs before env_cfg exists, and BatchedGraspPhase reads
        # these at construction. The matching cfg change (reach_randomize=False) has to happen
        # after load_env_cfg; see the arm_residual block below.
        os.environ["REFLEXIVE_REACH_DR"] = "1"
        os.environ["REFLEXIVE_REACH_DR_POS"] = args.reach_dr_pos
        os.environ["REFLEXIVE_REACH_DR_RPY"] = args.reach_dr_rpy
        print(
            f"[train] REACH pose DR: pos +-({args.reach_dr_pos}) m, "
            f"rpy +-({args.reach_dr_rpy}) deg, layered on the tuned pose"
        )

    # Per-world object size DR must be decided BEFORE the env cfg is built (the object mesh
    # variants are baked at cfg-build time). Encode the scale set in an env var the YCB variant
    # builder reads (see get_ycb_variant_object_cfg).
    if args.object_scale_range is not None:
        import numpy as _np

        _lo, _hi = (float(x) for x in args.object_scale_range.split(","))
        _scales = [round(float(s), 3) for s in _np.linspace(_lo, _hi, 5)]
        os.environ["REFLEXIVE_OBJECT_SCALES"] = ",".join(f"{s:g}" for s in _scales)
        print(f"[train] object size DR (per-world, fixed per env): scales {_scales}")

    register()
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.contact_weight is not None:
        env_cfg.rewards["contact"].weight = args.contact_weight
    if args.height_weight is not None:
        env_cfg.rewards["height"].weight = args.height_weight
    if args.hold_weight is not None:
        env_cfg.rewards["hold"].weight = args.hold_weight
        print(f"[train] hold reward weight -> {args.hold_weight}")
    if args.grasp_reward != "none":
        from blindgraspreflex.tasks.grasping.mjlab_task import set_grasp_metric_reward

        default_w = {"wrench": 30.0, "matrix": 30.0}
        weight = (
            args.grasp_reward_weight
            if args.grasp_reward_weight is not None
            else default_w[args.grasp_reward]
        )
        set_grasp_metric_reward(env_cfg, args.grasp_reward, weight)
        print(f"[train] grasp-quality reward '{args.grasp_reward}' added at weight {weight}")
    if args.goal_proximity_weight is not None:
        env_cfg.rewards["goal_proximity"].weight = float(args.goal_proximity_weight)
        env_cfg.rewards["goal_proximity"].params["sigma"] = float(args.goal_proximity_sigma)
        print(
            f"[train] goal-proximity reward weight={args.goal_proximity_weight} "
            f"sigma={args.goal_proximity_sigma} (ADDITIVE -- the height reward is unchanged)"
        )
    if args.hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = args.hand_pca_project
        print(f"[train] hand PCA projection alpha={args.hand_pca_project} (0=off, 1=hard-project)")
    if args.freeze_object_on_engage:
        env_cfg.actions["grasp"].freeze_object_on_engage = True
        print("[train] object position FROZEN at hand engage (perception occlusion model)")
    if args.pca_manifold_weight is not None:
        from mjlab.managers.reward_manager import RewardTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import hand_pca_manifold_reward

        env_cfg.rewards["pca_manifold"] = RewardTermCfg(
            func=hand_pca_manifold_reward, params={}, weight=args.pca_manifold_weight
        )
        print(f"[train] PCA-manifold reward added at weight {args.pca_manifold_weight}")
    if args.disturbance:
        # Periodic mass-independent velocity kicks on the object so the grasp must hold through
        # sudden shoves. A fixed *force* accelerates light YCB objects past the instability
        # threshold (they launch, spiking the height reward to 1e8); a bounded velocity delta
        # cannot, so it is fair across the ~10x object-mass range. mode="interval": fresh kick
        # every interval_range_s.
        from mjlab.envs.mdp.events import push_by_setting_velocity
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        v, w = args.disturbance_vel, args.disturbance_angvel
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
        print(
            f"[train] object disturbances ON: velocity kick +-{v} m/s / +-{w} rad/s every 0.3-0.8s"
        )
    if args.episode_length is not None:
        env_cfg.episode_length_s = args.episode_length
        print(f"[train] episode length -> {args.episode_length}s")
    if args.hold_to_end:
        # Hold-as-long-as-possible: no early success termination (episode runs full length), and
        # success only fires if still held at the timeout. Keeps the dense hold reward paying the
        # whole episode so slipping partway forfeits the rest.
        from mjlab.managers.reward_manager import RewardTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import success_held_to_end

        removed = env_cfg.terminations.pop("success", None)
        w_success = env_cfg.rewards["success"].weight
        env_cfg.rewards["success"] = RewardTermCfg(
            func=success_held_to_end, params={"threshold": 0.15}, weight=w_success
        )
        print(
            f"[train] hold-to-end: dropped early success termination ({removed is not None}); "
            f"success now fires at episode end if held (weight {w_success})"
        )
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.physics_dt is not None:
        env_cfg.sim.mujoco.timestep = args.physics_dt
    if args.decimation is not None or args.physics_dt is not None:
        dt = env_cfg.sim.mujoco.timestep
        print(
            f"[train] physics {1 / dt:.0f} Hz (dt {dt}), decimation {env_cfg.decimation} "
            f"-> control {1 / (dt * env_cfg.decimation):.1f} Hz"
        )
    if args.success_hold_time is not None:
        # Both must match: the bonus reward and the terminating condition share the hold duration.
        env_cfg.rewards["success"].params["hold_time_s"] = args.success_hold_time
        if "success" in env_cfg.terminations:
            env_cfg.terminations["success"].params["hold_time_s"] = args.success_hold_time
        print(f"[train] success = held at goal for {args.success_hold_time}s continuous")
    if args.no_success:
        removed_r = env_cfg.rewards.pop("success", None)
        removed_t = env_cfg.terminations.pop("success", None)
        print(
            f"[train] --no-success: removed success reward ({removed_r is not None}) + "
            f"termination ({removed_t is not None}); episode runs to timeout, hold reward is terminal"
        )
    if args.step_penalty is not None:
        from mjlab.managers.reward_manager import RewardTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import step_penalty as _step_penalty_fn

        # Reward manager scales by the control step dt, so weight = cost/dt gives ~`cost` per step.
        _ctrl_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
        env_cfg.rewards["step_cost"] = RewardTermCfg(
            func=_step_penalty_fn, params={}, weight=-args.step_penalty / _ctrl_dt
        )
        print(
            f"[train] per-step time cost = -{args.step_penalty}/step "
            f"(weight {-args.step_penalty / _ctrl_dt:.2f}, ctrl dt {_ctrl_dt:.3f}s)"
        )
    if args.hand_vf_dr is not None:
        from mjlab.managers.event_manager import EventTermCfg as _E1

        from blindgraspreflex.tasks.grasping.mjlab_task import randomize_hand_vf

        vlo, vhi, slo, shi = (float(x) for x in args.hand_vf_dr.split(","))
        env_cfg.events["hand_vf_dr"] = _E1(
            func=randomize_hand_vf,
            mode="reset",
            params={"v_range": (vlo, vhi), "slowdown_range": (slo, shi)},
        )
        print(
            f"[train] hand velocity-field DR: v_max [{vlo},{vhi}] rad/s, "
            f"slowdown [{slo},{shi}] rad (per episode)"
        )
    if args.hand_gain_dr is not None:
        from mjlab.envs.mdp.dr.actuator import pd_gains
        from mjlab.managers.event_manager import EventTermCfg as _E2
        from mjlab.managers.scene_entity_config import SceneEntityCfg as _SEC

        from blindgraspreflex.assets.rizon4_hand import ARM_ACTUATORS, HAND_ACTUATORS

        kplo, kphi, kdlo, kdhi = (float(x) for x in args.hand_gain_dr.split(","))
        # pd_gains selects by ACTUATOR index, not joint name. The entity is built as
        # (*ARM_ACTUATORS, *HAND_ACTUATORS), so the hand block starts after the arm. Computed
        # rather than hard-coded so adding an arm actuator cannot silently shift the slice onto
        # the wrong joints -- and the arm ones are torque motors, where scaling a "gain" would
        # change the impedance law itself rather than a PD stiffness.
        _n_arm = len(ARM_ACTUATORS)
        _hand_ids = slice(_n_arm, _n_arm + len(HAND_ACTUATORS))
        env_cfg.events["hand_gain_dr"] = _E2(
            func=pd_gains,
            mode="reset",
            params={
                "kp_range": (kplo, kphi),
                "kd_range": (kdlo, kdhi),
                "operation": "scale",
                "asset_cfg": _SEC("robot", actuator_ids=_hand_ids),
            },
        )
        print(f"[train] hand PD-gain DR: kp x[{kplo},{kphi}], kd x[{kdlo},{kdhi}] (per episode)")
    if args.vanish_prob > 0.0:
        from blindgraspreflex.tasks.grasping.mjlab_task import object_vanish

        env_cfg.events["object_vanish"] = EventTermCfg(
            func=object_vanish,
            mode="interval",
            interval_range_s=(0.084, 0.084),
            params={"prob": args.vanish_prob},
        )
        print(
            f"[train] object-vanish events ON: p={args.vanish_prob}/step once lifted "
            f"(creates the 'hand closed on nothing' state the drop test measures)"
        )
    if args.score_honesty > 0.0:
        from mjlab.managers.reward_manager import RewardTermCfg as _RTC

        from blindgraspreflex.tasks.grasping.mjlab_task import score_honesty_reward

        env_cfg.rewards["score_honesty"] = _RTC(
            func=score_honesty_reward,
            params={
                "radius": args.score_honesty_radius,
                "false_high": args.score_honesty_false_high,
            },
            weight=args.score_honesty,
        )
        print(
            f"[train] score-honesty reward w={args.score_honesty} "
            f"(radius {args.score_honesty_radius} m): grasp_score is now rewarded for being "
            f"correct, not just for keeping the arm in LIFT"
        )
    if args.teacher_contacts:
        from mjlab.managers.observation_manager import ObservationTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import contact_binary

        env_cfg.observations["actor"].terms["contacts"] = ObservationTermCfg(
            func=contact_binary, params={}
        )
        print(
            "[train] actor contacts -> full 16-link vector (teacher-privileged, was 5 fingertips)"
        )
    if args.hand_mode is not None:
        env_cfg.actions["grasp"].hand_mode = args.hand_mode
        print(f"[train] hand action space: {args.hand_mode}")
    if args.reach_dr:
        # Layered DR perturbs the NOMINAL tuned pose, and it lives in the FIXED-pose branch of
        # BatchedGraspPhase.update. reach_randomize is a different mechanism entirely -- it
        # REPLACES the nominal pose with an absolute sample from wide bounds (+-80 deg roll) -- and
        # if it is left on, the fixed-pose branch never runs and the perturbations are silently
        # ignored. So turn it off whenever layered DR is requested.
        env_cfg.actions["grasp"].phase.reach_randomize = False
        print("[train] reach_randomize OFF (layered --reach-dr offsets the nominal pose instead)")
    if args.arm_residual:
        # Residual-controlled arm: hand outputs a 6-D palm residual; disable the mutually
        # exclusive REACH-pose domain randomisation so the arm follows the fixed phase target.
        env_cfg.actions["grasp"].arm_residual = True
        env_cfg.actions["grasp"].phase.reach_randomize = False
        print("[train] arm_residual ON (6-D palm pos+rot residual), reach_randomize OFF")
    if args.reach_reward > 0.0:
        # Import locally: a later block in main() does its own `from ... import RewardTermCfg`,
        # which makes the name function-local for the WHOLE function and shadows the module-level
        # import here (UnboundLocalError, not a NameError -- it fails only at this line).
        from mjlab.managers.reward_manager import RewardTermCfg as _RewardTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import palm_proximity_reward

        env_cfg.rewards["reach"] = _RewardTermCfg(
            func=palm_proximity_reward,
            params={"sigma": args.reach_reward_sigma},
            weight=args.reach_reward,
        )
        print(
            f"[train] reach reward ON: {args.reach_reward} * (1 - tanh(d/{args.reach_reward_sigma}))"
        )
    if args.pregrasp_standoff is not None:
        from mjlab.managers.event_manager import EventTermCfg as _EvCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import reset_pregrasp_approach

        # Declared AFTER reset_object: it reads the object's post-reset pose to aim the palm, and
        # mjlab applies reset events in declaration order.
        _approach = tuple(float(v) for v in args.pregrasp_approach.split(","))
        _ivel = (
            tuple(float(v) for v in args.pregrasp_init_vel.split(","))
            if args.pregrasp_init_vel
            else None
        )
        env_cfg.events["reset_pregrasp"] = _EvCfg(
            func=reset_pregrasp_approach,
            mode="reset",
            params={
                "standoff": args.pregrasp_standoff,
                "hand_open_frac": args.pregrasp_hand_open,
                "approach_dir": _approach,
                "orientation": args.pregrasp_orientation,
                "aim_axis": args.pregrasp_aim_axis,
                "init_vel": _ivel,
            },
        )
        # Printed in full because these decide the ONLY state distribution this policy ever sees
        # at t=0, and a run that differs here is not resumable from one that does not.
        print(
            f"[train] pre-grasp init: palm {args.pregrasp_standoff} m from the object along "
            f"{_approach}, attitude={args.pregrasp_orientation}"
            + (
                f"/{args.pregrasp_aim_axis}"
                if args.pregrasp_orientation == "aim"
                else " (flat-hold: palm parallel to the table)"
            )
            + f", fingers {args.pregrasp_hand_open:.0%} curled, "
            + (f"initial palm velocity {_ivel} m/s" if _ivel else "from rest")
        )
    if args.arm_pose_target and args.arm_joint_target:
        p.error("--arm-pose-target and --arm-joint-target are mutually exclusive")
    if args.arm_pose_target:
        env_cfg.actions["grasp"].arm_control = "pose_policy"
        # Same reasoning as the joint-target baseline: the policy owns the arm, so there is no
        # REACH/LIFT latch to drive, no residual to add to a scripted target, and no hand gate
        # waiting on a scripted approach.
        env_cfg.actions["grasp"].hand_gate = False
        env_cfg.actions["grasp"].reach_hand_pose = False
        env_cfg.actions["grasp"].arm_residual = False
        # The policy drives the arm, so it must see the arm. Only joint positions here: the
        # joint-target baseline's arm_tracking is _arm_cmd - q, and pose_policy has no _arm_cmd
        # (the velocity law tracks a Cartesian setpoint), so that term would be identically zero.
        from mjlab.managers.observation_manager import ObservationTermCfg as _ObsCfg2

        from blindgraspreflex.tasks.grasping.mjlab_task import arm_joint_pos as _ajp

        for _grp in ("actor", "critic"):
            _t = dict(env_cfg.observations[_grp].terms)
            env_cfg.observations[_grp].terms = {
                "arm_joint_pos": _ObsCfg2(func=_ajp, params={}),
                **_t,
            }
        c = env_cfg.actions["grasp"]
        print(
            f"[train] arm_control = pose_policy TASK-SPACE BASELINE: action "
            f"[pos(3), rpy(3), hand(20)]; box centre {c.pose_box_centre} half {c.pose_box_half}, "
            f"rpy span +-{math.degrees(c.pose_rpy_span):.0f} deg"
        )
        print("[train] arm obs added: arm_joint_pos(7) to actor and critic")
    if args.arm_joint_target:
        env_cfg.actions["grasp"].arm_control = "joint_target"
        env_cfg.actions["grasp"].arm_v_max = args.arm_v_max
        env_cfg.actions["grasp"].arm_slowdown_distance = args.arm_slowdown
        print(
            f"[train] arm velocity field: v_max {args.arm_v_max} rad/s, "
            f"slowdown {args.arm_slowdown} rad"
        )
        if args.arm_joint_span is not None:
            env_cfg.actions["grasp"].arm_joint_span = args.arm_joint_span
            print(f"[train] arm joint targets limited to home +-{args.arm_joint_span} rad")
        # The hand gate and the learned reach pose both exist to wait on the SCRIPTED arm; with the
        # policy driving the arm there is nothing to wait for, and leaving them on would hold the
        # fingers open on a proximity latch the policy no longer controls.
        env_cfg.actions["grasp"].hand_gate = False
        env_cfg.actions["grasp"].reach_hand_pose = False
        env_cfg.actions["grasp"].arm_residual = False
        # The policy now drives the arm, so it must SEE the arm. The stock obs is hand-only
        # (contacts, palm->object, hand joints, hand tracking) because the arm was scripted.
        from mjlab.managers.observation_manager import ObservationTermCfg as _ObsCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import arm_joint_pos, arm_tracking_error

        for _grp in ("actor", "critic"):
            _t = dict(env_cfg.observations[_grp].terms)
            _new = {"arm_joint_pos": _ObsCfg(func=arm_joint_pos, params={})}
            _new["arm_tracking"] = _ObsCfg(func=arm_tracking_error, params={})
            # Put the arm block first so the layout reads [arm(14), <original hand obs>].
            env_cfg.observations[_grp].terms = {**_new, **_t}
        print("[train] arm obs added: arm_joint_pos(7) + arm_tracking(7) to actor and critic")
        print(
            "[train] arm_control = joint_target BASELINE: action [arm(7), hand(20)]; "
            "hand_gate/reach_hand_pose/arm_residual forced OFF"
        )
    elif args.arm_velocity:
        env_cfg.actions["grasp"].arm_control = "velocity"
        print("[train] arm_control = velocity (Jacobian-pseudoinverse deploy controller)")
    if args.hand_gate:
        env_cfg.actions["grasp"].hand_gate = True
        print("[train] hand_gate ON (hand policy engaged only when palm is near the object)")
    env_cfg.actions["grasp"].reach_hand_pose = args.reach_hand_pose
    if args.reach_hand_pose:
        print("[train] reach_hand_pose ON (learned REACH hand pose: 5-D PCA latent or 20-D joints)")
    else:
        print("[train] reach_hand_pose OFF (fixed pre-grasp pose during approach)")
    if args.pca_latent_scale is not None:
        env_cfg.actions["grasp"].pca_latent_scale = args.pca_latent_scale
        print(
            f"[train] pca_latent_scale = {args.pca_latent_scale} (PCA latent range x{args.pca_latent_scale})"
        )
    if args.lift_height is not None:
        env_cfg.actions["grasp"].lift_height_offset = args.lift_height
        print(f"[train] lift_height_offset = +{args.lift_height} m (higher LIFT/hold target)")
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
        print(f"[train] object mass DR: absolute uniform [{_mlo}, {_mhi}] kg per episode")
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
        print(
            f"[train] lift wrench disturbance: <= {args.lift_wrench} N at a random point once object z>0.2 m"
        )
    if args.blind:
        # Remove palm->object from the ACTOR group only (the critic dict is separate and keeps it).
        removed = env_cfg.observations["actor"].terms.pop("palm_to_object", None)
        print(f"[train] BLIND actor: removed palm_to_object from actor obs ({removed is not None})")
    if args.limited_obs:
        # Reduce the actor obs to the distillation student's set (fingertip contacts + proprioception);
        # critic keeps full obs (privileged). Matches distill_grasp.py's student group exactly.
        from mjlab.managers.observation_manager import ObservationTermCfg

        from blindgraspreflex.tasks.grasping.mjlab_task import fingertip_contact_binary

        actor = env_cfg.observations["actor"].terms
        env_cfg.observations["actor"].terms = {
            "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
            "hand_joint_pos": actor["hand_joint_pos"],
            "hand_tracking": actor["hand_tracking"],
        }
        print(
            "[train] LIMITED actor obs: fingertip_contacts + hand_joint_pos + hand_tracking (critic full)"
        )
    if args.history > 0:
        # hand_joint_pos / hand_tracking term objects are shared by actor+critic, so setting
        # history_length here stacks the last N frames in both groups.
        for group in ("actor", "critic"):
            for t in ("hand_joint_pos", "hand_tracking"):
                if t in env_cfg.observations[group].terms:
                    env_cfg.observations[group].terms[t].history_length = args.history
        print(f"[train] history_length={args.history} on hand_joint_pos + hand_tracking")
    rl_cfg = load_rl_cfg(args.task)
    if args.rnn is not None:
        for m in (rl_cfg.actor, rl_cfg.critic):
            m.class_name = "RNNModel"
            m.rnn_type = args.rnn
            m.rnn_hidden_dim = 256
            m.rnn_num_layers = 1
        print(f"[train] recurrent policy: {args.rnn.upper()} (hidden 256, 1 layer)")
    if args.const_reach:
        if args.rnn is None:
            raise SystemExit("--const-reach requires --rnn (uses RNNModelConstReach).")
        if not args.reach_hand_pose:
            raise SystemExit(
                "--const-reach requires the reach pose on (drop --no-reach-hand-pose)."
            )
        # Trailing action dims that become the obs-independent constant = the reach-pose block size
        # (the hand DOF: 20 for joints, 5 for PCA). Read by RNNModelConstReach via env var.
        reach_dim = 20 if args.hand_mode == "joints" else 5
        os.environ["REFLEXIVE_CONST_REACH_DIM"] = str(reach_dim)
        # Actor only -- the critic keeps full observation conditioning (value must see the state).
        rl_cfg.actor.class_name = "blindgraspreflex.models.const_reach_model:RNNModelConstReach"
        print(
            f"[train] const REACH pose: last {reach_dim} action dims are a learnable obs-independent constant"
        )
    if args.hidden_dims is not None:
        dims = tuple(int(x) for x in args.hidden_dims.split(","))
        rl_cfg.actor.hidden_dims = dims
        rl_cfg.critic.hidden_dims = dims
        print(f"[train] actor/critic hidden dims -> {dims}")
    rl_cfg.max_iterations = args.iterations
    rl_cfg.logger = args.logger
    rl_cfg.wandb_project = args.wandb_project
    if args.run_name:
        rl_cfg.run_name = args.run_name

    from mjlab.envs import ManagerBasedRlEnv

    base_env = ManagerBasedRlEnv(env_cfg, device=args.device)
    if args.video and args.video_every == 0:
        # Preserve the older flags as aliases, but route them through the Viser recorder.
        args.video_every = args.video_interval
        args.video_seconds = args.video_length * env_cfg.decimation * env_cfg.sim.mujoco.timestep

    env = SafeVecEnvWrapper(base_env)
    from mjlab.rl.runner import MjlabOnPolicyRunner

    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), args.log_dir, args.device)
    if args.resume_from is not None:
        # True continuation, unlike --init-from: runner.load already restores the iteration
        # counter, and NOT zeroing it keeps checkpoint numbering, the LR schedule and the W&B
        # x-axis continuous across blocks. Zeroing it made every block restart numbering at 0, so
        # `ls -v | tail -1` across blocks returned an EARLIER block's checkpoint (model_504 from
        # block 1 beat model_480 from block 2) and the next block silently resumed from it,
        # discarding a full block of training.
        runner.load(args.resume_from)
        print(
            f"[train] resumed from {args.resume_from} at iteration "
            f"{runner.current_learning_iteration} (counter preserved)"
        )
    elif args.init_from is not None:
        # Warm-start (fine-tune): load weights + obs-normaliser stats, then reset the iteration
        # counter so this run's checkpoints and LR schedule start fresh at 0.
        runner.load(args.init_from)
        runner.current_learning_iteration = 0
        print(f"[train] warm-started from {args.init_from}; iteration reset to 0")
    if args.video_every > 0:
        _train_with_video(runner, env, args, rl_cfg.max_iterations)
    else:
        runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)


def _train_with_video(runner, env, args, total_iters):
    """runner.learn in chunks, logging a clip to W&B between them.

    learn() resumes from current_learning_iteration, so chunking continues the same run rather
    than restarting it; init_at_random_ep_len is only meaningful on the first chunk.
    """
    step = args.video_every
    done = 0
    first = True
    # learn() calls logger.stop_logging_writer() at the END of every call, and for W&B that is
    # wandb.finish(). Chunked, the run is therefore closed after the first chunk and every later
    # add_scalar raises "You must call wandb.init() before wandb.log()". Suppress it for the
    # duration and close the writer once, after the last chunk.
    _orig_stop = runner.logger.stop_logging_writer
    runner.logger.stop_logging_writer = lambda: None
    while done < total_iters:
        k = min(step, total_iters - done)
        runner.learn(num_learning_iterations=k, init_at_random_ep_len=first)
        done += k
        # rsl-rl stores the index of the last completed iteration, while the next learn() call
        # treats the value as its inclusive starting index. Advance it explicitly between chunks
        # so video capture does not repeat an update, checkpoint number, or media filename.
        runner.current_learning_iteration += 1
        if first:
            # learn() calls logger.init_logging_writer() every time, and that unconditionally
            # CONSTRUCTS a new writer -- for W&B that is a fresh wandb.init(), i.e. a brand new run
            # per chunk (60 of them at --video-every 100 over 6000 iterations), with every metric
            # fragmented across them and wandb.log() then failing against the closed run. Neutering
            # it after the first chunk keeps one writer, one run, and one continuous set of curves.
            # It also skips the re-save of code state, which is already done.
            runner.logger.init_logging_writer = lambda: None
        first = False
        try:
            from blindgraspreflex.training.viser_video import capture_rollout, write_mp4

            it = runner.current_learning_iteration
            clip = capture_rollout(
                runner,
                env,
                device=args.device,
                seconds=args.video_seconds,
                observation_group="actor",
            )
            vdir = pathlib.Path(args.log_dir) / "videos"
            vdir.mkdir(parents=True, exist_ok=True)
            # UNIQUE per iteration: rsl_rl's wandb writer dedupes save_video by FILENAME, so a
            # fixed name would upload the first clip and silently drop every later one.
            path = vdir / f"rollout_{it:05d}.mp4"
            write_mp4(path, clip)
            writer = getattr(runner.logger, "writer", None)
            if writer is not None and hasattr(writer, "save_video"):
                writer.save_video(path, it)  # the logger's own supported path
                print(f"[train] logged rollout clip at iteration {it} -> {path.name}")
            else:
                print(f"[train] wrote rollout clip {path} (writer has no save_video)")
        except Exception as exc:  # a broken renderer must never kill a training run
            print(f"[train] video logging skipped: {type(exc).__name__}: {exc}")

    runner.logger.stop_logging_writer = _orig_stop
    runner.logger.stop_logging_writer()


if __name__ == "__main__":
    main()
