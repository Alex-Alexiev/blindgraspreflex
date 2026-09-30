"""Candidate proprioceptive contact metrics, and the machinery to score their INVARIANCE.

The goal is a per-finger scalar, computable from proprioception alone, whose value at contact is
the same whatever the hand's stiffness, velocity field or physics rate. That rules out the signals
the policy currently sees:

  hand_target - q : dominated by the velocity-field lag, whose magnitude scales with
                    tau = slowdown / v_max. Not velocity-invariant, and measured d' at contact is
                    NEGATIVE (~-0.3) -- the lag is a property of the command filter, not the world.
  q_cmd - q       : the servo error. A softer joint deflects further under the same contact load,
                    so the plateau grows as kp falls (0.20 rad at kp x1.0 -> 0.47 at x0.083).
  tau             : kp*(q_cmd-q) - kd*qd. Falls with kp (0.58 Nm -> 0.18), though only ~5x for a
                    12x gain cut, because the servo error partly compensates.

The invariant that IS available is KINEMATIC. Contact is "the joint stopped going where it was
told", and where it was told is q_cmd, which the controller knows exactly. A ratio of achieved to
commanded MOTION is dimensionless, cancels the speed scale, does not contain kp at all, and is a
displacement ratio over a fixed TIME window rather than a sample count, so it is rate-agnostic.

Its weakness is the converged hold: once q_cmd reaches hand_target there is no commanded motion
left to compare against, and the ratio goes 0/0. Metrics here that depend on motion therefore
carry a latch (see ``stall_ratio``), which is what a contact detector wants anyway -- contact
persists until release.
"""

from __future__ import annotations

import numpy as np

from blindgraspreflex.grasp_trace import FINGER_OF, N_FINGER

EPS = 1e-9


def _win(dt: float, seconds: float) -> int:
    """Window length in SAMPLES for a fixed duration -- this is what makes a metric rate-agnostic."""
    return max(2, int(round(seconds / max(dt, EPS))))


def _finger_reduce(x, reduce="absmax"):
    """(T, 20) -> (T, 5). absmax: one loaded phalanx IS the event; the mean dilutes it 4x."""
    out = np.zeros((x.shape[0], N_FINGER))
    for f in range(N_FINGER):
        b = x[:, FINGER_OF == f]
        out[:, f] = np.abs(b).max(axis=1) if reduce == "absmax" else b.mean(axis=1)
    return out


# --------------------------------------------------------------------------------------------
# Baselines: what the policy sees today, for reference. None of these are invariant.
# --------------------------------------------------------------------------------------------


def m_total(tr, **kw):
    return _finger_reduce(tr.hand_target - tr.q)


def m_servo(tr, **kw):
    return _finger_reduce(tr.q_cmd - tr.q)


def m_tau(tr, **kw):
    return _finger_reduce(tr.tau_cmd)


# --------------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------------


def m_stall_ratio(tr, window_s: float = 0.15, min_cmd: float = 5e-3, latch_s: float = 0.30, **kw):
    """M3 -- fraction of COMMANDED motion that did not happen, over a fixed time window.

        S = clip(1 - |dq| / |dq_cmd|, 0, 1)      over the last ``window_s`` seconds

    Free air: the joint tracks q_cmd (with a position offset that depends on kp, but the same
    VELOCITY), so dq ~ dq_cmd and S ~ 0. Blocked: dq ~ 0 and S ~ 1. The ratio cancels the speed
    scale (v_max multiplies both) and never references kp. The window is a fixed DURATION, so the
    sample count adapts to the physics rate and the value does not.

    ``min_cmd`` gates the 0/0 case where q_cmd has converged and there is no commanded motion to
    compare against; ``latch_s`` then holds the last valid value, since contact does not stop being
    true just because the command stopped moving.
    """
    n = _win(tr.dt, window_s)
    core = _stall_core(tr, n, min_cmd)  # (T-n, 5)
    padded = np.vstack([np.full((n, N_FINGER), np.nan), core])
    return _latch(padded, _win(tr.dt, latch_s))


def _stall_core(tr, n, min_cmd):
    dq = np.abs(tr.q[n:] - tr.q[:-n])
    dqc = np.abs(tr.q_cmd[n:] - tr.q_cmd[:-n])
    s = np.clip(1.0 - dq / np.maximum(dqc, EPS), 0.0, 1.0)
    s = np.where(dqc < min_cmd, np.nan, s)
    return _finger_reduce_nan(s)


def _finger_reduce_nan(x):
    out = np.full((x.shape[0], N_FINGER), np.nan)
    for f in range(N_FINGER):
        b = x[:, FINGER_OF == f]
        with np.errstate(all="ignore"):
            out[:, f] = np.nanmax(np.abs(b), axis=1) if not np.all(np.isnan(b)) else np.nan
    return out


def _latch(x, hold_n):
    """Forward-fill NaN for up to ``hold_n`` samples, then give up (back to NaN)."""
    out = x.copy()
    for c in range(out.shape[1]):
        last, age = np.nan, 10**9
        for i in range(out.shape[0]):
            if not np.isnan(out[i, c]):
                last, age = out[i, c], 0
            else:
                age += 1
                out[i, c] = last if age <= hold_n else np.nan
    return out


def m_vel_deficit(tr, window_s: float = 0.10, min_cmd_rate: float = 0.05, **kw):
    """M4 -- instantaneous velocity deficit, 1 - qd / qd_cmd, smoothed over a fixed time window.

    The differential form of M3. Same invariance argument; noisier, but it responds a window
    earlier because it does not need displacement to accumulate.
    """
    n = _win(tr.dt, window_s)
    qd_cmd = np.gradient(tr.q_cmd, tr.dt, axis=0)
    k = np.ones(n) / n

    def sm(a):
        return np.stack([np.convolve(a[:, j], k, mode="same") for j in range(a.shape[1])], 1)

    qd_s, qdc_s = np.abs(sm(tr.qd)), np.abs(sm(qd_cmd))
    d = np.clip(1.0 - qd_s / np.maximum(qdc_s, EPS), 0.0, 1.0)
    d = np.where(qdc_s < min_cmd_rate, np.nan, d)
    return _latch(_finger_reduce_nan(d), _win(tr.dt, 0.30))


def m_gap_ratio(tr, **kw):
    """M5 -- how much of the intended travel is still outstanding: (hand_target - q) / travel.

    ``travel`` is the commanded excursion so far, |hand_target - q_at_window_start|. Purely
    geometric: where the object stops the finger is a fact about the world, not about the gains.
    Survives the converged hold, which is exactly where the motion-based metrics go undefined.
    """
    gap = np.abs(tr.hand_target - tr.q)
    travel = np.abs(tr.hand_target - tr.q_cmd) + gap
    r = np.clip(gap / np.maximum(travel, EPS), 0.0, 1.0)
    return _finger_reduce(r)


def m_norm_servo(tr, **kw):
    """M6 -- servo error in units of the velocity-field slowdown radius: (q_cmd - q) / slowdown.

    A cheap scale-free version of the servo error. Removes the velocity-field scale but NOT kp,
    so it is expected to fail the stiffness leg -- included as the control that isolates which
    invariance each candidate actually buys.
    """
    slow = float(tr.meta.get("slowdown", 0.35))
    return _finger_reduce((tr.q_cmd - tr.q) / max(slow, EPS))


def m_tau_over_friction(tr, quant: float = 0.2, hist_s: float = 2.0, min_rate: float = 0.05, **kw):
    """M7 -- commanded torque in units of the joint's own free-running torque.

        R = |tau| / p20(|tau| while the command is moving, trailing hist_s)

    The normaliser is the torque it takes to keep the joint moving in free air: Coulomb friction
    plus a small viscous term. That quantity does NOT depend on kp -- a softer joint simply carries
    a proportionally larger position error to produce the same torque (kp*e = friction, so
    e = friction/kp). Dividing by it therefore cancels the stiffness AND the torque scale, leaving
    a dimensionless "how many times harder am I pushing than it takes to just move", which is what
    contact physically is.

    Unlike the kinematic ratios (M3/M4) this survives the converged hold, where q_cmd has stopped
    and there is no commanded motion left to compare against. The normaliser is estimated online
    from the joint's own recent history, so nothing needs calibrating in advance and it self-corrects
    as friction changes -- which is the whole difficulty on real hardware.
    """
    n = _win(tr.dt, hist_s)
    tau = np.abs(tr.tau_cmd)
    # MEASURED motion, not commanded. During contact the command keeps advancing while the joint
    # is stalled; gating on the command admits contact torque into its own baseline, and under a
    # sustained hold the metric decays back to 1.0 exactly when it should read highest.
    moving = np.abs(tr.qd) > min_rate
    T, J = tau.shape

    # The baseline is a slowly varying property of the joint, so it is recomputed on a coarse
    # stride (0.05 s) and held between updates. Re-quantiling every sample per joint was ~6M
    # quantile calls on one trace and dominated the entire analysis.
    stride = _win(tr.dt, 0.05)
    idx = np.arange(0, T, stride)
    base = np.full((len(idx), J), np.nan)
    masked = np.where(moving, tau, np.nan)
    for k, i in enumerate(idx):
        a = max(0, i - n)
        w = masked[a : i + 1]
        enough = np.count_nonzero(~np.isnan(w), axis=0) >= 8
        if not enough.any():
            continue
        with np.errstate(all="ignore"):
            q = np.nanquantile(w, quant, axis=0)  # one call for all 20 joints
        base[k] = np.where(enough, q, np.nan)

    full = np.repeat(base, stride, axis=0)[:T]
    if full.shape[0] < T:
        full = np.vstack([full, np.repeat(full[-1:], T - full.shape[0], axis=0)])
    with np.errstate(all="ignore"):
        out = np.where(full > 1e-6, tau / full, np.nan)
    # Clipped: the ratio spikes past 90 when the trailing baseline is momentarily tiny, and a few
    # such samples would dominate the cross-condition mean the invariance score is computed from.
    # 20x free-running torque is already unambiguous contact.
    return np.clip(_finger_reduce_nan(out), 0.0, 20.0)


def _running_ref(x, dt, quant=0.2, hist_s=2.0, stride_s=0.05, min_n=8):
    """Trailing low-quantile of ``x`` per column -- each signal's own free-running level."""
    T, J = x.shape
    n, stride = _win(dt, hist_s), _win(dt, stride_s)
    idx = np.arange(0, T, stride)
    ref = np.full((len(idx), J), np.nan)
    for k, i in enumerate(idx):
        w = x[max(0, i - n) : i + 1]
        enough = np.count_nonzero(~np.isnan(w), axis=0) >= min_n
        if not enough.any():
            continue
        with np.errstate(all="ignore"):
            q = np.nanquantile(w, quant, axis=0)
        ref[k] = np.where(enough, q, np.nan)
    full = np.repeat(ref, stride, axis=0)[:T]
    if full.shape[0] < T:
        full = np.vstack([full, np.repeat(full[-1:], T - full.shape[0], axis=0)])
    return full


def m_stall_norm(tr, window_s: float = 0.15, min_cmd: float = 5e-3, latch_s: float = 0.30, **kw):
    """M8 -- the stall ratio rescaled against its OWN free-running level.

        S* = clip((S - S_ref) / (1 - S_ref), 0, 1),   S_ref = trailing p20 of S

    M3 assumed the finger tracks its command in free air, so that S ~ 0 until contact. Measured,
    it does not: friction is large enough relative to the drive that the RH-5 finger already sits
    around S ~ 0.5 in free air, leaving M3 almost no contrast to spend on contact (d' ~ 0.7).

    Rescaling against the joint's own recent free-air level restores the full range -- 0 whatever
    the baseline tracking happens to be, 1 when fully blocked -- and, like M7, it self-calibrates,
    so a joint that tracks badly because it is stiff, slow or dirty does not read as permanently
    in contact.
    """
    n = _win(tr.dt, window_s)
    core = _stall_core(tr, n, min_cmd)
    S = np.vstack([np.full((n, N_FINGER), np.nan), core])
    ref = _running_ref(S, tr.dt)
    with np.errstate(all="ignore"):
        out = np.clip((S - ref) / np.maximum(1.0 - ref, 0.05), 0.0, 1.0)
    return _latch(out, _win(tr.dt, latch_s))


def m_tau_friction_smooth(tr, smooth_s: float = 0.15, **kw):
    """M9 -- M7 with a trailing average over a fixed duration.

    M7's separability (AUC ~0.78) trails the raw torque's (~0.86) purely because its normaliser is
    estimated online and carries its own noise. Averaging the RATIO over a fixed time window --
    fixed in seconds, so the sample count tracks the physics rate and the value does not -- trades
    a little response latency for that noise, keeping the stiffness invariance M7 buys.
    """
    r = m_tau_over_friction(tr, **kw)
    n = _win(tr.dt, smooth_s)
    k = np.ones(n) / n
    out = np.full_like(r, np.nan)
    for c in range(r.shape[1]):
        col = r[:, c]
        good = np.isfinite(col)
        if good.sum() < n:
            continue
        filled = np.where(good, col, np.nan)
        # nan-aware moving average: average the finite samples present in each window
        num = np.convolve(np.nan_to_num(filled, nan=0.0), k, mode="same")
        den = np.convolve(good.astype(float), k, mode="same")
        with np.errstate(all="ignore"):
            out[:, c] = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    return out


METRICS = {
    "total (obs today)": m_total,
    "servo": m_servo,
    "tau": m_tau,
    "M3 stall_ratio": m_stall_ratio,
    "M4 vel_deficit": m_vel_deficit,
    "M5 gap_ratio": m_gap_ratio,
    "M6 norm_servo": m_norm_servo,
    "M7 tau/friction": m_tau_over_friction,
    "M8 stall_norm": m_stall_norm,
    "M9 tau/fric smooth": m_tau_friction_smooth,
}
