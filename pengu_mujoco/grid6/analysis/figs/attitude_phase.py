#!/usr/bin/env python
"""Stability/attitude figure D — torso attitude over one gait cycle:
roll (solid) / pitch (dashed) / yaw (dotted) in degrees vs gait-cycle
percentage (-10..90, 0% = L mid-stance), with L/R support shading.

Angles use the gravity method relative to the rest pose (the torso body
frame is NOT z-up on this model; see torso_lat.py note), matching the
GRID-5 imu_roll/imu_pitch definitions. Yaw is the rotation of the rest
heading, mean-centered. Square format.

usage: python grid5/analysis/figs/attitude_phase.py [gait args]
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mech_common as mc
import gait_config as gc
import gait_sweep as gs


def main():
    ap = argparse.ArgumentParser()
    mc.add_gait_args(ap)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cyc = 1.0 / a.freq

    model, data, ids, act, kappa, com = mc.build(a)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    mujoco.mj_forward(model, data)
    Rt0 = data.xmat[tid].reshape(3, 3).copy()
    fh0 = mc.heading(model, data, rid)
    f6 = np.zeros(6); rows = []
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        Fn = mc.foot_forces(model, data, ids, f6)
        fh = mc.heading(model, data, rid)
        left = np.cross([0, 0, 1.0], fh)
        Rrel = data.xmat[tid].reshape(3, 3) @ Rt0.T
        u = Rrel @ np.array([0.0, 0.0, 1.0])
        roll = math.degrees(math.asin(np.clip(np.dot(u, left), -1, 1)))
        pitch = math.degrees(math.asin(np.clip(np.dot(u, fh), -1, 1)))
        f2 = Rrel @ fh0
        yaw = math.degrees(math.atan2(f2[1], f2[0]))
        rows.append((data.time, Fn["L"] - Fn["R"], roll, pitch, yaw))
    A = np.array(rows); t = A[:, 0]
    dom = mc.smooth_dom(t, A[:, 1], np.zeros(len(t)))
    cL = mc.stance_centers(t, dom, cyc, "L")
    t0 = cL[1] if len(cL) > 1 else cL[0]
    m = (t >= t0 - 0.10 * cyc) & (t <= t0 + 0.90 * cyc)
    ph = 100.0 * (t[m] - t0) / cyc
    roll, pitch, yaw = A[m, 2], A[m, 3], A[m, 4]
    yaw = yaw - yaw.mean()

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    domm = dom[m]
    lim = 1.1 * max(np.abs(np.concatenate([roll, pitch, yaw])))
    ax.fill_between(ph, -lim, lim, where=domm > 2, color="#dce9f5", zorder=0)
    ax.fill_between(ph, -lim, lim, where=domm < -2, color="#f5e3dc", zorder=0)
    ax.plot(ph, roll, "-", color="black", lw=2.0,
            label="torso roll (+ = lean left, rel. rest pose)")
    ax.plot(ph, pitch, "--", color="black", lw=1.8,
            label="torso pitch (+ = lean fwd, rel. rest pose)")
    ax.plot(ph, yaw, ":", color="black", lw=1.8,
            label="torso yaw (dev. from mean heading)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlim(-10, 90); ax.set_ylim(-lim, lim)
    ax.set_xlabel("gait cycle [%]  (0% = L mid-stance)")
    ax.set_ylabel("torso attitude [deg]")
    ax.set_title(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — torso "
                 "attitude over one gait cycle\nshading: blue = L support, "
                 "red = R support", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = a.out or os.path.join(
        mc.OUT_DIR,
        f"attitude_phase_{a.cfg}_mu{f'{a.mu:g}'.replace('0.','0')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
