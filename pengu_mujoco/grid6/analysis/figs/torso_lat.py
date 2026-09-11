#!/usr/bin/env python
"""Stability/attitude figure C — does the torso alternate sides? Three
aligned time-series over a few cycles:

  1. smoothed FnL - FnR (support dominance)
  2. lateral offset of (torso COM - whole-body COM), facing frame [mm]
  3. torso roll [deg], gravity method relative to the rest pose

NOTE: an earlier draft measured "roll" from the torso body z-axis, which on
this model lies HORIZONTAL at rest (the beam axis) — that number was the
beam-axis lateral swing, not roll. This script uses the gravity method
(rotation since the rest pose applied to the vertical), matching the GRID-5
imu_roll definition.

usage: python grid5/analysis/figs/torso_lat.py [gait args]
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
    Rt0 = data.xmat[tid].reshape(3, 3).copy()          # rest pose reference
    f6 = np.zeros(6); rows = []
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        Fn = mc.foot_forces(model, data, ids, f6)
        fh = mc.heading(model, data, rid)
        left = np.cross([0, 0, 1.0], fh)
        d = data.xipos[tid] - data.subtree_com[0]
        Rrel = data.xmat[tid].reshape(3, 3) @ Rt0.T
        u = Rrel @ np.array([0.0, 0.0, 1.0])
        roll = math.degrees(math.asin(np.clip(np.dot(u, left), -1, 1)))
        rows.append((data.time, Fn["L"] - Fn["R"], float(d @ left), roll))
    A = np.array(rows); t = A[:, 0]
    dom = mc.smooth_dom(t, A[:, 1], np.zeros(len(t)))   # A[:,1] = FnL - FnR
    lat = A[:, 2] * 1000; roll = A[:, 3]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    mm = (t > gs.SETTLE + 2 * cyc) & (t < gs.SETTLE + 6 * cyc)
    axes[0].plot(t[mm], dom[mm], "k-", lw=1)
    axes[0].axhline(0, color="gray", lw=.5)
    axes[0].set_ylabel("FnL - FnR [N]\n(>0 = L support)")
    axes[1].plot(t[mm], lat[mm], "-", color="#cc79a7", lw=1.5)
    axes[1].axhline(0, color="gray", lw=.5)
    axes[1].set_ylabel("torsoCOM - bodyCOM\nlateral [mm] (+=left)")
    axes[2].plot(t[mm], roll[mm], "-", color="#2c3e50", lw=1.5)
    axes[2].axhline(0, color="gray", lw=.5)
    axes[2].set_ylabel("torso roll [deg]\n(gravity, rel. rest)")
    axes[2].set_xlabel("t [s]")
    for ax_ in axes:
        ax_.grid(alpha=0.3)
    fig.suptitle(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — torso "
                 "side alternation (heading frame)", fontsize=11)
    plt.tight_layout()
    out = a.out or os.path.join(
        mc.OUT_DIR, f"torso_lat_{a.cfg}_mu{f'{a.mu:g}'.replace('0.','0')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
