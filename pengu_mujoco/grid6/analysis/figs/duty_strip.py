#!/usr/bin/env python
"""Stability/attitude figure B — duty-cycle strip: one gait cycle as a
left-to-right series of top-down screenshots with COM markers, aligned with
the support duty bar and the lateral COM-shift trace.

Row 1: N+1 top-down frames at evenly spaced phases (0..100%), camera
       follows the robot, orientation fixed to the walking direction.
       Markers: whole-body COM (star), torso COM (triangle).
Row 2: force-dominance duty bar (L / R support) over phase.
Row 3: lateral shift of both COMs (path frame) over phase.

Runs one short nominal simulation (see mech_common).

usage: python grid5/analysis/figs/duty_strip.py [--nframes 8] [gait args]
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

H, W = 480, 480
CROP = 190
CAM_D = 0.62


def main():
    ap = argparse.ArgumentParser()
    mc.add_gait_args(ap)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cyc = 1.0 / a.freq
    NFR = a.nframes

    model, data, ids, act, kappa, com = mc.build(a)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    f6 = np.zeros(6); rows = []
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        Fn = mc.foot_forces(model, data, ids, f6)
        rows.append((data.time, Fn["L"], Fn["R"], *data.subtree_com[0],
                     *data.xipos[tid], *data.xpos[rid][:2]))
    A = np.array(rows); t = A[:, 0]
    dom = mc.smooth_dom(t, A[:, 1], A[:, 2])
    cL = mc.stance_centers(t, dom, cyc, "L")
    t0 = cL[0]; t1 = t0 + cyc
    mwin = (t >= t0) & (t <= t1)
    frame_ts = [t0 + k * cyc / NFR for k in range(NFR + 1)]
    dxy = np.array([A[mwin, 3][-1] - A[mwin, 3][0],
                    A[mwin, 4][-1] - A[mwin, 4][0]])
    AZ = math.degrees(math.atan2(dxy[1], dxy[0]))
    fh_m = dxy / max(np.linalg.norm(dxy), 1e-9)
    left_m = np.array([-fh_m[1], fh_m[0]])

    model, data, ids, act, kappa, com = mc.build(a)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = CAM_D, -89.9, AZ
    frames = []
    with mujoco.Renderer(model, height=H, width=W) as ren:
        k = 0
        while data.time < t1 + 0.05 and k < len(frame_ts):
            gc.apply_ctrl(data, act, data.time)
            mujoco.mj_step(model, data)
            if data.time >= frame_ts[k]:
                look = data.xpos[rid].copy()
                cam.lookat[:] = look
                ren.update_scene(data, cam)
                frames.append(dict(
                    t=data.time, img=ren.render().copy(), look=look,
                    com=data.subtree_com[0].copy(),
                    tor=data.xipos[tid].copy(),
                    Fn=dict(mc.foot_forces(model, data, ids, f6))))
                k += 1

    fig = plt.figure(figsize=(2.1 * (NFR + 1), 6.4))
    gsp = fig.add_gridspec(3, NFR + 1, height_ratios=[3.1, 0.75, 1.35],
                           hspace=0.16, wspace=0.03)
    for i, fr in enumerate(frames):
        ax = fig.add_subplot(gsp[0, i])
        ax.imshow(fr["img"][H//2-CROP:H//2+CROP, W//2-CROP:W//2+CROP])
        for p, c, mk, ms in [(fr["tor"], "#cc79a7", "^", 11),
                             (fr["com"], "#ffdd44", "*", 17)]:
            u, v = mc.project(p, fr["look"], CAM_D, AZ, -89.9, W, H)
            ax.plot(u - (W//2-CROP), v - (H//2-CROP), mk, color=c, ms=ms,
                    mec="black", mew=1.1)
        ax.set_xticks([]); ax.set_yticks([])
        ph = 100.0 * (fr["t"] - t0) / cyc
        sup = ("L" if fr["Fn"]["L"] > 4 and fr["Fn"]["R"] < 4 else
               "R" if fr["Fn"]["R"] > 4 and fr["Fn"]["L"] < 4 else
               "D" if fr["Fn"]["L"] > 4 else "air")
        ax.set_title(f"{ph:.0f}%  [{sup}]", fontsize=9)

    axd = fig.add_subplot(gsp[1, :])
    ph_t = 100.0 * (t[mwin] - t0) / cyc
    axd.fill_between(ph_t, 1.05, 1.95, where=dom[mwin] > 2,
                     color="#2c3e50", step="mid")
    axd.fill_between(ph_t, 0.05, 0.95, where=dom[mwin] < -2,
                     color="#7f8c8d", step="mid")
    axd.set_yticks([0.5, 1.5]); axd.set_yticklabels(["R", "L"], fontsize=9)
    axd.set_ylim(0, 2); axd.set_xlim(0, 100); axd.set_xticks([])
    axd.set_ylabel("support", fontsize=8)
    for k in range(NFR + 1):
        axd.axvline(100 * k / NFR, color="white", lw=0.6)

    axl = fig.add_subplot(gsp[2, :])
    com_xy = A[mwin, 3:5]; tor_xy = A[mwin, 6:8]
    ref = com_xy.mean(axis=0)
    axl.plot(ph_t, (com_xy - ref) @ left_m * 1000, "-", color="#e69f00",
             lw=2.2, label="whole-body COM")
    axl.plot(ph_t, (tor_xy - ref) @ left_m * 1000, "--", color="#cc79a7",
             lw=1.8, label="torso COM")
    axl.axhline(0, color="gray", lw=0.6)
    for k in range(NFR + 1):
        axl.axvline(100 * k / NFR, color="gray", lw=0.5, alpha=0.5)
    axl.set_xlim(0, 100); axl.set_xlabel("gait cycle phase [%]", fontsize=9)
    axl.set_ylabel("lateral shift [mm]\n(+ = left)", fontsize=8)
    axl.legend(fontsize=8, loc="upper right"); axl.grid(alpha=0.3, axis="y")
    fig.suptitle(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — one "
                 "gait cycle, top-down duty strip", fontsize=11, y=0.99)
    out = a.out or os.path.join(
        mc.OUT_DIR, f"duty_strip_{a.cfg}_mu{f'{a.mu:g}'.replace('0.','0')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
