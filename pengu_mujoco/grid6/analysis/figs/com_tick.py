#!/usr/bin/env python
"""Stability/attitude figure A — "tick": front profile + top-down stacked
silhouettes over one gait cycle, showing that the torso COM swings wide
while the whole-body COM stays over the support foot.

Left panel : front-view render at the R mid-stance instant with markers
             (whole-body COM star, torso COM triangle, support foot) and the
             support-foot -> COM pendulum line vs the vertical.
Right panel: top-down composition, three postures (L / R / L mid-stance,
             labelled as 0% / 50% / 100% of the cycle at the far left),
             synthetic footprints (feet are occluded from above), COM paths,
             per-posture torso lateral offsets.

Runs one short nominal simulation (see mech_common). Camera azimuth is
aligned to the walking direction (forward = image up).

usage:
  python grid5/analysis/figs/com_tick.py                  # default gait, floor
  python grid5/analysis/figs/com_tick.py --bg light|white|floor
  python grid5/analysis/figs/com_tick.py --cfg c3 --freq 1.61 --phi 330 \
         --leg 115 --hip 28 --off 10                      # kappa=0 contrast
"""
import os, sys, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import mech_common as mc
import gait_config as gc
import gait_sweep as gs

H, W = 480, 640
CAM_D = 1.05


def main():
    ap = argparse.ArgumentParser()
    mc.add_gait_args(ap)
    ap.add_argument("--bg", default="floor", choices=["floor", "white", "light"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cyc = 1.0 / a.freq

    # ---- pass 1: timeline -> key instants (L / R / L mid-stance)
    model, data, ids, act, kappa, com = mc.build(a)
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    FOOT = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n, s in gs.FOOT_BODIES.items()}
    f6 = np.zeros(6); rows = []
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        Fn = mc.foot_forces(model, data, ids, f6)
        rows.append((data.time, Fn["L"], Fn["R"],
                     *data.subtree_com[0][:2], *data.xipos[tid][:2],
                     *data.xpos[FOOT["L"]][:2], *data.xpos[FOOT["R"]][:2]))
    A = np.array(rows); t = A[:, 0]
    dom = mc.smooth_dom(t, A[:, 1], A[:, 2])
    cL = mc.stance_centers(t, dom, cyc, "L")
    cR = mc.stance_centers(t, dom, cyc, "R")
    mids, sides = None, ["L", "R", "L"]
    for l0 in cL:
        rs = [r for r in cR if l0 < r < l0 + cyc]
        if rs:
            l1s = [x for x in cL if rs[0] < x < rs[0] + cyc]
            if l1s:
                mids = [l0, rs[0], l1s[0]]
                break
    if mids is None:
        sys.exit("no clean L-R-L stance sequence found for this gait")
    w0, w1 = mids[0] - 0.05, mids[-1] + 0.05
    mwin = (t >= w0) & (t <= w1)
    look = np.array([A[mwin, 3].mean(), A[mwin, 4].mean(), 0.10])
    dxy = np.array([A[mwin, 3][-1] - A[mwin, 3][0],
                    A[mwin, 4][-1] - A[mwin, 4][0]])
    AZ = math.degrees(math.atan2(dxy[1], dxy[0]))

    # ---- pass 2: deterministic replay, render
    model, data, ids, act, kappa, com = mc.build(a)
    floor_id = ids[0]
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = CAM_D, -89.9, AZ
    snaps, bg = [], None
    with mujoco.Renderer(model, height=H, width=W) as ren:
        k = 0
        while data.time < w1 + 0.05 and k < len(mids):
            gc.apply_ctrl(data, act, data.time)
            mujoco.mj_step(model, data)
            if bg is None and data.time >= mids[0] - 0.05:
                q0 = data.qpos.copy()
                data.qpos[0] += 50.0
                mujoco.mj_forward(model, data)
                cam.lookat[:] = look
                ren.update_scene(data, cam)
                bg = ren.render().copy().astype(float) / 255.0
                data.qpos[:] = q0
                mujoco.mj_forward(model, data)
            if data.time >= mids[k]:
                fh0 = mc.heading(model, data, rid)
                az_f = math.degrees(math.atan2(fh0[1], fh0[0])) + 180.0
                fcam = mujoco.MjvCamera()
                fcam.type = mujoco.mjtCamera.mjCAMERA_FREE
                flook = data.subtree_com[0].copy(); flook[2] = 0.16
                fcam.lookat[:] = flook
                fcam.distance, fcam.elevation, fcam.azimuth = 0.75, -8.0, az_f
                ren.update_scene(data, fcam)
                front_img = ren.render().copy()
                cam.lookat[:] = look
                ren.update_scene(data, cam)
                ren.enable_segmentation_rendering()
                seg = ren.render().copy()
                ren.disable_segmentation_rendering()
                gid = seg[:, :, 0]
                mask = (gid >= 0) & (gid != floor_id)
                snaps.append(dict(
                    t=data.time, mask=mask, fh=fh0, front=front_img,
                    faz=az_f, flook=flook,
                    com=data.subtree_com[0].copy(),
                    tor=data.xipos[tid].copy(),
                    pL=data.xpos[FOOT["L"]].copy(),
                    pR=data.xpos[FOOT["R"]].copy()))
                k += 1

    P = lambda p: mc.project(p, look, CAM_D, AZ, -89.9, W, H)
    if a.bg == "floor":
        canvas = bg.copy()
        for s, a_ in zip(snaps, [0.45, 0.62, 0.82]):
            canvas[s["mask"]] = canvas[s["mask"]] * (1 - a_) + 0.93 * a_
    elif a.bg == "light":
        canvas = bg.copy() * 0.45 + 0.55
        for s, a_ in zip(snaps, [0.60, 0.78, 0.95]):
            canvas[s["mask"]] = canvas[s["mask"]] * (1 - a_) + 0.995 * a_
    else:
        canvas = np.ones((H, W, 3))
        for s, a_ in zip(snaps, [0.16, 0.26, 0.38]):
            canvas[s["mask"]] = np.clip(canvas[s["mask"]] * (1 - a_), 0, 1)

    fig = plt.figure(figsize=(15.5, 8.0))
    gsp = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15], wspace=0.04)
    axF = fig.add_subplot(gsp[0]); ax = fig.add_subplot(gsp[1])

    # ---- front profile (middle instant = R support)
    sF, sideF = snaps[1], sides[1]
    axF.imshow(sF["front"]); axF.set_xticks([]); axF.set_yticks([])
    PF = lambda p: mc.project(p, sF["flook"], 0.75, sF["faz"], -8.0, W, H)
    supF = sF["pL"] if sideF == "L" else sF["pR"]
    fg_ = np.array([supF[0], supF[1], 0.02])
    ucom, vcom = PF(sF["com"]); utor, vtor = PF(sF["tor"]); ufo, vfo = PF(fg_)
    uvv = PF(fg_ + np.array([0, 0, 0.30]))
    axF.plot([ufo, uvv[0]], [vfo, uvv[1]], ":", color="cyan", lw=1.6)
    axF.plot([ufo, ucom], [vfo, vcom], "-", color="#ffdd44", lw=2.0)
    axF.plot([ucom, utor], [vcom, vtor], "-", color="#cc79a7", lw=1.5)
    axF.plot(ucom, vcom, "*", color="#ffdd44", ms=26, mec="black", mew=1.5)
    axF.plot(utor, vtor, "^", color="#cc79a7", ms=14, mec="black", mew=1.2)
    axF.plot(ufo, vfo, "s", color="black", ms=10, mec="white", mew=1.2)
    axF.annotate("whole-body COM", (ucom, vcom), fontsize=9, color="white",
                 xytext=(-115, 6), textcoords="offset points")
    axF.annotate("torso COM", (utor, vtor), fontsize=9, color="white",
                 xytext=(10, 4), textcoords="offset points")
    axF.annotate(f"support foot ({sideF})", (ufo, vfo), fontsize=9,
                 color="white", xytext=(8, -14), textcoords="offset points")
    axF.set_title("front view @ %s support (50%% of cycle)\n"
                  "yellow: support-foot to COM pendulum;  dotted: vertical"
                  % sideF, fontsize=10)

    # ---- top-down
    ax.imshow(canvas); ax.set_xticks([]); ax.set_yticks([])
    Zup = np.array([0.0, 0.0, 1.0])
    for s, side in zip(snaps, sides):        # support footprints
        left = np.cross(Zup, s["fh"])
        p0 = s["pL"] if side == "L" else s["pR"]
        corners = [p0 + i * 0.055 * s["fh"] + j * 0.030 * left
                   for i, j in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        ax.add_patch(Polygon([P((c[0], c[1], 0.02)) for c in corners],
                             closed=True, facecolor="black",
                             edgecolor="black", lw=1.2, alpha=0.85, zorder=3))
    uv = np.array([P((x, y, 0.19)) for x, y in zip(A[mwin, 3], A[mwin, 4])])
    ax.plot(uv[:, 0], uv[:, 1], "-", color="#e69f00", lw=2.6,
            label="whole-body COM path")
    uvt = np.array([P((x, y, 0.30)) for x, y in zip(A[mwin, 5], A[mwin, 6])])
    ax.plot(uvt[:, 0], uvt[:, 1], "--", color="#cc79a7", lw=2.0,
            label="torso COM path")
    for s, side in zip(snaps, sides):
        uc, vc = P(s["com"]); ut, vt = P(s["tor"])
        ax.plot([uc, ut], [vc, vt], "-", color="#cc79a7", lw=1.4, alpha=0.8)
        ax.plot(uc, vc, "*", color="#ffdd44", ms=24, mec="black", mew=1.4,
                zorder=5)
        ph = 100.0 * (s["t"] - mids[0]) / cyc
        ax.annotate(f"{ph:.0f}% — {side} support", (0.02, vc),
                    xycoords=("axes fraction", "data"), fontsize=9,
                    color="black", va="center", ha="left", zorder=6)
        ax.plot(ut, vt, "^", color="#cc79a7", ms=13, mec="black", mew=1.2,
                zorder=5)
        lft = np.cross(Zup, s["fh"])
        dlat = float((s["tor"] - s["com"]) @ lft) * 1000
        ax.annotate(f"torso {dlat:+.0f} mm", (ut, vt), fontsize=8,
                    color="#a04c7f", xytext=(8, 8),
                    textcoords="offset points", zorder=6)
    ax.plot([], [], "*", color="#ffdd44", ms=14, mec="black",
            label="whole-body COM @ single support")
    ax.plot([], [], "^", color="#cc79a7", ms=9, mec="black",
            label="torso COM @ single support")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — one "
                 "gait cycle, top-down\ntorso swings wide, whole-body COM "
                 "stays over the support foot", fontsize=11)
    plt.tight_layout()
    out = a.out or os.path.join(
        mc.OUT_DIR, f"com_tick_{a.cfg}_mu{f'{a.mu:g}'.replace('0.','0')}"
        + {"floor": "", "white": "_white", "light": "_light"}[a.bg] + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
