#!/usr/bin/env python
"""Stability/attitude figure E — foot clearance in sim.

Clearance = (min foot-geom center z) minus that foot's mean height while
loaded, so the curve reads ~0 during stance and peaks at the swing apex.

Two styles:
  --style stride (default): vertical vs HORIZONTAL foot displacement over
      one gait cycle (the classic human-gait presentation: plain rectangle
      frame, max-clearance circled, stride length annotated)
  --style phase: clearance vs gait-cycle percentage with L/R support shading

usage: python grid5/analysis/figs/foot_clearance.py [--style stride|phase]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mech_common as mc
import gait_config as gc
import gait_sweep as gs

COL = {"L": "#2c3e50", "R": "#7f8c8d"}
LS = {"L": "-", "R": "--"}


def main():
    ap = argparse.ArgumentParser()
    mc.add_gait_args(ap)
    ap.add_argument("--style", default="stride", choices=["stride", "phase"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cyc = 1.0 / a.freq

    model, data, ids, act, kappa, com = mc.build(a)
    floor_id, foot_geom, foot_bid, root = ids
    zgeoms = {s: [g for g, sd in foot_geom.items() if sd == s]
              for s in ("L", "R")}
    FOOT = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n, s in gs.FOOT_BODIES.items()}
    f6 = np.zeros(6); rows = []
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        Fn = mc.foot_forces(model, data, ids, f6)
        zL = min(data.geom_xpos[g][2] for g in zgeoms["L"])
        zR = min(data.geom_xpos[g][2] for g in zgeoms["R"])
        rows.append((data.time, Fn["L"], Fn["R"], zL, zR,
                     *data.xpos[FOOT["L"]][:2], *data.xpos[FOOT["R"]][:2]))
    A = np.array(rows); t = A[:, 0]
    dom = mc.smooth_dom(t, A[:, 1], A[:, 2])
    cL = mc.stance_centers(t, dom, cyc, "L")
    t0 = cL[1] if len(cL) > 1 else cL[0]
    m = (t >= t0 - 0.10 * cyc) & (t <= t0 + 0.90 * cyc)
    ph = 100.0 * (t[m] - t0) / cyc

    base = {s: A[(A[:, 1 if s == "L" else 2] > 4) & (t > gs.SETTLE),
                 3 if s == "L" else 4].mean() for s in ("L", "R")}
    clr = {s: (A[m, 3 if s == "L" else 4] - base[s]) * 1000
           for s in ("L", "R")}
    lo = min(clr["L"].min(), clr["R"].min()) - 2       # full range, no clip
    top = 1.15 * max(clr["L"].max(), clr["R"].max())

    if a.style == "phase":
        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        domm = dom[m]
        ax.fill_between(ph, lo, top, where=domm > 2, color="#dce9f5", zorder=0)
        ax.fill_between(ph, lo, top, where=domm < -2, color="#f5e3dc", zorder=0)
        for s in ("L", "R"):
            ax.plot(ph, clr[s], LS[s], color=COL[s], lw=2.0, label=f"{s} foot")
            i = int(np.argmax(clr[s]))
            ax.annotate(f"{clr[s][i]:.0f} mm", (ph[i], clr[s][i]),
                        fontsize=9, color=COL[s], ha="center",
                        xytext=(0, 8), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlim(-10, 90); ax.set_ylim(lo, top)
        ax.set_xlabel("gait cycle [%]  (0% = L mid-stance)")
        ax.set_ylabel("foot clearance above stance height [mm]")
        ax.set_title(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — "
                     "foot clearance over one gait cycle\nshading: blue = L "
                     "support, red = R support; labels = swing apex",
                     fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
    else:
        # stride style: vertical vs horizontal displacement, rectangle frame
        dxy = np.array([A[m, 5][-1] - A[m, 5][0], A[m, 6][-1] - A[m, 6][0]])
        fh_m = dxy / max(np.linalg.norm(dxy), 1e-9)
        fig, ax = plt.subplots(figsize=(9.2, 5.4))
        for s, ix in (("L", 5), ("R", 7)):
            fxy = A[m, ix:ix + 2]
            fwd = (fxy - fxy[0]) @ fh_m * 1000
            ax.plot(fwd, clr[s], LS[s], color=COL[s], lw=2.0,
                    label=f"{s} foot")
            i = int(np.argmax(clr[s]))
            ax.plot(fwd[i], clr[s][i], "o", ms=13, mfc="none", mec=COL[s],
                    mew=1.6)
            ax.annotate(f"max clearance {clr[s][i]:.0f} mm",
                        (fwd[i], clr[s][i]), fontsize=9, color=COL[s],
                        xytext=(8, 6), textcoords="offset points")
            if s == "L":
                ax.annotate(f"stride length {fwd[-1] - fwd[0]:.0f} mm",
                            (fwd[-1], clr[s][-1]), fontsize=9, color=COL[s],
                            ha="right", xytext=(-4, -14),
                            textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylim(lo, top)
        ax.set_xlabel("horizontal displacement [mm]  (walking direction)")
        ax.set_ylabel("vertical displacement [mm]\n(above stance height)")
        ax.set_title(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) @ μ={a.mu:g} — "
                     "foot clearance over one stride", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(False)

    plt.tight_layout()
    out = a.out or os.path.join(
        mc.OUT_DIR,
        f"foot_clearance_{a.cfg}_mu{f'{a.mu:g}'.replace('0.','0')}"
        + ("_phase" if a.style == "phase" else "") + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
