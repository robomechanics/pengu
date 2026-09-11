#!/usr/bin/env python
"""GRF friction-cone scatter — the slippery-test mechanism picture: per-foot
(Fn, |Ft|) of every loaded sample over the walk, one panel per mu, with the
friction cone |Ft| = mu*Fn drawn from the origin. A cloud pressed against
the cone line is a gait spending its stance at the traction limit; a cloud
well below it has margin. The share of samples within 5% of the cone
(the sweep's SLIP_CONE_EPS) is annotated per panel.

Runs one short nominal simulation per mu (see mech_common).

usage: python grid5/analysis/figs/grf_cone.py [--mus 0.1 0.7] [gait args]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mujoco
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mech_common as mc
import gait_config as gc
import gait_sweep as gs

COL = {"L": "#2c3e50", "R": "#7f8c8d"}
CONE_EPS = 0.05          # = grid5 SLIP_CONE_EPS (frozen); physics/ has none


def collect(a, mu):
    """(Fn, Ft) per foot per sample while loaded, after SETTLE."""
    import copy
    a2 = argparse.Namespace(**vars(a)); a2.mu = mu
    model, data, ids, act, kappa, com = mc.build(a2)
    floor_id, foot_geom, _, _ = ids
    f6 = np.zeros(6); cyc = 1.0 / a.freq
    pts = {"L": [], "R": []}
    fell = False
    while data.time < gs.SETTLE + 8 * cyc:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        rootz = data.xpos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)][2]
        if rootz < 0.05:
            fell = True
            break
        if data.time < gs.SETTLE:
            continue
        Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}
        for c in range(data.ncon):
            ct = data.contact[c]
            fg = ct.geom2 if ct.geom1 == floor_id else (
                 ct.geom1 if ct.geom2 == floor_id else -1)
            ft = foot_geom.get(fg)
            if ft:
                mujoco.mj_contactForce(model, data, c, f6)
                Fn[ft] += abs(f6[0])
                Ft[ft] += math.hypot(f6[1], f6[2])
        for s in ("L", "R"):
            if Fn[s] > gs.F_HI:
                pts[s].append((Fn[s], Ft[s]))
    return {s: np.array(v) for s, v in pts.items()}, kappa, com, fell


def main():
    ap = argparse.ArgumentParser()
    mc.add_gait_args(ap)
    ap.add_argument("--mus", nargs="*", type=float, default=[0.1, 0.7])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results, fells = {}, {}
    for mu in a.mus:
        results[mu], kappa, com, fells[mu] = collect(a, mu)
        n = sum(len(v) for v in results[mu].values())
        print(f"mu={mu}: {n} loaded samples" + ("  (FELL)" if fells[mu] else ""))

    fig, axes = plt.subplots(1, len(a.mus), figsize=(5.6 * len(a.mus), 5.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    nonempty = [v[:, 0].max() for r in results.values()
                for v in r.values() if v.size]
    fmax = max(nonempty) if nonempty else 50.0
    for ax, mu in zip(axes, a.mus):
        r = results[mu]
        xs = np.array([0, fmax * 1.05])
        ax.fill_between(xs, mu * xs, fmax * 10, color="#f5dcdc", zorder=0)
        ax.plot(xs, mu * xs, "-", color="#b03030", lw=1.8,
                label=f"friction cone |Ft| = μ·Fn (μ={mu:g})")
        ax.plot(xs, (1 - CONE_EPS) * mu * xs, ":", color="#b03030",
                lw=1.2, label="cone − 5% (slip criterion)")
        have = [v for v in r.values() if v.size]
        if have:
            allpts = np.vstack(have)
            for s in ("L", "R"):
                if r[s].size:
                    ax.plot(r[s][:, 0], r[s][:, 1], ".", ms=2.5, alpha=0.35,
                            color=COL[s], label=f"{s} foot samples")
            near = np.mean(allpts[:, 1] >= (1 - CONE_EPS) * mu
                           * allpts[:, 0]) * 100
            ax.annotate(f"{near:.1f}% of samples at the cone",
                        (0.03, 0.95), xycoords="axes fraction", fontsize=9,
                        color="#b03030")
        else:
            ax.annotate("FELL before measurement\n(no loaded samples)",
                        (0.5, 0.5), xycoords="axes fraction", fontsize=11,
                        color="crimson", ha="center")
        ax.set_title(f"μ = {mu:g}", fontsize=11)
        ax.set_xlabel("normal force Fn [N]")
        ax.set_xlim(0, fmax * 1.05)
        ax.set_ylim(0, fmax * 0.8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|tangential force Ft| [N]")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{a.cfg} (κ={kappa:g}, COM {com:.2f}) — ground reaction "
                 "forces vs the friction cone, same gait at each μ",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = a.out or os.path.join(
        mc.OUT_DIR, f"grf_cone_{a.cfg}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
