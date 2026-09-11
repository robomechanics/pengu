#!/usr/bin/env python
"""F12 cone utilization — how close to the friction cone does each config
operate? Median cone_util_p95 among passing cells vs mu (IQR band), one
line per config, 1.0 reference = pegged at the cone.

cone_util = |Ft| / (mu_trial * Fn) per loaded contact point; the map stores
p50/p95 per cell. GRID-5 only (GRID-4 carried no GRF columns).

usage: python grid5/analysis/figs/cone_util.py [--configs ...]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()
    cfgs = args.configs or [c for c in style5.CONFIGS
                            if load5._csv_path(c, "grid5")]
    grids = {}
    for c in cfgs:
        try:
            grids[c] = load5.load(c, rnd="grid5")
        except (FileNotFoundError, ValueError) as e:
            print(f"  skip {c}: {e}")
    if not grids:
        sys.exit("no grid5 configs loadable")
    load5.compatible(grids.values())
    mus = next(iter(grids.values())).axes["mu"]
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c, g in grids.items():
        k, com = style5.CONFIGS[c]
        med, lo, hi = [], [], []
        for m in range(len(mus)):
            cu = g["cone_util_p95"][m]
            vals = cu[(g["pass_rate"][m] > 0) & np.isfinite(cu)]
            if vals.size:
                med.append(np.median(vals))
                lo.append(np.percentile(vals, 25))
                hi.append(np.percentile(vals, 75))
            else:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        st = style5.style_for(k, com)
        ax.plot(mus, med, **st,
                label=style5.label_for(c)
                + (" [PARTIAL]" if not g.complete else ""))
        ax.fill_between(mus, lo, hi, color=st["color"], alpha=0.15, lw=0)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax.annotate("1.0 = pegged at the friction cone (slipping)", (mus[0], 1.005),
                fontsize=8, color="gray", va="bottom")
    ax.set_xlabel("floor friction μ"); ax.set_xticks(mus)
    ax.set_ylabel("cone_util_p95  = p95 of |Ft| / (μ·Fn)")
    ax.set_title("friction-cone utilization of passing cells", fontsize=11)
    ax.grid(alpha=0.3)
    style5.legend_combined(ax, coms=sorted({style5.CONFIGS[c][1]
                                            for c in grids}))
    note = "median over passers, band = IQR"
    if partial:
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in grids[c].present['hip_off']]})"
            for c in partial)
    style5.finish(
        fig, os.path.join(args.out, "cone_util_grid5.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier="pass (pass_rate > 0)",
        stat="median cone_util_p95 per (config, μ), IQR band",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
