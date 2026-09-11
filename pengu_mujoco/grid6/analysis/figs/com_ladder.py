#!/usr/bin/env python
"""F6 com_ladder — the round-2 headline: where is the COM cliff, and does
kappa move it? x = COM ratio, one line per gait (kappa), one panel per mu.

Default metric: robust fraction of the config's own grid (nbhd-mean pass
>= 0.8 over freq x phi, frozen definition in load5.Grid.nbhd). --metric
passfrac plots the raw pass_rate>0 share instead.

usage:
  python grid5/analysis/figs/com_ladder.py                   # grid4 (3 COMs)
  python grid5/analysis/figs/com_ladder.py --round grid5     # 5 COMs
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
THRESH = 0.8


def metric_per_mu(g, which):
    out = []
    if which == "robust":
        N = g.nbhd("pass_rate")
        for m in range(len(g.axes["mu"])):
            v = N[m]; den = int(np.isfinite(v).sum())
            out.append(np.nansum(v >= THRESH) / den if den else np.nan)
    else:                                     # passfrac
        for m in range(len(g.axes["mu"])):
            p = g["pass_rate"][m]; fin = np.isfinite(p)
            out.append(float((p[fin] > 0).mean()) if fin.any() else np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid4", choices=["grid4", "grid5"])
    ap.add_argument("--metric", default="robust", choices=["robust", "passfrac"])
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    grids = {}
    for c in style5.CONFIGS:
        if load5._csv_path(c, args.round):
            try:
                grids[c] = load5.load(c, rnd=args.round)
            except (FileNotFoundError, ValueError) as e:
                print(f"  skip {c}: {e}")
    if not grids:
        sys.exit("no configs loadable")
    load5.compatible(grids.values())

    vals = {c: metric_per_mu(g, args.metric) for c, g in grids.items()}
    mus = next(iter(grids.values())).axes["mu"]
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}
    ylab = ("robust fraction of own grid" if args.metric == "robust"
            else "share of cells with pass > 0")

    fig, axes = plt.subplots(1, len(mus), figsize=(3.4 * len(mus) + 1, 4.6),
                             sharey=True)
    for m, (ax, mu) in enumerate(zip(axes, mus)):
        for kappa in (0.0, 2.0):
            pts = sorted((style5.CONFIGS[c][1], c) for c in grids
                         if style5.CONFIGS[c][0] == kappa)
            xs = [p[0] for p in pts]
            ys = [vals[p[1]][m] for p in pts]
            gst = style5.GAIT[kappa]
            ax.plot(xs, ys, color=gst["color"], ls=gst["ls"], lw=1.9,
                    label=gst["name"])
            for (com, c), y in zip(pts, ys):        # marker = COM, per point
                st = style5.style_for(kappa, com)
                ax.plot([com], [y], marker=st["marker"], color=st["color"],
                        mfc=st["mfc"], mec=st["mec"], ms=st["ms"],
                        mew=st["mew"], ls="none")
                if c in partial:
                    ax.annotate("partial", (com, y), fontsize=6, color="crimson",
                                xytext=(0, 8), textcoords="offset points",
                                ha="center")
        ax.set_title(f"μ={mu:g}", fontsize=10)
        ax.set_xlabel("COM ratio")
        ax.set_xticks(sorted({style5.CONFIGS[c][1] for c in grids}))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylab)
    axes[0].legend(fontsize=8, title="gait", title_fontsize=8)

    note = "x = COM ratio; marker shape also encodes COM (contract)"
    if partial:
        note += "; PARTIAL " + " ".join(
            f"{c}({grids[c].present['hip_off']})" for c in partial)
    style5.finish(
        fig, os.path.join(args.out,
                          f"com_ladder_{args.round}_{args.metric}.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier=("robust (nbhd ≥ 0.8)" if args.metric == "robust" else
              "pass (pass_rate > 0)"),
        stat="fraction of own grid per (config, μ)",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
