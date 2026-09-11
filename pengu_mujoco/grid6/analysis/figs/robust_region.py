#!/usr/bin/env python
"""Robust-region volume vs mu — reference implementation of the frozen style
contract (style5), replacing the default-colour-cycle round-1 figure.

One line per config: colour+linestyle = gait (kappa), marker = COM ratio.
Marker shape alone separates COM within a gait; a greyscale twin lands in
bw/ automatically.

Robust cell (frozen, PLOT_GRID5.md §5): neighborhood-mean pass_rate >= 0.8
over freq ±2 x true-circular phi adjacency, divided by the count of valid
contributors. Volume covers freq x phi only — thinness on leg/hip/off axes
does not show up here (caveat stamped in the footer).

usage:
  python grid5/analysis/figs/robust_region.py                    # grid4 ref
  python grid5/analysis/figs/robust_region.py --round grid5 --frac
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # pengu_mujoco/
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
THRESH = 0.8


def robust_counts(g):
    """Per-mu robust-cell count and per-mu denominator (cells in the config's
    own grid restricted to what is actually present — matters for partials)."""
    N = g.nbhd("pass_rate")
    cnt, den = [], []
    for m in range(len(g.axes["mu"])):
        v = N[m]
        cnt.append(int(np.nansum(v >= THRESH)))
        den.append(int(np.isfinite(v).sum()))
    return cnt, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid4", choices=["grid4", "grid5"])
    ap.add_argument("--configs", nargs="*", default=None,
                    help="default: every config with a CSV on disk")
    ap.add_argument("--frac", action="store_true",
                    help="plot fraction of the config's own grid instead of "
                         "counts (counts are NOT comparable across rounds)")
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    cfgs = args.configs or [c for c in style5.CONFIGS
                            if load5._csv_path(c, args.round)]
    grids = {}
    for c in cfgs:
        try:
            grids[c] = load5.load(c, rnd=args.round)
        except FileNotFoundError as e:
            print(f"  skip {c}: {e}")
    if not grids:
        sys.exit("no configs loadable")
    load5.compatible(grids.values())

    partial = [c for c, g in grids.items() if not g.complete]
    data = {c: robust_counts(g) for c, g in grids.items()}
    mus = next(iter(grids.values())).axes["mu"]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for c, g in grids.items():
        k, com = style5.CONFIGS[c]
        cnt, den = data[c]
        ys = ([n / d if d else np.nan for n, d in zip(cnt, den)]
              if args.frac else cnt)
        lbl = style5.label_for(c) + (" [PARTIAL]" if not g.complete else "")
        ax.plot(mus, ys, **style5.style_for(k, com), label=lbl)
        if not g.complete:                       # mark, never hide
            ax.annotate("partial", (mus[-1], ys[-1]), fontsize=6.5,
                        color="crimson", xytext=(4, 0),
                        textcoords="offset points")
    ax.set_xlabel("floor friction μ"); ax.set_xticks(mus)
    if args.frac:
        ax.set_ylabel("robust fraction of own grid (per μ)")
    else:
        ax.set_ylabel("robust cells (count)")
        ax.set_yscale("symlog", linthresh=10)
        top = max(max(d[0]) for d in data.values())
        ax.set_ylim(0, top * 1.6)
    ax.set_title(f"{args.round.upper()} robust-region volume "
                 f"(nbhd-mean pass ≥ {THRESH})", fontsize=11)
    ax.grid(alpha=0.3)
    style5.legend_two(ax, coms=sorted({style5.CONFIGS[c][1] for c in grids}))
    note = "freq×phi neighborhood only"
    if partial:
        pres = {c: grids[c].present["hip_off"] for c in partial}
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in p]})" for c, p in pres.items())
    style5.finish(
        fig, os.path.join(args.out, f"robust_region_{args.round}.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier=f"robust (nbhd ≥ {THRESH})",
        stat=("fraction of own grid" if args.frac else
              "count of robust cells") + " per (config, μ)",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
