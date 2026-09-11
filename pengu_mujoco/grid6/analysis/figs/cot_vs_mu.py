#!/usr/bin/env python
"""Mechanical cost of transport vs mu.

--round grid5 (default): map-level cot_net, robust eligibility
(pass_rate > 0 AND nbhd-mean pass >= 0.8). Three panels:
mean of all eligible | top-20 mean (lowest) | champion (single lowest).
cot_net is NaN when net < 0.02 m (a stalled gait has no CoT - trap T6);
NaN cells are excluded and n is reported. No speed floor is applied
(the frozen T-cot track floors net >= 50% of T-speed #1).

--round grid4: the GRID-4 map carries no energy columns; source is the
finalists RE-RUN table results/grid4_report/fig45_metrics.csv (cot_pos,
20 rows per (config, mu), already the per-mu robust+speed selection).
The pool IS the top-20, so panels reduce to: top-20 mean | champion.
cot_pos (per-forward-distance) and cot_net (per-net-displacement) are
similar but not identical definitions - never overlay the two rounds.

usage: python grid5/analysis/figs/cot_vs_mu.py [--round grid4] [--ylim a b]
"""
import os, sys, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
TOP = 20


def cot_track(g):
    """Per mu: (mean-all, top20-mean, champion, n) over robust-eligible cells
    with finite cot_net, cot ascending (lower = better)."""
    N = g.nbhd("pass_rate")
    out = []
    for m in range(len(g.axes["mu"])):
        cot = g["cot_net"][m]
        elig = ((g["pass_rate"][m] > 0) & np.isfinite(cot)
                & np.isfinite(N[m]) & (N[m] >= 0.8))
        vals = np.sort(cot[elig])                  # ascending: best first
        if vals.size == 0:
            out.append((np.nan, np.nan, np.nan, 0))
            continue
        out.append((float(vals.mean()), float(vals[:TOP].mean()),
                    float(vals[0]), int(vals.size)))
    return out


def load_fig45():
    """GRID-4 finalists re-run table -> {(cfg, mu): [cot_pos of survivors]}."""
    path = os.path.join(_ROOT, "results", "grid4_report", "fig45_metrics.csv")
    pools = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(float(r["survived"])) != 1:
                    continue
                cot = float(r["cot_pos"])
                if not np.isfinite(cot):
                    continue
                pools.setdefault((r["config"], float(r["mu"])), []).append(cot)
            except (KeyError, ValueError):
                continue
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid5", choices=["grid4", "grid5"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    if args.round == "grid4":
        pools = load_fig45()
        cfgs = args.configs or sorted({k[0] for k in pools},
                                      key=lambda c: int(c[1:]))
        mus = sorted({k[1] for k in pools})
        data = {}
        for c in cfgs:
            rows = []
            for mu in mus:
                v = np.sort(np.array(pools.get((c, mu), [])))
                rows.append((float(v.mean()) if v.size else np.nan,
                             float(v[:TOP].mean()) if v.size else np.nan,
                             float(v[0]) if v.size else np.nan,
                             int(v.size)))
            data[c] = rows
        partial, Ks, commits = [], {1}, set()
        panels = [(1, f"top-{TOP} mean (the finalists pool)"),
                  (2, "champion (single lowest)")]
        metric = "cot_pos"
    else:
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
        data = {c: cot_track(g) for c, g in grids.items()}
        mus = next(iter(grids.values())).axes["mu"]
        partial = [c for c, g in grids.items() if not g.complete]
        Ks = {g.K for g in grids.values()}
        commits = {g.commit for g in grids.values() if g.commit}
        panels = [(0, "mean of all eligible"),
                  (1, f"top-{TOP} mean (lowest CoT)"),
                  (2, "champion (single lowest)")]
        metric = "cot_net"

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.2 * len(panels) + 0.5, 5.0),
                             sharey=True)
    for ax, (col, ttl) in zip(axes, panels):
        for c in data:
            k, com = style5.CONFIGS[c]
            ys = [d[col] for d in data[c]]
            ax.plot(mus, ys, **style5.style_for(k, com),
                    label=style5.label_for(c)
                    + (" [PARTIAL]" if c in partial else ""))
            for mu, d in zip(mus, data[c]):
                if d[3] == 0:
                    ax.annotate("none eligible", (mu, 0.5),
                                xycoords=("data", "axes fraction"),
                                fontsize=6.5, color="crimson", ha="center",
                                rotation=90)

        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("floor friction μ"); ax.set_xticks(mus); ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"mechanical CoT, {metric}  (lower = better)")
    if args.ylim:
        axes[0].set_ylim(*args.ylim)
    fig.suptitle("Mechanical cost of transport", fontsize=12)
    style5.legend_combined(axes[-1],
                           coms=sorted({style5.CONFIGS[c][1] for c in data}))

    if args.round == "grid4":
        note = ("source: fig45_metrics.csv finalists RE-RUN (per-μ robust+speed "
                "selection), survivors only; cot_pos ≠ cot_net — no cross-round overlay")
        tier = "finalists (robust top-20 per μ, re-run)"
        stat = "left: mean of the 20-row pool; right: best-of-best"
    else:
        note = "no speed floor applied (frozen T-cot floors net ≥ 50% of T-speed #1)"
        tier = "robust (pass ∧ nbhd ≥ 0.8), finite cot_net"
        stat = f"left: mean all; mid: mean of {TOP} lowest; right: best-of-best"
        if partial:
            note += "; PARTIAL " + " ".join(
                f"{c}(hip_off={[int(v) for v in grids[c].present['hip_off']]},"
                f" freq {len(grids[c].present['freq'])}/{len(grids[c].axes['freq'])})"
                for c in partial)
    style5.finish(
        fig, os.path.join(args.out, f"cot_vs_mu_{args.round}.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier=tier,
        stat=stat,
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
