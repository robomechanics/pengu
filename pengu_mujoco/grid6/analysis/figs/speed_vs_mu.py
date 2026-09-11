#!/usr/bin/env python
"""Forward speed vs mu — two panels from the same per-(config, mu) selection:

  left   top-20 mean : mean net_fwd_mean of the 20 fastest passing cells
  right  champion    : net_fwd_mean of the single fastest passing cell

Selection is the frozen T-speed track (docs/grid5_design.md step 2 /
PLOT_GRID5.md §8): eligibility = pass (pass_rate > 0), ranked by
net_fwd_mean descending, chosen PER (config, mu) — never selected at one mu
and scored at another (trap T1). The champion is a best-of-best: one cell,
often one grid step wide (trap T8) — read it next to the robust-region
figure, not alone. Fewer than 20 passers -> the point is annotated with n;
zero passers -> a gap labelled "no passers", never a zero.

usage:
  python grid5/analysis/figs/speed_vs_mu.py                  # grid4 reference
  python grid5/analysis/figs/speed_vs_mu.py --round grid5
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # pengu_mujoco/
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
TOP = 20                     # default; --top overrides (stamped in filename)


def speed_track(g, top, tier="pass"):
    """Per mu: (top-N mean, champion, n eligible) from the T-speed ranking.
    tier 'pass' = pass_rate > 0; tier 'robust' additionally requires the
    frozen neighborhood test (nbhd-mean pass >= 0.8; freq-edge NaN cells are
    excluded by construction)."""
    N = g.nbhd("pass_rate") if tier == "robust" else None
    out = []
    for m in range(len(g.axes["mu"])):
        nf = g["net_fwd_mean"][m]
        elig = (g["pass_rate"][m] > 0) & np.isfinite(nf)
        if N is not None:
            elig &= np.isfinite(N[m]) & (N[m] >= 0.8)
        vals = nf[elig]
        if vals.size == 0:
            out.append((np.nan, np.nan, 0))
            continue
        best = np.sort(vals)[-top:]
        out.append((float(best.mean()), float(best[-1]),
                    int(min(vals.size, top))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid4", choices=["grid4", "grid5"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--tier", default="pass", choices=["pass", "robust"])
    ap.add_argument("--single", action="store_true",
                    help="one panel, top-N mean only (no champion panel)")
    ap.add_argument("--ylim", nargs=2, type=float, default=None)
    ap.add_argument("--legend-loc", default="upper right",
                    help="single mode: location of the combined legend")
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

    data = {c: speed_track(g, args.top, args.tier) for c, g in grids.items()}
    mus = next(iter(grids.values())).axes["mu"]
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    if args.single:
        fig, ax1 = plt.subplots(figsize=(8.2, 5.2))
        axes = [ax1]
        panels = [(0, f"top-{args.top} mean (per-μ selection, {args.tier} tier)")]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
        panels = [(0, f"top-{args.top} mean (per-μ selection)"),
                  (1, "champion (best-of-best)")]
    for ax, (col, ttl) in zip(axes, panels):
        for c, g in grids.items():
            k, com = style5.CONFIGS[c]
            ys = [d[col] for d in data[c]]
            ax.plot(mus, ys, **style5.style_for(k, com),
                    label=style5.label_for(c)
                    + (" [PARTIAL]" if not g.complete else ""))
            for mu, d in zip(mus, data[c]):          # annotate thin selections
                if d[2] == 0:
                    ax.annotate("no passers", (mu, 0), fontsize=6.5,
                                color="crimson", ha="center",
                                xytext=(0, -12), textcoords="offset points")
                elif d[2] < args.top:
                    ax.annotate(f"n={d[2]}", (mu, d[col]), fontsize=6.5,
                                color="gray", xytext=(4, -10),
                                textcoords="offset points")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("floor friction μ"); ax.set_xticks(mus); ax.grid(alpha=0.3)
    axes[0].set_ylabel("net_fwd_mean [m/s]")
    if args.ylim:
        axes[0].set_ylim(*args.ylim)
    if args.single:
        style5.legend_combined(axes[0],
                               coms=sorted({style5.CONFIGS[c][1] for c in grids}),
                               loc=args.legend_loc)
    else:
        style5.legend_two(axes[0],
                          coms=sorted({style5.CONFIGS[c][1] for c in grids}),
                          loc_gait="upper right", loc_com="upper left")

    note = ("T-speed: eligibility " +
            ("pass>0 ∧ nbhd≥0.8" if args.tier == "robust" else "pass_rate>0") +
            ", ranked by net_fwd_mean, per (config, μ)")
    if partial:
        pres = {c: grids[c].present["hip_off"] for c in partial}
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in p]})" for c, p in pres.items())
    style5.finish(
        fig, os.path.join(args.out, f"speed_vs_mu_{args.round}"
                          + (f"_top{args.top}" if args.top != TOP else "")
                          + ("_robust" if args.tier == "robust" else "")
                          + ".png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier=("robust (pass ∧ nbhd ≥ 0.8)" if args.tier == "robust"
              else "pass (pass_rate > 0)"),
        stat=(f"mean of top-{args.top} per (config, μ)" if args.single else
              f"left: mean of top-{args.top}; right: best-of-best (one cell)"),
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
