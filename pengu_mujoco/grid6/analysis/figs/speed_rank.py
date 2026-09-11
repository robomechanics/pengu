#!/usr/bin/env python
"""Speed vs selection rank at one mu — is a config's top-N a plateau or a
spike? One line per config: net_fwd_mean of the rank-r passing cell,
r = 1..N, per-(config, mu) T-speed ranking (pass_rate > 0, net_fwd_mean
descending). A curve that collapses within the first few ranks is a handful
of isolated cells; a flat curve is a broad fast region. Read next to the
robust-region figure (trap T8: best-of-best is one cell).

usage:
  python grid5/analysis/figs/speed_rank.py --mu 0.3            # grid4 ref
  python grid5/analysis/figs/speed_rank.py --round grid5 --mu 0.1
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # pengu_mujoco/
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid4", choices=["grid4", "grid5"])
    ap.add_argument("--mu", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--configs", nargs="*", default=None)
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

    g0 = next(iter(grids.values()))
    m = g0.axes["mu"].index(args.mu)
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for c, g in grids.items():
        k, com = style5.CONFIGS[c]
        nf = g["net_fwd_mean"][m]
        vals = np.sort(nf[(g["pass_rate"][m] > 0) & np.isfinite(nf)])[::-1]
        n = min(args.n, vals.size)
        if n == 0:
            print(f"  {c}: no passers at mu={args.mu}")
            continue
        st = style5.style_for(k, com)
        st["ms"] = 7                      # markers every ~decade on a log axis
        ax.plot(np.arange(1, n + 1), vals[:n], **st, markevery=0.12,
                label=style5.label_for(c)
                + (" [PARTIAL]" if not g.complete else ""))
    ax.set_xscale("log")
    ax.set_xlabel("rank in per-μ T-speed selection")
    ax.set_ylabel("net_fwd_mean [m/s]")
    ax.set_title(f"{args.round.upper()} speed vs rank @ μ={args.mu:g} "
                 f"(spike vs plateau)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.axvline(20, color="gray", lw=0.8, ls=":")
    ax.annotate("top-20", (20, ax.get_ylim()[1]), fontsize=7, color="gray",
                ha="center", va="top")
    style5.legend_two(ax, coms=sorted({style5.CONFIGS[c][1] for c in grids}),
                      loc_gait="upper right", loc_com="lower left")

    note = "T-speed: pass_rate>0, ranked by net_fwd_mean, per (config, μ)"
    if partial:
        pres = {c: grids[c].present["hip_off"] for c in partial}
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in p]})" for c, p in pres.items())
    mt = f"{args.mu:g}".replace("0.", "0")
    style5.finish(
        fig, os.path.join(args.out, f"speed_rank_{args.round}_mu{mt}.png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier="pass (pass_rate > 0)",
        stat=f"per-cell net_fwd_mean by rank, top {args.n}",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
