"""hw_compare.py — the hardware-model sweep on its own: c1 / c2 / c5 / c6 at mu 0.12 and 0.45.
Robust-region volume (count of robust cells, and fraction of the cells each config was
evaluated on -- the pruned candidate lists differ per config, so both are shown) and
forward speed (top-20 mean and champion; feedforward torso as bars, the kappa=2 PID loop
as x markers). Definitions as in hw_vs_grid5.py.

    python grid6/analysis/hw_compare.py            -> results/grid6_hw/figs/hw_robust.png, hw_speed.png
    python grid6/analysis/hw_compare.py --grid5    -> results/grid5_report/cross4/grid5_robust.png, grid5_speed.png
                                                      (same four configs, same bar layout, GRID-5 mu 0.1/0.3/0.5/0.7)
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hw_vs_grid5 as H                          # noqa: E402
import style5                                    # noqa: E402
import matplotlib.pyplot as plt                  # noqa: E402

OUT = H.OUT
CFGS = H.CFGS
MUS = H.HW_MU


def finish(fig, path, K, tier, stat, note=""):
    """style5.finish without the greyscale twin, and with room for a legend under the panels"""
    fig.text(0.5, 0.005, f"K={K}   tier: {tier}   {stat}   {note}", ha="center", fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.09, 1, 0.97])
    fig.savefig(path, dpi=style5.DPI)
    plt.close(fig)
    print(f"wrote {path}")


def bar_style(c):
    k, com = style5.CONFIGS[c]
    col = style5.GAIT[k]["color"]
    return dict(color=col if k == 0.0 else "white", edgecolor=col, linewidth=1.8,
                hatch="" if k == 0.0 else "//"), style5.COM_MK[com], col


def collect_hw():
    D = {}
    for c in CFGS:
        for mu in MUS:
            pas, v = H.hw_planes(c, mu, "ff")
            fr, n, den = H.robust_frac(H.nbhd_hw(pas))
            top, ch, ne = H.speed_track(pas, v)
            row = dict(rob_n=n, rob_den=den, rob_frac=fr, top=top, champ=ch, n_pass=ne, evaluated=int(np.isfinite(pas).sum()))
            if style5.CONFIGS[c][0] != 0.0:
                pp, vp = H.hw_planes(c, mu, "pid")
                row["pid_top"], row["pid_champ"], _ = H.speed_track(pp, vp)
            D[(c, mu)] = row
    return D


def collect_grid5():
    import load5
    grids = {c: load5.load(c, rnd="grid5", verbose=False) for c in CFGS}
    load5.compatible(grids.values())
    mus = [float(m) for m in grids["c1"].axes["mu"]]
    D = {}
    for c in CFGS:
        g = grids[c]
        nb = g.nbhd("pass_rate")
        for m, mu in enumerate(mus):
            fr, n, den = H.robust_frac(nb[m])
            top, ch, ne = H.speed_track(g["pass_rate"][m], g["net_fwd_mean"][m])
            D[(c, mu)] = dict(rob_n=n, rob_den=den, rob_frac=fr, top=top, champ=ch, n_pass=ne, evaluated=int(np.isfinite(g["pass_rate"][m]).sum()))
    return D, mus


def main():
    global MUS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid5", action="store_true")
    a = ap.parse_args()
    if a.grid5:
        D, MUS = collect_grid5()
        OUT = os.path.join(H.ROOT, "results", "grid5_report", "cross4")
        prefix, head = "grid5", "GRID-5 (ideal actuators, κ PID torso, slide COM models)"
        pass_note = "pass = survived ∧ head>0.5 ∧ net_fwd>0.05 (root); full 4,147,200-cell grid per config"
        speed_lab, rob_tier = "net_fwd_mean of the root [m/s]", "robust (nbhd-mean pass ≥ 0.8, freq ±0.04 × phi ±10)"
    else:
        D = collect_hw()
        prefix, head = "hw", "Hardware-model sweep (cap 354°/s, 56 ms torso lag, feedforward torso, CAD models)"
        pass_note = "pass = stood ∧ straight>0.5 ∧ v_net>0.05 (whole-body COM); grid pruned per config by hw_mask r=1"
        speed_lab, rob_tier = "v_net of the whole-body COM [m/s]", "robust (nbhd-mean pass ≥ 0.8, freq ±0.05 × phi ±10)"
    os.makedirs(OUT, exist_ok=True)
    print(f"{'cfg':4s} {'mu':>5s} {'evaluated':>9s} {'passers':>8s} {'robust':>7s} {'robust%':>8s} {'top20':>6s} {'champ':>6s} {'pid top20':>9s} {'pid champ':>9s}")
    for (c, mu), r in D.items():
        print(f"{c:4s} {mu:5.2f} {r['evaluated']:9d} {r['n_pass']:8d} {r['rob_n']:7d} {r['rob_frac']*100:7.1f}% {r['top']:6.3f} {r['champ']:6.3f} "
              f"{r.get('pid_top', float('nan')):9.3f} {r.get('pid_champ', float('nan')):9.3f}")

    W = 0.18
    x0 = np.arange(len(MUS))
    FW = 6.0 + 2.75 * len(MUS)
    def bars(ax, key, scale=1.0):
        for i, c in enumerate(CFGS):
            st, mk, col = bar_style(c)
            xs = x0 + (i - 1.5) * W
            ys = [D[(c, mu)][key] * scale for mu in MUS]
            ax.bar(xs, ys, W * 0.9, label=style5.label_for(c), **st)
            ax.plot(xs, ys, linestyle="none", marker=mk, ms=11, mfc="white" if st["hatch"] else col, mec=col, mew=1.8)
            for x, y in zip(xs, ys):                    # config name at the foot of every bar
                ax.annotate(c, (x, 0), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.5, color="white" if not st["hatch"] else col, fontweight="bold")
        ax.set_xticks(x0); ax.set_xticklabels([f"μ = {m:g}" for m in MUS])
        ax.grid(axis="y", alpha=0.3)

    # ---- robust region
    fig, axes = plt.subplots(1, 2, figsize=(FW, 5.4))
    bars(axes[0], "rob_n"); axes[0].set_ylabel("robust cells (count)")
    axes[0].set_title("robust-region volume", fontsize=10)
    bars(axes[1], "rob_frac", 100.0); axes[1].set_ylabel("robust cells / evaluated cells [%]")
    axes[1].set_title("same, as a fraction of each config's evaluated grid", fontsize=10)
    for i, c in enumerate(CFGS):                       # denominators, so the fraction panel is honest
        for j, mu in enumerate(MUS):
            r = D[(c, mu)]
            axes[1].annotate(f"n={r['rob_den']//1000}k", (x0[j] + (i - 1.5) * W, r["rob_frac"] * 100), fontsize=6, ha="center",
                             xytext=(0, 3), textcoords="offset points", color="0.4")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.suptitle(head, fontsize=11)
    finish(fig, os.path.join(OUT, f"{prefix}_robust.png"), K="1", tier=rob_tier,
           stat="count / fraction of evaluated cells", note=pass_note)

    # ---- speed
    fig, axes = plt.subplots(1, 2, figsize=(FW, 5.4), sharey=True)
    for ax, key, ttl in zip(axes, ("top", "champ"), ("top-20 mean of passing cells", "champion (best-of-best, one cell)")):
        bars(ax, key)
        for i, c in enumerate(CFGS):
            if style5.CONFIGS[c][0] == 0.0 or a.grid5:
                continue
            _, _, col = bar_style(c)
            ax.plot(x0 + (i - 1.5) * W, [D[(c, mu)][f"pid_{key}"] for mu in MUS], linestyle="none", marker="x", ms=9, mew=2.2, color="black")
        ax.set_title(ttl, fontsize=10)
    axes[0].set_ylabel(speed_lab)
    from matplotlib.lines import Line2D
    h0, l0 = axes[0].get_legend_handles_labels()
    if not a.grid5:
        h0.append(Line2D([], [], color="black", marker="x", ms=9, mew=2.2, linestyle="none")); l0.append("κ=2 PID torso loop (56 ms)")
    fig.legend(h0, l0, fontsize=8, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.035), frameon=False)
    fig.suptitle(head.split(" (")[0] + ": forward speed, per (config, μ) selection", fontsize=11)
    finish(fig, os.path.join(OUT, f"{prefix}_speed.png"), K="1", tier="pass",
           stat="left: mean of top-20; right: best-of-best", note=pass_note + ("" if a.grid5 else "; bars = feedforward torso, × = κ=2 PID loop"))


if __name__ == "__main__":
    main()
