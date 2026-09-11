#!/usr/bin/env python
"""Duty factor figures (TIME-based, percent of gait cycle — not distance).

Per-foot duty factor D = fraction of the cycle the foot is on the ground,
measured as D = double_frac_t + single_frac_t / 2 (each foot is down for all
of double support plus half of single support in a symmetric alternating
gait). The support split is then reported through the symmetric-gait
identity Ben specified:

    double-support share = max(2D - 1, 0)   (any D > 0.5 is double support)
    single-support share = 1 - |2D - 1|
    aerial share         = max(1 - 2D, 0)   (D < 0.5: hopping, no double)

Two figures from the same data:
  duty_factor_bars_grid4.png       stacked single/double share, one panel
                                   per mu, x = config (duty.png style)
  duty_factor_violin_grid4_muXX.png  violin of per-foot D across the top-20
                                   finalists at one mu (--mu, default 0.1),
                                   kappa=0 blue / kappa=2 red, 0.5 line

Data: results/grid4_report/fig45_metrics.csv — the per-mu finalists RE-RUN
(selection = robust top-20: nbhd >= 0.8 AND pass, per mu), survivors only.
single_frac_t / double_frac_t are time fractions of the measurement window.

usage: python grid5/analysis/figs/duty_factor.py [--mu 0.1]
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import style5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
MUS = [0.1, 0.3, 0.5, 0.7]
CFGS = ["c1", "c2", "c3", "c4", "c5", "c6"]
C_SINGLE, C_DOUBLE = "#5b8db8", "#6aa96a"


def load_pools():
    """(cfg, mu) -> list of per-foot duty factors D (survivors only)."""
    path = os.path.join(_ROOT, "results", "grid4_report", "fig45_metrics.csv")
    pools = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(float(r["survived"])) != 1:
                    continue
                sf, df = float(r["single_frac_t"]), float(r["double_frac_t"])
                if not (np.isfinite(sf) and np.isfinite(df)):
                    continue
                D = df + sf / 2.0
                pools.setdefault((r["config"], float(r["mu"])), []).append(D)
            except (KeyError, ValueError):
                continue
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.1,
                    help="mu level for the violin figure")
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()
    pools = load_pools()

    # ---- figure 1: stacked single/double share, one panel per mu
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.6), sharey=True)
    x = np.arange(len(CFGS))
    for ax, mu in zip(axes, MUS):
        Dm = np.array([np.mean(pools.get((c, mu), [np.nan])) for c in CFGS])
        single = (1 - np.abs(2 * Dm - 1)) * 100
        double = np.clip(2 * Dm - 1, 0, 1) * 100
        air = np.clip(1 - 2 * Dm, 0, 1) * 100
        ax.bar(x, single, 0.62, color=C_SINGLE, label="single support")
        ax.bar(x, double, 0.62, bottom=single, color=C_DOUBLE,
               label="double support")
        ax.bar(x, air, 0.62, bottom=single + double, color="#c9c9c9",
               label="aerial")
        for xi, c in zip(x, CFGS):
            if not np.isfinite(Dm[xi]):
                ax.annotate("no data", (xi, 50), ha="center", rotation=90,
                            fontsize=7, color="crimson")
        ax.set_title(f"μ = {mu:g}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\nκ={style5.CONFIGS[c][0]:g}" for c in CFGS],
                           fontsize=8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("share of gait cycle [%]")
    axes[0].legend(fontsize=8, loc="lower left")
    style5.finish(
        fig, os.path.join(args.out, "duty_factor_bars_grid4.png"),
        K=1, tier="robust finalists (nbhd ≥ 0.8 ∧ pass, per-μ top-20), "
                  "survivors",
        stat="mean per-foot duty factor D → single = 2(1−D), double = 2D−1",
        note="time-based (single_frac_t/double_frac_t), symmetric-gait identity")

    # ---- figure 2: violin of D at one mu
    mu = args.mu
    data = [pools.get((c, mu), []) for c in CFGS]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    vp = ax.violinplot(data, positions=np.arange(len(CFGS)), widths=0.7,
                       showmedians=True, showextrema=True)
    for body, c in zip(vp["bodies"], CFGS):
        k = style5.CONFIGS[c][0]
        body.set_facecolor(style5.GAIT[k]["color"])
        body.set_alpha(0.55)
        body.set_edgecolor("black")
    for part in ("cmedians", "cmaxes", "cmins", "cbars"):
        vp[part].set_color("black"); vp[part].set_linewidth(1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1.2)
    ax.annotate("D > 0.5 ⇒ double support;  D < 0.5 ⇒ aerial phase",
                (-0.45, 0.502), fontsize=8, color="gray", ha="left",
                va="bottom")
    ax.set_xticks(np.arange(len(CFGS)))
    ax.set_xticklabels([style5.label_for(c) for c in CFGS], fontsize=8)
    ax.set_ylabel("per-foot duty factor D  (time fraction on ground)")
    ax.set_title(f"duty factor distribution @ μ={mu:g} — "
                 "blue κ=0, red κ=2", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    mt = f"{mu:g}".replace("0.", "0")
    style5.finish(
        fig, os.path.join(args.out, f"duty_factor_violin_grid4_mu{mt}.png"),
        K=1, tier="robust finalists (nbhd ≥ 0.8 ∧ pass, per-μ top-20), "
                  "survivors",
        stat=f"per-foot D = double_frac_t + single_frac_t/2, "
             f"top-20 pool per config",
        note="time-based; dotted line D=0.5")


if __name__ == "__main__":
    main()
